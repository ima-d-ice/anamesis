#include "embedding_client.h"

#include "epoll_compat.h"
#include "third_party/nlohmann/json.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

using namespace std;
using json = nlohmann::json;

namespace embedding_client {
namespace {

using Clock = chrono::steady_clock;

int remaining_ms(Clock::time_point deadline) {
    auto left = chrono::duration_cast<chrono::milliseconds>(deadline - Clock::now()).count();
    if (left < 0) {
        return 0;
    }
    return static_cast<int>(left);
}

bool wait_fd(int fd, bool for_write, int timeout_ms) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    timeval tv{};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    int rc = for_write ? select(fd + 1, nullptr, &fds, nullptr, &tv)
                       : select(fd + 1, &fds, nullptr, nullptr, &tv);
    return rc > 0;
}

int connect_with_timeout(const Url& url, Clock::time_point deadline, string& err) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    addrinfo* res = nullptr;
    string port_str = to_string(url.port);
    int rc = getaddrinfo(url.host.c_str(), port_str.c_str(), &hints, &res);
    if (rc != 0) {
        err = string("resolve failed: ") + gai_strerror(rc);
        return -1;
    }

    int sock = -1;
    for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
        sock = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (sock < 0) {
            continue;
        }
        COMPAT_SET_NOSIGPIPE(sock);
        int flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, flags | O_NONBLOCK);

        rc = ::connect(sock, ai->ai_addr, ai->ai_addrlen);
        if (rc < 0 && errno == EINPROGRESS) {
            int wait_ms = remaining_ms(deadline);
            if (wait_fd(sock, true, wait_ms)) {
                int soerr = 0;
                socklen_t len = sizeof(soerr);
                if (getsockopt(sock, SOL_SOCKET, SO_ERROR, &soerr, &len) == 0 && soerr == 0) {
                    break;
                }
            }
        } else if (rc == 0) {
            break;
        }
        close(sock);
        sock = -1;
    }
    freeaddrinfo(res);

    if (sock < 0) {
        if (err.empty()) {
            err = "connect timed out or refused";
        }
        return -1;
    }
    return sock;
}

// Split raw HTTP into (status, body). Handles Content-Length and chunked.
bool extract_http(const string& raw, int& status, string& body, bool& complete) {
    complete = false;
    size_t hdr_end = raw.find("\r\n\r\n");
    size_t skip = 4;
    if (hdr_end == string::npos) {
        hdr_end = raw.find("\n\n");
        skip = 2;
    }
    if (hdr_end == string::npos) {
        return false;
    }

    string head = raw.substr(0, hdr_end);
    size_t body_start = hdr_end + skip;

    size_t line_end = head.find('\n');
    string status_line = line_end == string::npos ? head : head.substr(0, line_end);
    if (status_line.compare(0, 5, "HTTP/") != 0) {
        return false;
    }
    size_t sp = status_line.find(' ');
    if (sp == string::npos) {
        return false;
    }
    status = 0;
    for (size_t i = sp + 1; i < status_line.size() && i < sp + 4; ++i) {
        if (status_line[i] < '0' || status_line[i] > '9') {
            break;
        }
        status = status * 10 + (status_line[i] - '0');
    }
    if (status < 100) {
        return false;
    }

    // Header lookup (case-insensitive).
    auto header = [&](const string& name) -> string {
        string lower_head = head;
        transform(lower_head.begin(), lower_head.end(), lower_head.begin(),
                  [](unsigned char c) { return static_cast<char>(tolower(c)); });
        size_t pos = lower_head.find("\r\n" + name);
        if (pos == string::npos) {
            return "";
        }
        size_t vstart = pos + 2 + name.size();
        if (vstart >= lower_head.size() || lower_head[vstart] != ':') {
            return "";
        }
        vstart += 1;
        while (vstart < lower_head.size() && lower_head[vstart] == ' ') {
            ++vstart;
        }
        size_t vend = lower_head.find("\r\n", vstart);
        if (vend == string::npos) {
            vend = lower_head.size();
        }
        return head.substr(vstart, vend - vstart);
    };

    string clen = header("content-length");
    string te = header("transfer-encoding");
    bool chunked = te.find("chunked") != string::npos;

    string payload = raw.substr(body_start);

    if (chunked) {
        // Find the terminating zero-length chunk.
        size_t p = 0;
        bool done = false;
        while (true) {
            size_t eol = payload.find("\r\n", p);
            if (eol == string::npos) {
                break;
            }
            string size_line = payload.substr(p, eol - p);
            size_t semi = size_line.find(';');
            if (semi != string::npos) {
                size_line = size_line.substr(0, semi);
            }
            size_t chunk_size = 0;
            try {
                chunk_size = static_cast<size_t>(stoul(size_line, nullptr, 16));
            } catch (const exception&) {
                break;
            }
            p = eol + 2;
            if (chunk_size == 0) {
                done = true;
                break;
            }
            if (payload.size() < p + chunk_size + 2) {
                break;
            }
            p += chunk_size + 2;
        }
        if (!done) {
            return true;  // headers ok, body still arriving
        }
        // Dechunk.
        body.clear();
        p = 0;
        while (true) {
            size_t eol = payload.find("\r\n", p);
            if (eol == string::npos) {
                break;
            }
            string size_line = payload.substr(p, eol - p);
            size_t semi = size_line.find(';');
            if (semi != string::npos) {
                size_line = size_line.substr(0, semi);
            }
            size_t chunk_size = 0;
            try {
                chunk_size = static_cast<size_t>(stoul(size_line, nullptr, 16));
            } catch (const exception&) {
                break;
            }
            p = eol + 2;
            if (chunk_size == 0) {
                break;
            }
            if (payload.size() < p + chunk_size) {
                break;
            }
            body.append(payload, p, chunk_size);
            p += chunk_size + 2;
        }
        complete = true;
        return true;
    }

    if (!clen.empty()) {
        size_t want = 0;
        try {
            want = static_cast<size_t>(stoul(clen));
        } catch (const exception&) {
            return false;
        }
        if (payload.size() < want) {
            return true;  // headers ok, body still arriving
        }
        body.assign(payload, 0, want);
        complete = true;
        return true;
    }

    // No framing: wait for connection close.
    body = payload;
    return true;
}

}  // namespace

bool parse_url(const string& url, Url& out) {
    string rest = url;
    if (rest.compare(0, 7, "http://") == 0) {
        rest = rest.substr(7);
    } else if (rest.compare(0, 8, "https://") == 0) {
        return false;  // plain HTTP client only
    }
    if (rest.empty()) {
        return false;
    }

    string authority;
    size_t slash = rest.find('/');
    if (slash != string::npos) {
        authority = rest.substr(0, slash);
        out.path = rest.substr(slash);
    } else {
        authority = rest;
        out.path = "/embedding";
    }
    if (authority.empty()) {
        return false;
    }

    // [host]:port] — IPv6 literals in brackets are not supported.
    size_t colon = authority.rfind(':');
    if (colon != string::npos && authority.find(':') == colon) {
        string port_str = authority.substr(colon + 1);
        if (port_str.empty()) {
            return false;
        }
        try {
            out.port = stoi(port_str);
        } catch (const exception&) {
            return false;
        }
        out.host = authority.substr(0, colon);
    } else {
        out.host = authority;
        out.port = 80;
    }
    return !out.host.empty() && out.port > 0 && out.port < 65536;
}

string json_escape(const string& text) {
    string out;
    out.reserve(text.size() + 8);
    for (unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

string build_request_body(const string& text, int embd_normalize) {
    return "{\"input\":\"" + json_escape(text) + "\",\"embd_normalize\":" +
           to_string(embd_normalize) + "}";
}

bool http_post_json(const string& url, const string& body,
                    int timeout_ms, string& out_body, string& err) {
    out_body.clear();
    err.clear();

    Url parsed;
    if (!parse_url(url, parsed)) {
        err = "bad url: " + url;
        return false;
    }
    if (timeout_ms <= 0) {
        timeout_ms = 5000;
    }

    auto deadline = Clock::now() + chrono::milliseconds(timeout_ms);
    int sock = connect_with_timeout(parsed, deadline, err);
    if (sock < 0) {
        return false;
    }

    string host_hdr = parsed.host;
    if (parsed.port != 80) {
        host_hdr += ":" + to_string(parsed.port);
    }
    string request =
        "POST " + parsed.path + " HTTP/1.1\r\n"
        "Host: " + host_hdr + "\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + to_string(body.size()) + "\r\n"
        "Accept: application/json\r\n"
        "Connection: close\r\n"
        "\r\n" + body;

    // Send (loop: partial writes happen).
    size_t sent = 0;
    while (sent < request.size()) {
        int wait_ms = remaining_ms(deadline);
        if (wait_ms <= 0) {
            err = "send timed out";
            close(sock);
            return false;
        }
        if (!wait_fd(sock, true, wait_ms)) {
            err = "send timed out";
            close(sock);
            return false;
        }
        ssize_t n = send(sock, request.data() + sent, request.size() - sent,
                         COMPAT_MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                continue;
            }
            err = string("send failed: ") + strerror(errno);
            close(sock);
            return false;
        }
        sent += static_cast<size_t>(n);
    }

    // Read until the framed body is complete or the peer closes.
    string raw;
    char buf[8192];
    int status = 0;
    bool got_head = false;
    bool complete = false;
    string body_out;
    while (true) {
        int wait_ms = remaining_ms(deadline);
        if (wait_ms <= 0) {
            break;
        }
        if (!wait_fd(sock, false, wait_ms)) {
            break;
        }
        ssize_t n = recv(sock, buf, sizeof(buf), 0);
        if (n == 0) {
            break;  // peer closed
        }
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                continue;
            }
            err = string("recv failed: ") + strerror(errno);
            close(sock);
            return false;
        }
        raw.append(buf, static_cast<size_t>(n));
        got_head = extract_http(raw, status, body_out, complete);
        if (got_head && complete) {
            break;
        }
    }
    close(sock);

    if (!got_head) {
        err = "no HTTP response";
        return false;
    }
    // Without Content-Length the body arrives only at close; re-extract.
    if (!complete) {
        got_head = extract_http(raw, status, body_out, complete);
    }
    if (!got_head || !complete) {
        err = "incomplete HTTP response";
        return false;
    }

    out_body = body_out;
    if (status < 200 || status >= 300) {
        err = "http status " + to_string(status);
        string snippet = out_body.substr(0, 200);
        if (!snippet.empty()) {
            err += ": " + snippet;
        }
        return false;
    }
    return true;
}

namespace {

bool parse_float_vector(const json& arr, vector<float>& out, string& err,
                        size_t expected_dim) {
    if (!arr.is_array() || arr.empty()) {
        err = "embedding is empty";
        return false;
    }
    const json* flat = &arr;
    if (arr[0].is_array()) {
        // Legacy llama.cpp: one row per pooled result, [[...]].
        if (arr[0].empty()) {
            err = "embedding row is empty";
            return false;
        }
        flat = &arr[0];
    } else if (!arr[0].is_number()) {
        err = "embedding is not numeric";
        return false;
    }

    out.clear();
    out.reserve(flat->size());
    for (const auto& el : *flat) {
        if (!el.is_number()) {
            err = "embedding element is not a number";
            return false;
        }
        float v = static_cast<float>(el.get<double>());
        if (!isfinite(v)) {
            err = "embedding contains a non-finite value";
            return false;
        }
        out.push_back(v);
    }
    if (out.empty()) {
        err = "embedding is empty";
        return false;
    }
    if (expected_dim != 0 && out.size() != expected_dim) {
        err = "expected " + to_string(expected_dim) + "-dim embedding, got " +
              to_string(out.size());
        return false;
    }
    return true;
}

}  // namespace

bool parse_embedding_response(const string& body, vector<float>& out_embedding,
                              string& err, size_t expected_dim) {
    out_embedding.clear();
    err.clear();

    json j;
    try {
        j = json::parse(body);
    } catch (const json::exception& e) {
        err = string("json parse failed: ") + e.what();
        return false;
    }

    if (j.is_object() && j.contains("error")) {
        const auto& e = j["error"];
        if (e.is_object() && e.contains("message")) {
            err = "worker error: " + e["message"].dump();
        } else {
            err = "worker error: " + e.dump();
        }
        return false;
    }

    try {
        if (j.is_array()) {
            if (j.empty()) {
                err = "empty response array";
                return false;
            }
            if (!j[0].is_object() || !j[0].contains("embedding")) {
                err = "no embedding field in response";
                return false;
            }
            return parse_float_vector(j[0]["embedding"], out_embedding, err,
                                      expected_dim);
        }
        if (j.is_object()) {
            if (j.contains("data") && j["data"].is_array() && !j["data"].empty()) {
                const auto& first = j["data"][0];
                if (!first.is_object() || !first.contains("embedding")) {
                    err = "no embedding field in data[0]";
                    return false;
                }
                return parse_float_vector(first["embedding"], out_embedding, err,
                                          expected_dim);
            }
            if (j.contains("embedding")) {
                return parse_float_vector(j["embedding"], out_embedding, err,
                                          expected_dim);
            }
            err = "no embedding field in response";
            return false;
        }
        err = "unexpected response type";
        return false;
    } catch (const json::exception& e) {
        err = string("json access failed: ") + e.what();
        return false;
    }
}

bool embed_text(const string& url, const string& text, int timeout_ms,
                vector<float>& out_embedding, string& err, size_t expected_dim) {
    string body = build_request_body(text);
    string resp;
    if (!http_post_json(url, body, timeout_ms, resp, err)) {
        return false;
    }
    return parse_embedding_response(resp, out_embedding, err, expected_dim);
}

}  // namespace embedding_client
