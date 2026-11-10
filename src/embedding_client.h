#pragma once

// HTTP client for an external embedding worker (llama-server --embedding).
// No ML dependencies: this is a plain HTTP + JSON codec.

#include <cstddef>
#include <string>
#include <vector>

namespace embedding_client {

struct Url {
    std::string host;
    int port = 80;
    std::string path = "/embedding";
};

// Parse http://host[:port][/path]. Returns false on a malformed URL.
bool parse_url(const std::string& url, Url& out);

// RFC 8259 string escaping (quotes, backslashes, control characters).
std::string json_escape(const std::string& text);

// Body for POST /embedding: {"input":"<text>","embd_normalize":2}
std::string build_request_body(const std::string& text, int embd_normalize = 2);

// POST a JSON body and read the full response body.
// Returns true on HTTP 2xx; out_body holds the (dechunked) body either way.
bool http_post_json(const std::string& url, const std::string& body,
                    int timeout_ms, std::string& out_body, std::string& err);

// Accepts llama.cpp legacy  [{"index":0,"embedding":[[...]]}]
//                and OpenAI  {"data":[{"embedding":[...]}]}
// Also accepts {"embedding":[...]} / {"embedding":[[...]]}.
// Rejects empty, non-finite, or (when expected_dim != 0) wrong-sized vectors.
bool parse_embedding_response(const std::string& body,
                              std::vector<float>& out_embedding,
                              std::string& err,
                              std::size_t expected_dim = 0);

// Convenience: build + POST + parse in one call.
bool embed_text(const std::string& url, const std::string& text,
                int timeout_ms, std::vector<float>& out_embedding,
                std::string& err, std::size_t expected_dim = 0);

}  // namespace embedding_client
