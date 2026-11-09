#!/usr/bin/env python3
"""Embedding worker HTTP server for Anamesis.

Uses sentence-transformers (all-MiniLM-L6-v2) to generate embeddings.
Anamesis calls this via --embedding-worker-url.
"""

import json
import os
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Force single-threaded operation to avoid torch multiprocessing crashes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
model = None
model_lock = threading.Lock()


class EmbedHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global model
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
            text = data.get("text", "")
        except (json.JSONDecodeError, KeyError):
            self.send_error(400, "invalid JSON")
            return

        if not text:
            self.send_error(400, "missing 'text' field")
            return

        with model_lock:
            if model is None:
                model = SentenceTransformer(MODEL_NAME)
            embedding = model.encode([text], show_progress_bar=False)[0].tolist()

        response = json.dumps({"embedding": embedding}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(),
                                                 self.log_date_time_string(),
                                                 format % args))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(("127.0.0.1", port), EmbedHandler)
    print(f"Embedding worker listening on 127.0.0.1:{port} (model: {MODEL_NAME})",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
