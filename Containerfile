FROM alpine:3.20

RUN apk add --no-cache ca-certificates

WORKDIR /anamesis

# Copy the static binary
COPY anamesis /usr/local/bin/anamesis

# Non-root user
RUN adduser -D -s /sbin/nologin anamesis
USER anamesis

# Default cache port and admin port
EXPOSE 6379 8080

# Default AOF path inside container
ENV SEM_CACHE_PORT=6379 \
    SEM_CACHE_ADMIN_PORT=8080 \
    SEM_CACHE_MEMORY_MB=1024 \
    SEM_CACHE_AOF_INTERVAL_SEC=60 \
    SEM_CACHE_POLICY=sieve \
    SEM_CACHE_TTL_DEFAULT_SEC=86400 \
    SEM_CACHE_SEMANTIC_THRESHOLD=0.80 \
    SEM_CACHE_AOF_FILE=/data/anamesis.aof

VOLUME ["/data"]

ENTRYPOINT ["/usr/local/bin/anamesis"]
CMD ["--port", "6379", "--admin-port", "8080", "--memory-limit", "1024", "--aof-file", "/data/anamesis.aof", "--no-aof"]
