# syntax=docker/dockerfile:1

FROM ubuntu:23.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages ucall usearch

WORKDIR /usearch
COPY python/usearch/server.py server.py

CMD ["python3", "./server.py"]
EXPOSE 8545
