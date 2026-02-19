# Stage 1: Build ns-3
FROM ubuntu:24.04 AS ns3-builder

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install Build Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    librdkafka-dev \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Clone ns-3-allinone (Specific version)
# We use the same version as the setup script
RUN git clone https://gitlab.com/nsnam/ns-3-allinone.git && \
    cd ns-3-allinone && \
    ./download.py && \
    ln -s ns-3.* ns-3-dev

# 3. Copy Scenario Files & Link them
COPY . /app

RUN cd ns-3-allinone/ns-3-dev/scratch && \
    rm -rf * && \
    ln -sf /app/oran-congestion-scenario.cc . && \
    ln -sf /app/CMakeLists.txt .

# 4. Configure and Build ns-3
# Warning: This takes a significant amount of time
WORKDIR /app/ns-3-allinone/ns-3-dev
RUN ./ns3 configure -d optimized --enable-examples --enable-tests && \
    ./ns3 build

# Stage 2: Runtime Environment
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/venv/bin:$PATH"
ENV NS3_PATH="/app/ns-3-allinone/ns-3-dev"

# 1. Install Runtime Dependencies
# Note: librdkafka1 is the runtime library, librdkafka-dev is headers
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    librdkafka1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Set up Python Virtual Environment
RUN python3 -m venv /app/venv

# 3. Install Python Dependencies
# We copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Built ns-3 from Builder Stage
COPY --from=ns3-builder /app/ns-3-allinone /app/ns-3-allinone

# 5. Copy Application Code
COPY . /app

# 6. Set Environment Variables for ns-3
# Ensure ns-3 libraries are found
ENV LD_LIBRARY_PATH="${NS3_PATH}/build/lib:${LD_LIBRARY_PATH}"

# 7. Expose Gradio UI Port
EXPOSE 7860

# 8. Default Command
# Uses the environment variable KAFKA_BOOTSTRAP which corresponds to the service name in docker-compose
ENV KAFKA_BOOTSTRAP="kafka:9092"

CMD ["python3", "radio_cortex_complete.py", "--mode", "train", "--scenario", "all", "--n-envs", "4", "--total-timesteps", "100000"]
