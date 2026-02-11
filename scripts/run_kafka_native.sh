#!/bin/bash
# Script to run Kafka natively (since Docker is unavailable)

# Move to the project root directory
cd "$(dirname "$0")/.." || exit 1

KAFKA_VER="3.6.1"
SCALA_VER="2.13"
KAFKA_TGZ="kafka_${SCALA_VER}-${KAFKA_VER}.tgz"
KAFKA_DIR="kafka_${SCALA_VER}-${KAFKA_VER}"
URL="https://archive.apache.org/dist/kafka/${KAFKA_VER}/${KAFKA_TGZ}"

# 1. Download if not exists
sudo apt-get update && sudo apt-get install -y librdkafka-dev
if [ ! -d "$KAFKA_DIR" ]; then
    echo "Downloading Kafka $KAFKA_VER..."
    if [ ! -f "$KAFKA_TGZ" ]; then
        wget "$URL" -O "$KAFKA_TGZ"
    fi
    echo "Extracting..."
    tar -xzf "$KAFKA_TGZ"
fi

# 2. Cleanup previous logs
rm -rf /tmp/kafka-logs /tmp/zookeeper

# 3. Start Zookeeper
echo "Starting Zookeeper..."
mkdir -p telemetry
"$KAFKA_DIR/bin/zookeeper-server-start.sh" "$KAFKA_DIR/config/zookeeper.properties" > telemetry/zookeeper.log 2>&1 &
ZOOKEEPER_PID=$!
sleep 5

# 4. Start Kafka Broker
echo "Starting Kafka Broker..."
"$KAFKA_DIR/bin/kafka-server-start.sh" "$KAFKA_DIR/config/server.properties" > telemetry/kafka.log 2>&1 &
KAFKA_PID=$!
sleep 5

echo "==================================================="
echo "Kafka running at localhost:9092"
echo "Zookeeper PID: $ZOOKEEPER_PID"
echo "Kafka     PID: $KAFKA_PID"
echo "==================================================="
echo "Press Ctrl+C to stop"

trap "kill $KAFKA_PID $ZOOKEEPER_PID; exit" INT TERM

wait
