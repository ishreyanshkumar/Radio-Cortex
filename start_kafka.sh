#!/bin/bash


SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KAFKA_DIR="$SCRIPT_DIR/kafka_2.13-3.6.1"
LOG_DIR="$SCRIPT_DIR"

if [ ! -x "$KAFKA_DIR/bin/kafka-server-start.sh" ]; then
	echo "Kafka binaries not found in $KAFKA_DIR. Run ./run_kafka_native.sh first." >&2
	exit 1
fi

echo "Starting Zookeeper..."
nohup "$KAFKA_DIR/bin/zookeeper-server-start.sh" "$KAFKA_DIR/config/zookeeper.properties" > "$LOG_DIR/zookeeper.log" 2>&1 &
echo "Waiting for Zookeeper..."
sleep 5
echo "Starting Kafka Broker..."
nohup "$KAFKA_DIR/bin/kafka-server-start.sh" "$KAFKA_DIR/config/server.properties" > "$LOG_DIR/kafka.log" 2>&1 &
echo "Kafka started. Logs: $LOG_DIR/zookeeper.log, $LOG_DIR/kafka.log"
