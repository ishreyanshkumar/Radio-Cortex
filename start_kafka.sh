#!/bin/bash


SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KAFKA_DIR="$SCRIPT_DIR/kafka_2.13-3.6.1"
LOG_DIR="$SCRIPT_DIR/telemetry"
mkdir -p "$LOG_DIR"

if [ ! -x "$KAFKA_DIR/bin/kafka-server-start.sh" ]; then
	echo "Kafka binaries not found in $KAFKA_DIR. Run ./run_kafka_native.sh first." >&2
	exit 1
fi

# Ensure a clean state before starting
echo "Ensuring no zombie sessions..."
rm -rf /tmp/kafka-logs /tmp/zookeeper 2>/dev/null || true

echo "Starting Zookeeper..."
nohup "$KAFKA_DIR/bin/zookeeper-server-start.sh" "$KAFKA_DIR/config/zookeeper.properties" > "$LOG_DIR/zookeeper.log" 2>&1 &
echo "Waiting for Zookeeper..."
sleep 8

echo "Starting Kafka Broker..."
nohup "$KAFKA_DIR/bin/kafka-server-start.sh" "$KAFKA_DIR/config/server.properties" > "$LOG_DIR/kafka.log" 2>&1 &

echo "Waiting for Kafka Broker to become ready..."
MAX_RETRIES=30
RETRY=0
until (echo > /dev/tcp/localhost/9092) >/dev/null 2>&1 || [ $RETRY -eq $MAX_RETRIES ]; do
  sleep 1
  RETRY=$((RETRY+1))
  printf "."
done

if [ $RETRY -eq $MAX_RETRIES ]; then
  echo -e "\n✗ Kafka failed to start within 30 seconds. Check $LOG_DIR/kafka.log"
  exit 1
else
  echo -e "\n✓ Kafka is ready. Logs: $LOG_DIR/zookeeper.log, $LOG_DIR/kafka.log"
fi
