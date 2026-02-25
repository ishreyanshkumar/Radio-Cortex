#!/bin/bash
# Simplified starter script that wraps run_kafka_native.sh or starts existing installation

# Move to the project root directory and get absolute path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

KAFKA_VER="3.6.1"
SCALA_VER="2.13"
KAFKA_DIR="kafka_${SCALA_VER}-${KAFKA_VER}"
KAFKA_PATH="$PROJECT_ROOT/$KAFKA_DIR"

if [ ! -d "$KAFKA_PATH" ]; then
    echo "Kafka not found in $KAFKA_PATH"
    echo "Running native setup script..."
    bash "$PROJECT_ROOT/scripts/run_kafka_native.sh"
    exit $?
fi

# 1. Cleanup previous logs
echo "Cleaning up temporary logs..."
rm -rf /tmp/kafka-logs /tmp/zookeeper 2>/dev/null || true

# Create a symlink without spaces to avoid bash path expansion issues with Kafka scripts
KAFKA_LINK="/tmp/rc_kafka_link"
rm -f "$KAFKA_LINK" 2>/dev/null || true
ln -s "$KAFKA_PATH" "$KAFKA_LINK"

# 2. Start Zookeeper
echo "Starting Zookeeper..."
mkdir -p "$PROJECT_ROOT/logs"
nohup "$KAFKA_LINK/bin/zookeeper-server-start.sh" "$KAFKA_LINK/config/zookeeper.properties" > "$PROJECT_ROOT/logs/zookeeper.log" 2>&1 &
ZOOKEEPER_PID=$!
echo "Zookeeper PID: $ZOOKEEPER_PID"
sleep 5

# 3. Start Kafka Broker
echo "Starting Kafka Broker..."
nohup "$KAFKA_LINK/bin/kafka-server-start.sh" "$KAFKA_LINK/config/server.properties" > "$PROJECT_ROOT/logs/kafka.log" 2>&1 &
KAFKA_PID=$!
echo "Kafka PID: $KAFKA_PID"

echo "Waiting for Kafka Broker to become ready..."
MAX_RETRIES=30
RETRY=0
# Wait for port 9092 to be open
until (echo > /dev/tcp/localhost/9092) >/dev/null 2>&1 || [ $RETRY -eq $MAX_RETRIES ]; do
  sleep 1
  RETRY=$((RETRY+1))
  printf "."
done

echo "" # Newline after loop

if [ $RETRY -eq $MAX_RETRIES ]; then
  echo "✗ Kafka failed to start within 30 seconds. Check logs/kafka.log"
  # Try to kill processes if failed
  kill $KAFKA_PID $ZOOKEEPER_PID 2>/dev/null
  exit 1
else
  echo "✓ Kafka is ready. Logs in logs/"
  echo "==================================================="
  echo "Kafka running at localhost:9092"
  echo "Project Root:  $PROJECT_ROOT"
  echo "Zookeeper PID: $ZOOKEEPER_PID"
  echo "Kafka     PID: $KAFKA_PID"
  echo "==================================================="
  echo "Use 'pkill -f kafka' to stop"
fi
