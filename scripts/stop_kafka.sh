#!/bin/bash
# Move to the project root directory
cd "$(dirname "$0")/.." || exit 1

KAFKA_DIR="kafka_2.13-3.6.1"

if [ ! -x "$KAFKA_DIR/bin/kafka-server-stop.sh" ]; then
	echo "Kafka binaries not found in $KAFKA_DIR. Nothing to stop." >&2
	exit 0
fi

echo "Stopping Kafka Broker..."
"$KAFKA_DIR/bin/kafka-server-stop.sh"
sleep 2

echo "Stopping Zookeeper..."
"$KAFKA_DIR/bin/zookeeper-server-stop.sh"
sleep 2

echo "Cleaning up any remaining processes..."
pkill -f kafka.Kafka || true
pkill -f org.apache.zookeeper.server.quorum.QuorumPeerMain || true

echo "Wiping temporary data for clean start..."
rm -rf /tmp/kafka-logs /tmp/zookeeper

echo "Kafka and Zookeeper stopped."
