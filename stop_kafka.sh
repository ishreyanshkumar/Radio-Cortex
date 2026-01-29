#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KAFKA_DIR="$SCRIPT_DIR/kafka_2.13-3.6.1"

if [ ! -x "$KAFKA_DIR/bin/kafka-server-stop.sh" ]; then
	echo "Kafka binaries not found in $KAFKA_DIR. Nothing to stop." >&2
	exit 0
fi

echo "Stopping Kafka Broker..."
"$KAFKA_DIR/bin/kafka-server-stop.sh"

echo "Stopping Zookeeper..."
"$KAFKA_DIR/bin/zookeeper-server-stop.sh"

echo "Cleaning up any remaining processes..."
pkill -f kafka.Kafka || true
pkill -f org.apache.zookeeper.server.quorum.QuorumPeerMain || true

echo "Kafka and Zookeeper stopped."
