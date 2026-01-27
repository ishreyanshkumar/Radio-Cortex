#!/bin/bash
KAFKA_DIR=/home/hp/Radio-Cortex/kafka_2.13-3.6.1

echo "Stopping Kafka Broker..."
$KAFKA_DIR/bin/kafka-server-stop.sh

echo "Stopping Zookeeper..."
$KAFKA_DIR/bin/zookeeper-server-stop.sh

echo "Cleaning up any remaining processes..."
pkill -f kafka.Kafka || true
pkill -f org.apache.zookeeper.server.quorum.QuorumPeerMain || true

echo "Kafka and Zookeeper stopped."
