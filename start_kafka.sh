#!/bin/bash
KAFKA_DIR=/home/hp/Radio-Cortex/kafka_2.13-3.6.1
echo "Starting Zookeeper..."
nohup $KAFKA_DIR/bin/zookeeper-server-start.sh $KAFKA_DIR/config/zookeeper.properties > zookeeper.log 2>&1 &
echo "Waiting for Zookeeper..."
sleep 5
echo "Starting Kafka Broker..."
nohup $KAFKA_DIR/bin/kafka-server-start.sh $KAFKA_DIR/config/server.properties > kafka.log 2>&1 &
echo "Kafka started."
