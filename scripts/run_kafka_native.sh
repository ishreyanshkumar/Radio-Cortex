#!/bin/bash
# Script to run Kafka natively (since Docker is unavailable)

# Move to the project root directory and get absolute path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

KAFKA_VER="3.6.1"
SCALA_VER="2.13"
KAFKA_TGZ="kafka_${SCALA_VER}-${KAFKA_VER}.tgz"
KAFKA_DIR="kafka_${SCALA_VER}-${KAFKA_VER}"
KAFKA_PATH="$PROJECT_ROOT/$KAFKA_DIR"
URL="https://archive.apache.org/dist/kafka/${KAFKA_VER}/${KAFKA_TGZ}"

# 1. Install dependencies
if ! dpkg-query -W -f='${Status}' librdkafka-dev 2>/dev/null | grep -q "ok installed"; then
    echo "Installing librdkafka-dev..."
    sudo apt-get update && sudo apt-get install -y librdkafka-dev
fi

if ! command -v java &> /dev/null; then
    echo "Java not found. Installing default-jre..."
    sudo apt-get update && sudo apt-get install -y default-jre
fi

# 2. Download and Extract Kafka if not valid
if [ ! -d "$KAFKA_PATH" ] || [ ! -x "$KAFKA_PATH/bin/kafka-server-start.sh" ]; then
    echo "Kafka binaries not found or incomplete in $KAFKA_PATH. Preparing..."
    
    # Remove potentially corrupted directory
    rm -rf "$KAFKA_PATH"

    if [ ! -f "$KAFKA_TGZ" ]; then
        echo "Downloading Kafka $KAFKA_VER..."
        wget "$URL" -O "$KAFKA_TGZ" || { echo "Download failed"; exit 1; }
    fi

    echo "Extracting..."
    tar -xzf "$KAFKA_TGZ" || { 
        echo "Extraction failed. Deleting corrupted archive."
        rm -f "$KAFKA_TGZ"
        exit 1
    }
    
    # Final check after extraction
    if [ ! -x "$KAFKA_PATH/bin/kafka-server-start.sh" ]; then
        echo "Error: Extraction did not produce expected binaries in $KAFKA_PATH"
        exit 1
    fi
fi

# 3. Cleanup previous logs
echo "Cleaning up temporary logs..."
rm -rf /tmp/kafka-logs /tmp/zookeeper

# Create a symlink without spaces to avoid bash path expansion issues with Kafka scripts
KAFKA_LINK="/tmp/rc_kafka_link"
rm -f "$KAFKA_LINK"
ln -s "$KAFKA_PATH" "$KAFKA_LINK"

# 4. Start Zookeeper
echo "Starting Zookeeper..."
mkdir -p "$PROJECT_ROOT/logs"
"$KAFKA_LINK/bin/zookeeper-server-start.sh" "$KAFKA_LINK/config/zookeeper.properties" > "$PROJECT_ROOT/logs/zookeeper.log" 2>&1 &
ZOOKEEPER_PID=$!
sleep 5

# 5. Start Kafka Broker
echo "Starting Kafka Broker..."
"$KAFKA_LINK/bin/kafka-server-start.sh" "$KAFKA_LINK/config/server.properties" > "$PROJECT_ROOT/logs/kafka.log" 2>&1 &
KAFKA_PID=$!
sleep 5

echo "==================================================="
echo "Kafka running at localhost:9092"
echo "Project Root:  $PROJECT_ROOT"
echo "Zookeeper PID: $ZOOKEEPER_PID"
echo "Kafka     PID: $KAFKA_PID"
echo "==================================================="
echo "Use 'pkill -f kafka' to stop"
