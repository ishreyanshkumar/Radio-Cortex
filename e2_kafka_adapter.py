import socket
import json
import threading
import time
import sys
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable

# Configuration
NS3_HOST = 'localhost'
NS3_PORT = 36421
KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
KPM_TOPIC = 'e2_kpm_stream'
RC_TOPIC = 'e2_rc_control'

class E2KafkaBridge:
    def __init__(self):
        self.sock = None
        self.producer = None
        self.consumer = None
        self.running = True
        
    def connect_kafka(self):
        """Connect to Kafka Producer and Consumer"""
        print(f"[Adapter] Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}...")
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            
            self.consumer = KafkaConsumer(
                RC_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                auto_offset_reset='latest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            print("[Adapter] ✓ Kafka Connected")
        except Exception as e:
            print(f"[Adapter] ✗ Kafka Connection Failed: {e}")
            sys.exit(1)

    def connect_ns3_socket(self):
        """Connect to ns-3 TCP Server"""
        print(f"[Adapter] Connecting to ns-3 E2 socket at {NS3_HOST}:{NS3_PORT}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Retry loop for ns-3 startup
        for i in range(30):
            try:
                self.sock.connect((NS3_HOST, NS3_PORT))
                print(f"[Adapter] ✓ Connected to ns-3 on {NS3_HOST}:{NS3_PORT}")
                return True
            except ConnectionRefusedError:
                time.sleep(1)
                
        print("[Adapter] ✗ Could not connect to ns-3 after 30s")
        return False

    def forward_socket_to_kafka(self):
        """Read from ns-3 Socket -> Publish to Kafka"""
        buffer = ""
        while self.running:
            try:
                chunk = self.sock.recv(4096).decode('utf-8')
                if not chunk:
                    print("[Adapter] ns-3 socket closed")
                    break
                
                buffer += chunk
                
                # Simple JSON parsing logic to handle stream fragmentation
                while '{' in buffer and '}' in buffer:
                    start = buffer.find('{')
                    end = buffer.find('}', start) + 1
                    
                    if start != -1 and end != 0:
                        json_str = buffer[start:end]
                        try:
                            data = json.loads(json_str)
                            # Forward to Kafka
                            self.producer.send(KPM_TOPIC, value=data)
                            # print(f"[Adapter] Forwarded KPM report to {KPM_TOPIC}")
                            
                            # Remove processed part from buffer
                            buffer = buffer[end:]
                        except json.JSONDecodeError:
                            # Incomplete JSON, wait for more data
                            break
            except Exception as e:
                if self.running:
                    print(f"[Adapter] Socket Read Error: {e}")
                break
        
        self.running = False

    def forward_kafka_to_socket(self):
        """Read from Kafka -> Send to ns-3 Socket"""
        print("[Adapter] Listening for RC commands from Kafka...")
        for message in self.consumer:
            if not self.running:
                break
            
            try:
                command = message.value
                # print(f"[Adapter] Received RC command, forwarding to ns-3...")
                
                # Send to socket
                json_str = json.dumps(command)
                self.sock.sendall(json_str.encode('utf-8'))
                
            except Exception as e:
                print(f"[Adapter] Kafka Consumer Error: {e}")

    def run(self):
        self.connect_kafka()
        
        if not self.connect_ns3_socket():
            return
            
        # Start threads
        t1 = threading.Thread(target=self.forward_socket_to_kafka)
        t2 = threading.Thread(target=self.forward_kafka_to_socket)
        
        t1.start()
        t2.start()
        
        try:
            t1.join()
            t2.join()
        except KeyboardInterrupt:
            self.running = False

if __name__ == "__main__":
    bridge = E2KafkaBridge()
    bridge.run()
