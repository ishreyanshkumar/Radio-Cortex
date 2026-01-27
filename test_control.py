
from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Command to change TxPower of Cell 0 to 40 dBm
command = {
    "cell_0": {
        "TxPower": 40.0
    }
}

print(f"Sending command: {command}")
producer.send('e2_rc_control', command)
producer.flush()
print("Sent.")
time.sleep(2)
