import pathway as pw
import congestion_scenarios as cs
import time

def run_producer():
    print("Generating scenario data...")
    # Generate data
    generator = cs.CongestionScenarioGenerator(num_ues=5, num_cells=2)
    config = cs.ScenarioConfig(
        scenario_type=cs.ScenarioType.FLASH_CROWD,
        start_time=0,
        duration=10,
        severity=0.5,
        affected_cells=[0]
    )
    data = generator.generate_scenario(config)
    print(f"Generated {len(data)} records.")

    # Define Schema matching the generator output
    class KPMSchema(pw.Schema):
        timestamp: float
        ue_id: int
        cell_id: int
        throughput: float
        delay: float
        packet_loss: float
        sinr: float
        queue_length: int
        rb_utilization: float

    # Create Pathway Table
    print("Creating Pathway table...")
    table = pw.debug.table_from_list(data, schema=KPMSchema)

    # Write to Kafka
    topic = "input_topic"
    print(f"Writing to Kafka topic: {topic}")
    
    rdkafka_settings = {
        "bootstrap.servers": "localhost:9092",
    }

    pw.io.kafka.write(
        table,
        rdkafka_settings=rdkafka_settings,
        topic=topic,
        format="json",
    )

    print("Running Pathway pipeline to push data to Kafka...")
    pw.run()

if __name__ == "__main__":
    run_producer()
