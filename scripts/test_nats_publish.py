import asyncio
import json
import uuid
from datetime import datetime, timezone

import nats

async def main():
    # 1. Connect to the local NATS server
    print("Connecting to NATS at nats://localhost:4222...")
    try:
        nc = await nats.connect("nats://localhost:4222")
    except Exception as e:
        print(f"Failed to connect to NATS: {e}")
        print("Make sure you have a NATS server running (e.g., 'nats-server' or via Docker).")
        return

    print("Connected.")

    # 2. Construct the AgentMessage payload simulating POV4
    subject = "pov4.alerts.optimization"
    
    payload = {
        "message_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sender": "POV4AlertAgent",
        "receiver": "AnalysisAgent",
        "task": "Analyze slow query 01c587bb-0002-33bc-000e-044e000583a6",
        "payload": {
            "query_id": "01c587bb-0002-33bc-000e-044e000583a6",
            "warehouse": "COMPUTE_WH",
            "credits_used": 0.000023,
            "execution_time_seconds": 0.200,
            "issue_type": "NON_SARGABLE_PREDICATE",
            "query_text": "SELECT * FROM MALL_CUSTOMERS WHERE CAST(AGE AS VARCHAR) LIKE '3%' ORDER BY TO_VARCHAR(ANNUAL_INCOME_K) || ' thousand';"
        }
    }

    # 3. Publish the message
    data = json.dumps(payload).encode()
    print(f"\nPublishing to subject '{subject}':")
    print(json.dumps(payload, indent=2))
    
    await nc.publish(subject, data)
    print("\nMessage published successfully!")

    # 4. Gracefully close connection
    await nc.drain()

if __name__ == '__main__':
    asyncio.run(main())
