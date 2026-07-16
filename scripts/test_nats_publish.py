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
        "task": "Analyze slow query 01c5a808-0002-3a35-000e-044e0008348e",
        "payload": {
            "query_id": "01c5a808-0002-3a35-000e-044e0008348e",
            "warehouse": "COMPUTE_WH",
            "credits_used": 0.00002,
            "execution_time_seconds": 1.859,
            "bytes_scanned": 147609264,
            "issue_type": "NON_SARGABLE_PREDICATE",
            "query_text": "SELECT L_ORDERKEY, L_QUANTITY, L_EXTENDEDPRICE FROM LINEITEM WHERE YEAR(L_SHIPDATE) = 1995 AND MONTH(L_SHIPDATE) = 3;"
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
