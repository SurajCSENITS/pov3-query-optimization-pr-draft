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
        "task": "Analyze slow query 01c549ca-0002-2472-000e-044e0001723e",
        "payload": {
            "query_id": "01c549ca-0002-2472-000e-044e0001723e",
            "warehouse": "COMPUTE_WH",
            "credits_used": 0.00034,
            "execution_time_seconds": 0.507,
            "issue_type": "NON_SARGABLE_JOIN_CONDITION",
            "query_text": "SELECT l.L_ORDERKEY, o.O_ORDERSTATUS, l.L_QUANTITY, o.O_TOTALPRICE FROM LINEITEM l, ORDERS o WHERE l.L_QUANTITY > 49 AND o.O_TOTALPRICE > 400000 AND l.L_ORDERKEY = o.O_ORDERKEY + 1"
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
