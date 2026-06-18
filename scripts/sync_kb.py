import sys
import os

# Ensure the 'src' module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.connectors.s3_manager import get_s3_manager

def main():
    s3 = get_s3_manager()
    print(f"Triggering Bedrock Knowledge Base Sync...")
    s3.sync_knowledge_base()
    print("Sync job triggered successfully!")

if __name__ == "__main__":
    main()
