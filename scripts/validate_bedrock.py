#!/usr/bin/env python3
"""
POV3 — AWS Bedrock Connectivity Validation Script

This script validates that all AWS Bedrock services are correctly
configured before any implementation coding begins.

Usage:
    python scripts/validate_bedrock.py

All checks must pass (✅) before proceeding to Phase 2.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ── Load .env from project root ──────────────────────────────────────────────
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_file)
    print(f"📂 Loaded .env from: {env_file}")
except ImportError:
    print("⚠️  python-dotenv not found — reading raw environment variables")

# ─────────────────────────────────────────────────────────────────────────────
# Validation Checks
# ─────────────────────────────────────────────────────────────────────────────

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def check_env_variables() -> bool:
    """Check all required AWS environment variables are set."""
    print("\n" + "─" * 60)
    print("CHECK 1: Environment Variables")
    print("─" * 60)

    required = {
        "AWS_ACCESS_KEY_ID": "IAM → Users → pov3-bedrock-agent → Security credentials",
        "AWS_SECRET_ACCESS_KEY": "IAM → Users → pov3-bedrock-agent → Security credentials",
        "AWS_REGION": "AWS Console top-right (e.g. us-east-1)",
        "BEDROCK_MODEL_ID": "e.g. amazon.nova-pro-v1:0",
    }

    optional = {
        "BEDROCK_SCREENER_MODEL_ID": "e.g. amazon.nova-lite-v1:0",
        "BEDROCK_EMBED_MODEL_ID": "e.g. amazon.titan-embed-text-v2:0",
        "S3_BUCKET_NAME": "e.g. pov3-optimization-reports",
        "BEDROCK_KB_ID": "Created in Phase 2 — leave blank for now",
    }

    all_present = True
    for key, hint in required.items():
        val = os.getenv(key)
        if val:
            # Mask sensitive values
            display = val[:8] + "..." if len(val) > 8 else val
            print(f"  {PASS} {key} = {display}")
        else:
            print(f"  {FAIL} {key} is NOT set")
            print(f"       Source: {hint}")
            all_present = False

    print()
    for key, hint in optional.items():
        val = os.getenv(key)
        if val:
            print(f"  {PASS} {key} = {val}")
        else:
            print(f"  {WARN} {key} not set (optional for Phase 1)")

    return all_present


def check_boto3_installed() -> bool:
    """Verify boto3 is installed."""
    print("\n" + "─" * 60)
    print("CHECK 2: boto3 Installation")
    print("─" * 60)

    try:
        import boto3
        import botocore
        print(f"  {PASS} boto3 version: {boto3.__version__}")
        print(f"  {PASS} botocore version: {botocore.__version__}")
        return True
    except ImportError:
        print(f"  {FAIL} boto3 is not installed")
        print("       Fix: pip install boto3")
        return False


def check_bedrock_client() -> bool:
    """Test Bedrock control plane connectivity and list available models."""
    print("\n" + "─" * 60)
    print("CHECK 3: Bedrock Client Connectivity")
    print("─" * 60)

    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    region = os.getenv("AWS_REGION", "us-east-1")
    try:
        client = boto3.client(
            "bedrock",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        response = client.list_foundation_models()
        models = response.get("modelSummaries", [])
        print(f"  {PASS} Connected to Bedrock in region: {region}")
        print(f"  {PASS} Found {len(models)} available foundation models")

        # Check if our target models are accessible
        target_models = {
            "amazon.nova-pro-v1:0": False,
            "amazon.nova-lite-v1:0": False,
            "amazon.titan-embed-text-v2:0": False,
        }
        for m in models:
            model_id = m.get("modelId", "")
            if model_id in target_models:
                target_models[model_id] = True

        print()
        for model_id, found in target_models.items():
            if found:
                print(f"  {PASS} Model listed: {model_id}")
            else:
                print(f"  {WARN} Model not found in listing: {model_id}")
                print("        (May still be accessible — check Model Catalog in console)")

        return True

    except NoCredentialsError:
        print(f"  {FAIL} No AWS credentials found")
        print("       Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set in .env")
        return False
    except ClientError as e:
        code = e.response["Error"]["Code"]
        print(f"  {FAIL} Bedrock client error: {code}")
        print(f"       {e.response['Error']['Message']}")
        if code == "AccessDeniedException":
            print("       Fix: Check IAM policy has bedrock:ListFoundationModels permission")
        return False
    except Exception as e:
        print(f"  {FAIL} Unexpected error: {e}")
        return False


def check_model_invocation() -> bool:
    """Test invoking the primary optimization model with a SQL question."""
    print("\n" + "─" * 60)
    print("CHECK 4: Model Invocation")
    print("─" * 60)

    import boto3
    from botocore.exceptions import ClientError

    model_id = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")
    region = os.getenv("AWS_REGION", "us-east-1")

    print(f"  🔍 Invoking model: {model_id}")

    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    payload = {
        "system": [
            {
                "text": "You are a Snowflake SQL optimization expert."
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "In exactly one sentence, explain why SELECT * is inefficient in Snowflake."
                    }
                ],
            }
        ],
        "inferenceConfig": {
            "max_new_tokens": 256,
            "temperature": 0.1
        }
    }

    try:
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        result = json.loads(response["body"].read())
        answer = result["output"]["message"]["content"][0]["text"]
        input_tokens = result.get("usage", {}).get("inputTokens", "?")
        output_tokens = result.get("usage", {}).get("outputTokens", "?")

        print(f"  {PASS} Model invocation successful!")
        print(f"  {PASS} Tokens: input={input_tokens}, output={output_tokens}")
        print(f"\n  Model response:")
        print(f"  ┌{'─' * 56}┐")
        # Word-wrap the response to 54 chars
        words = answer.split()
        line = "  │ "
        for word in words:
            if len(line) + len(word) + 1 > 58:
                print(f"{line:<58}│")
                line = f"  │ {word} "
            else:
                line += word + " "
        if line.strip() != "│":
            print(f"{line:<58}│")
        print(f"  └{'─' * 56}┘")
        return True

    except ClientError as e:
        code = e.response["Error"]["Code"]
        print(f"  {FAIL} Model invocation failed: {code}")
        print(f"       {e.response['Error']['Message']}")
        if code == "AccessDeniedException":
            print()
            print("       Possible causes:")
            print("       1. Model access not enabled in Bedrock console")
            print("          → Bedrock → Model catalog → Enable Amazon Nova Pro")
            print("       2. Wrong region — access was enabled in a different region")
            print("          → Check AWS_REGION matches where you enabled access")
            print("       3. IAM policy missing bedrock:InvokeModel")
            print("          → Check POV3BedrockPolicy has InvokeModel permission")
        elif code == "ValidationException":
            print()
            print("       The model ID format is incorrect.")
            print(f"       Current: {model_id}")
            print("       Expected: amazon.nova-pro-v1:0")
        return False
    except Exception as e:
        print(f"  {FAIL} Unexpected error: {e}")
        return False


def check_s3_access() -> bool:
    """Test S3 bucket accessibility."""
    print("\n" + "─" * 60)
    print("CHECK 5: S3 Bucket Access")
    print("─" * 60)

    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        print(f"  {WARN} S3_BUCKET_NAME not set — skipping S3 check")
        print("       This is OK for Phase 1. Set this before starting Phase 2.")
        return True  # Non-blocking for Phase 1

    import boto3
    from botocore.exceptions import ClientError

    region = os.getenv("AWS_REGION", "us-east-1")
    client = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    try:
        client.head_bucket(Bucket=bucket)
        print(f"  {PASS} S3 bucket '{bucket}' exists and is accessible")
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "404" or code == "NoSuchBucket":
            print(f"  {WARN} S3 bucket '{bucket}' does not exist yet")
            print("       This is expected — create it in Phase 2")
            print("       Fix: aws s3 mb s3://pov3-optimization-reports --region us-east-1")
            return True  # Non-blocking for Phase 1
        elif code == "403":
            print(f"  {FAIL} S3 bucket '{bucket}' access denied")
            print("       Fix: Check IAM policy has s3:ListBucket and s3:GetObject")
            return False
        else:
            print(f"  {FAIL} S3 error ({code}): {e}")
            return False
    except Exception as e:
        print(f"  {FAIL} S3 check failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  POV3 — AWS Bedrock Connectivity Validation")
    print("  Phase 1 Gate Check")
    print("=" * 60)

    results: dict[str, bool] = {}

    results["env_variables"] = check_env_variables()

    if not results["env_variables"]:
        print("\n" + "=" * 60)
        print(f"  {FAIL} BLOCKED: Populate .env before continuing")
        print("=" * 60)
        sys.exit(1)

    results["boto3"] = check_boto3_installed()

    if not results["boto3"]:
        print("\n" + "=" * 60)
        print(f"  {FAIL} BLOCKED: Install boto3 before continuing")
        print("       Run: pip install boto3")
        print("=" * 60)
        sys.exit(1)

    results["bedrock_client"] = check_bedrock_client()
    results["model_invocation"] = check_model_invocation()
    results["s3_access"] = check_s3_access()

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)

    check_labels = {
        "env_variables": "Environment Variables",
        "boto3": "boto3 Installation",
        "bedrock_client": "Bedrock Connectivity",
        "model_invocation": "Model Invocation",
        "s3_access": "S3 Access",
    }

    all_passed = all(results.values())
    for key, label in check_labels.items():
        icon = PASS if results.get(key) else FAIL
        print(f"  {icon} {label}")

    print()
    if all_passed:
        print(f"  {PASS} ALL CHECKS PASSED")
        print()
        print("  ✨ You are ready to begin Phase 2: RAG Foundation")
        print("     Next step: Create S3 bucket and Bedrock Knowledge Base")
    else:
        failed = [check_labels[k] for k, v in results.items() if not v]
        print(f"  {FAIL} FAILED CHECKS: {', '.join(failed)}")
        print()
        print("  Resolve the above issues before proceeding to Phase 2.")
        print("  Refer to the setup guide for troubleshooting steps.")

    print("=" * 60)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
