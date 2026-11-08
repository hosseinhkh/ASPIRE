#!/usr/bin/env python3
"""Two small Bedrock calls to the configured model: Converse and ConverseStream.

Uses the same credentials Claude Code uses (AWS_BEARER_TOKEN_BEDROCK, or keys, or AWS_PROFILE)
and the model in ANTHROPIC_MODEL. Prints the replies, latency and token usage.
"""

from __future__ import annotations

import os
import sys
import time


def main() -> int:
    import boto3
    import botocore

    model = os.environ.get("ANTHROPIC_MODEL", "us.anthropic.claude-opus-4-6-v1").removesuffix(
        "[1m]"
    )
    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client("bedrock-runtime", region_name=region)
    prompt = "In one sentence, tell a robot arm a bedtime story about stacking cups."
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    try:
        started = time.perf_counter()
        reply = client.converse(
            modelId=model, messages=messages, inferenceConfig={"maxTokens": 120}
        )
        latency = time.perf_counter() - started
        text = reply["output"]["message"]["content"][0]["text"].strip()
        usage = reply.get("usage", {})
        print(
            f"converse ({model}, {region}): {latency:.1f} s, "
            f"{usage.get('inputTokens')} in / {usage.get('outputTokens')} out tokens"
        )
        print(f"  {text}")

        started = time.perf_counter()
        stream = client.converse_stream(
            modelId=model, messages=messages, inferenceConfig={"maxTokens": 120}
        )
        first_token = None
        chunks = []
        for event in stream["stream"]:
            delta = event.get("contentBlockDelta", {}).get("delta", {}).get("text")
            if delta:
                first_token = first_token or time.perf_counter() - started
                chunks.append(delta)
        print(
            f"converse_stream: first token {first_token or 0:.1f} s, total {time.perf_counter() - started:.1f} s"
        )
        print(f"  {''.join(chunks).strip()}")
    except botocore.exceptions.NoCredentialsError:
        print(f"FAIL no credentials (boto3 {boto3.__version__}; API keys need boto3 >= 1.39)")
        return 1
    except botocore.exceptions.ClientError as exc:
        error = exc.response["Error"]
        print(f"FAIL {error.get('Code')}: {error.get('Message')}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
