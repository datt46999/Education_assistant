# check_env.py
import os
from dotenv import load_dotenv

load_dotenv()

required = [
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "REGION_NAME",
    "MEMORY_ID", "MEMORY_STRATEGY_ID", "ACCOUNT_ID", "AGENT_RUNTIME_ARN",
]

for k in required:
    v = os.getenv(k)
    if not v:
        print(f"❌ {k}: missing")
    elif v.startswith("<") and v.endswith(">"):
        print(f"❌ {k}: still a placeholder")
    else:
        print(f"✅ {k}: set (len={len(v)})")

# Format checks
acct = os.getenv("ACCOUNT_ID", "")
print("ACCOUNT_ID 12 digits:", acct.isdigit() and len(acct) == 12)
print("AGENT_RUNTIME_ARN format:", (os.getenv("AGENT_RUNTIME_ARN") or "").startswith("arn:aws:bedrock-agentcore:"))

# Real credential test: confirms the keys work and match ACCOUNT_ID
import boto3
sts = boto3.client(
    "sts",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("REGION_NAME"),
)
ident = sts.get_caller_identity()
print("STS OK, account:", ident["Account"], "| matches ACCOUNT_ID:", ident["Account"] == acct)


# Save as test.py and run: python test.py

