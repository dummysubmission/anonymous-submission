import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")

client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)



# Upload a file with a purpose of "batch"
file_path = "safety_eval/azure_batch_judge/batch_input_files/Gemma4-31B-Instruct/batch_explicit.jsonl"
file = client.files.create(
    file=open(file_path, "rb"),
    purpose="batch",
    extra_body={
        "expires_after": {"seconds": 1209600, "anchor": "created_at"}
    },  # Optional you can set to a number between 1209600-2592000. This is equivalent to 14-30 days
)

print(file.model_dump_json(indent=4))

print(
    f"File expiration: {datetime.fromtimestamp(file.expires_at) if file.expires_at is not None else 'Not set'}"
)

file_id = file.id
