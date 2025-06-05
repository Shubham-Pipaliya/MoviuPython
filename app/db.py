from mongoengine import connect
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env values

MONGO_URI = os.getenv("MONGO_URI")

try:
    connect(
        db="SpakMobileApi",
        host=MONGO_URI,
        alias="default"  # THIS IS MANDATORY
    )
    print("✅ MongoEngine connection successful")
except Exception as e:
    raise RuntimeError("❌ Failed to connect to MongoDB via MongoEngine:", e)
