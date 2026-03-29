"""MongoDB connection, collection handles, and related config."""

import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

_uri = os.getenv("MONGO_URI")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


WHITEBOARD_DB_REFRESH_MS = _env_int("WHITEBOARD_DB_REFRESH_MS", 250)

drawings_col = None
points_col   = None
erases_col   = None

if _uri:
    try:
        _client = MongoClient(_uri, serverSelectionTimeoutMS=2000)
        _db = _client["cavepainting"]
        drawings_col = _db["drawings"]
        points_col   = _db["points"]
        erases_col   = _db["erases"]
        _client.admin.command("ping")
        drawings_col.create_index([("sessionId", 1), ("createdAt", 1)])
        points_col.create_index([("sessionId", 1), ("_id", 1)])
        erases_col.create_index([("sessionId", 1), ("_id", 1)])
    except Exception as exc:
        print(f"[mongo] Disabled (connection failed): {exc}")
        drawings_col = None
        points_col   = None
        erases_col   = None
