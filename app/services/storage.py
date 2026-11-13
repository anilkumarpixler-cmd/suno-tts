"""Finished stories and voice recordings in the S3-compatible bucket from .env. Calls block: run
them in a thread."""

from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import (
    STORAGE_ACCESS_KEY_ID,
    STORAGE_BUCKET,
    STORAGE_ENDPOINT,
    STORAGE_REGION,
    STORAGE_SECRET_ACCESS_KEY,
)

# Prefix of every bucket key, so final_key tells a bucket file from a local path.
STORY_PREFIX = "stories/"
# voices/<voice id>/original.<ext> and voices/<voice id>/prompt.wav (app/services/voice_files.py).
VOICE_PREFIX = "voices/"
# tmp/<story key>/partial.flac: chunks of a story left halfway (app/services/story_partial.py).
# A bucket lifecycle rule deletes them after 7 days.
PARTIAL_PREFIX = "tmp/"
SIGNED_URL_SECONDS = 3600

_client = None


def storage_configured() -> bool:
    return all(
        (STORAGE_ENDPOINT, STORAGE_REGION, STORAGE_ACCESS_KEY_ID, STORAGE_SECRET_ACCESS_KEY, STORAGE_BUCKET)
    )


def _s3():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=STORAGE_ENDPOINT,
            region_name=STORAGE_REGION,
            aws_access_key_id=STORAGE_ACCESS_KEY_ID,
            aws_secret_access_key=STORAGE_SECRET_ACCESS_KEY,
            # Path style: bucket names with capitals don't work as a host name.
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
    return _client


def story_key_for(story_key: str, path: Path) -> str:
    return f"{STORY_PREFIX}{story_key}{path.suffix}"


def partial_key_for(story_key: str) -> str:
    return f"{PARTIAL_PREFIX}{story_key}/partial.flac"


def is_stored(key: str | None) -> bool:
    return bool(key) and key.startswith(STORY_PREFIX)


CONTENT_TYPES = {
    ".wav": "audio/wav", ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".flac": "audio/flac",
    ".mp3": "audio/mpeg", ".aac": "audio/aac",
}


def upload_file(path: Path, key: str) -> None:
    content_type = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
    _s3().upload_file(str(path), STORAGE_BUCKET, key, ExtraArgs={"ContentType": content_type})


def exists(key: str) -> bool:
    # Checked before the local copy of a story is deleted.
    try:
        _s3().head_object(Bucket=STORAGE_BUCKET, Key=key)
        return True
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def signed_url(key: str, seconds: int = SIGNED_URL_SECONDS) -> str:
    # The bucket is private: this link plays the file for `seconds`, then stops working.
    return _s3().generate_presigned_url(
        "get_object", Params={"Bucket": STORAGE_BUCKET, "Key": key}, ExpiresIn=seconds
    )


def delete_file(key: str) -> None:
    _s3().delete_object(Bucket=STORAGE_BUCKET, Key=key)


def download_file(key: str, path: Path) -> None:
    # Through a temp file, so a failed download never leaves half a file behind.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.download")
    try:
        _s3().download_file(STORAGE_BUCKET, key, str(tmp))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def delete_prefix(prefix: str) -> int:
    """Deletes every file under `prefix`. Returns how many."""
    deleted = 0
    for page in _s3().get_paginator("list_objects_v2").paginate(Bucket=STORAGE_BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            delete_file(item["Key"])
            deleted += 1
    return deleted
