"""
File-based cache for LLM call results.
Keyed on SHA-256 hash of (transcript_text + llm_name + cache_type).
Saves/loads JSON from data/cache/.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


def _cache_key(transcript: str, llm_name: str, cache_type: str) -> str:
    raw = f"{cache_type}::{llm_name}::{transcript.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def get_cached(transcript: str, llm_name: str, cache_type: str) -> Optional[dict]:
    path = _cache_path(_cache_key(transcript, llm_name, cache_type))
    if path.exists():
        logger.info(f"[CACHE HIT] {cache_type} / {llm_name}")
        with open(path) as f:
            return json.load(f)
    return None


def save_cache(transcript: str, llm_name: str, cache_type: str, data: dict) -> None:
    path = _cache_path(_cache_key(transcript, llm_name, cache_type))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"[CACHE SAVE] {cache_type} / {llm_name} → {path.name}")


def list_cache_entries() -> list[dict]:
    """Return metadata for all cached entries."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for p in sorted(CACHE_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p) as f:
                data = json.load(f)
            entries.append({
                "file": p.name,
                "type": data.get("_cache_type", "unknown"),
                "llm": data.get("_llm_name", "unknown"),
                "call_id": data.get("call_id", "unknown"),
                "size_kb": round(p.stat().st_size / 1024, 1),
            })
        except Exception:
            pass
    return entries


def delete_cache_entry(filename: str) -> bool:
    """Delete a single cache file by filename. Returns True if deleted."""
    path = CACHE_DIR / filename
    if path.exists():
        path.unlink()
        logger.info(f"[CACHE DELETE] {filename}")
        return True
    return False


def clear_cache() -> int:
    """Delete all cache files. Returns count deleted."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = list(CACHE_DIR.glob("*.json"))
    for f in files:
        f.unlink()
    return len(files)
