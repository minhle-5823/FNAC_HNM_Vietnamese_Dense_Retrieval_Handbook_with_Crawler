"""Exact-content duplicate detection using SHA-256."""
import hashlib


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


class DuplicateDetector:
    def __init__(self):
        self.hashes: set[str] = set()

    def restore(self, hashes) -> None:
        self.hashes.update(h for h in hashes if h)

    def check_and_add(self, text: str) -> tuple[bool, str]:
        fp = content_hash(text)
        duplicate = fp in self.hashes
        if not duplicate:
            self.hashes.add(fp)
        return duplicate, fp
