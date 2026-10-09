"""Semantic cache for LLM Router"""
import hashlib
import json
import time
import re
from collections import OrderedDict
from typing import Optional


class SemanticCache:
    def __init__(self, max_size=1000, ttl=3600, threshold=0.88):
        self.max_size = max_size
        self.ttl = ttl
        self.threshold = threshold
        self._cache = OrderedDict()
        self._stats = {"hits": 0, "misses": 0, "size": 0}

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text[:2000]

    def _ngram_hash(self, text: str, n=3) -> set:
        norm = self._normalize(text)
        words = norm.split()
        ngrams = set()
        for i in range(len(words) - n + 1):
            h = hashlib.md5(" ".join(words[i:i+n]).encode()).hexdigest()
            ngrams.add(h)
        return ngrams

    def _similarity(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def get(self, messages: list, model: str) -> Optional[dict]:
        if not self._cache:
            self._stats["misses"] += 1
            return None

        query = self._build_key(messages)
        query_ngrams = self._ngram_hash(query)
        now = time.time()

        for key, entry in list(self._cache.items()):
            if now - entry["time"] > self.ttl:
                del self._cache[key]
                continue

            if entry["model"] != model:
                continue

            stored_ngrams = entry.get("ngrams", set())
            if not stored_ngrams:
                continue

            sim = self._similarity(query_ngrams, stored_ngrams)
            if sim >= self.threshold:
                self._stats["hits"] += 1
                self._cache.move_to_end(key)
                return entry["response"]

        self._stats["misses"] += 1
        return None

    def set(self, messages: list, model: str, response: dict):
        if self.max_size <= 0:
            return

        query = self._build_key(messages)
        key = hashlib.md5(query.encode()).hexdigest()
        ngrams = self._ngram_hash(query)

        if key in self._cache:
            self._cache.move_to_end(key)
            return

        self._cache[key] = {
            "time": time.time(),
            "model": model,
            "response": response,
            "ngrams": ngrams,
            "query_sample": self._normalize(query)[:100],
        }

        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

        self._stats["size"] = len(self._cache)

    def _build_key(self, messages: list) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                )
            parts.append(f"{role}:{content}")
        return " || ".join(parts)

    def stats(self) -> dict:
        return {**self._stats, "size": len(self._cache)}

    def clear(self):
        self._cache.clear()
        self._stats["hits"] = 0
        self._stats["misses"] = 0
        self._stats["size"] = 0
