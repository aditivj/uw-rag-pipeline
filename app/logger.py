import time
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("logs/requests.jsonl")
LOG_FILE.parent.mkdir(exist_ok=True)

class RequestLogger:
    def __init__(self):
        self.records = []

    def log(self, query: str, latency_ms: float,
            input_tokens: int, output_tokens: int,
            cache_hit: bool):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "latency_ms": round(latency_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost_usd": round((input_tokens * 0.00000059) +
                                        (output_tokens * 0.00000079), 6),
            "cache_hit": cache_hit
        }
        self.records.append(record)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def summary(self):
        if not self.records:
            return {}
        latencies = [r["latency_ms"] for r in self.records]
        latencies.sort()
        n = len(latencies)
        return {
            "total_requests": n,
            "cache_hit_rate": sum(r["cache_hit"] for r in self.records) / n,
            "p50_latency_ms": latencies[n // 2],
            "p95_latency_ms": latencies[int(n * 0.95)],
            "avg_cost_usd": sum(r["estimated_cost_usd"]
                                for r in self.records) / n
        }

logger = RequestLogger()
