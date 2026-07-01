import json
import time
import requests
from datetime import datetime

QUERY_URL = "http://localhost:8001/query"
GROUND_TRUTH_FILE = "evals/ground_truth.json"
RESULTS_FILE = "evals/eval_results.json"

def keyword_score(answer: str, keywords: list) -> float:
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 2)

def run_eval():
    with open(GROUND_TRUTH_FILE) as f:
        questions = json.load(f)

    results = []
    passed = 0
    total = len(questions)

    print(f"\nRunning eval on {total} questions...\n")
    print(f"{'ID':<10} {'Score':<8} {'Latency':<12} {'Cache':<8} Question")
    print("-" * 80)

    for q in questions:
        try:
            start = time.time()
            response = requests.post(QUERY_URL, json={
                "query": q["question"],
                "top_k": 10
            }, timeout=30)
            latency = round((time.time() - start) * 1000, 1)

            if response.status_code != 200:
                print(f"{q['id']:<10} HTTP {response.status_code}: {response.text[:150]}")
                continue

            data = response.json()

            if "answer" not in data:
                print(f"{q['id']:<10} UNEXPECTED RESPONSE: {data}")
                continue

            score = keyword_score(data["answer"], q["keywords"])
            passed += 1 if score >= 0.5 else 0

            result = {
                "id": q["id"],
                "question": q["question"],
                "expected_keywords": q["keywords"],
                "answer": data["answer"],
                "keyword_score": score,
                "latency_ms": data["latency_ms"],
                "cost_usd": data["cost_usd"],
                "cache_hit": data["cache_hit"],
                "chunks_used": data["chunks_used"],
                "pass": score >= 0.5
            }
            results.append(result)

            print(f"{q['id']:<10} {score:<8} {data['latency_ms']:<12.0f} "
                  f"{str(data['cache_hit']):<8} {q['question'][:45]}")

            # Small delay to avoid hammering Groq's rate limit
            time.sleep(1)

        except Exception as e:
            print(f"{q['id']:<10} ERROR: {type(e).__name__}: {e}")

    if not results:
        print("\nNo successful results — cannot compute summary.")
        return

    latencies = [r["latency_ms"] for r in results]
    latencies.sort()
    n = len(latencies)

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_questions": total,
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results) * 100, 1),
        "p50_latency_ms": latencies[n // 2],
        "p95_latency_ms": latencies[int(n * 0.95)] if n > 1 else latencies[0],
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 6),
        "avg_cost_per_query_usd": round(sum(r["cost_usd"] for r in results) / n, 6),
        "cache_hit_rate": round(sum(r["cache_hit"] for r in results) / n * 100, 1),
        "results": results
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"EVAL COMPLETE")
    print(f"Pass rate:        {summary['pass_rate']}% ({passed}/{len(results)})")
    print(f"P50 latency:      {summary['p50_latency_ms']} ms")
    print(f"P95 latency:      {summary['p95_latency_ms']} ms")
    print(f"Total cost:       ${summary['total_cost_usd']}")
    print(f"Avg cost/query:   ${summary['avg_cost_per_query_usd']}")
    print(f"Cache hit rate:   {summary['cache_hit_rate']}%")
    print(f"\nFull results saved to: {RESULTS_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    run_eval()

