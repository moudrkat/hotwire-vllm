"""Benchmark: what does hotwire cost?

Mirrors the metrics discussed in vllm-project/vllm#36998:
  1. baseline      — vanilla vLLM, hotwire not installed (or HOTWIRE_VECTORS unset
                     and no request steered): CUDA graphs on
  2. idle          — hotwire armed, zero requests steered (target: within noise)
  3. steered       — every request steered (worst case)
  4. eager         — enforce_eager=True, no steering (what vllm-lens-style
                     hook tools pay before they even steer)

Reports TTFT and decode TPOT per condition. Run on the GPU box:
  HOTWIRE_VECTORS=... python benchmarks/bench_decode.py --model Qwen/Qwen3-4B-Instruct-2507
"""
import argparse
import json
import statistics
import time

from vllm import LLM, SamplingParams

PROMPT = "Write a long, detailed essay about the history of transportation."


def run(llm, n_reqs, max_tokens, extra_args=None, repeats=3):
    sps = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                         extra_args=extra_args or {})
    ttfts, tpots = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        first = llm.generate([PROMPT] * n_reqs,
                             SamplingParams(temperature=0.0, max_tokens=1,
                                            extra_args=extra_args or {}),
                             use_tqdm=False)
        ttfts.append((time.perf_counter() - t0) / n_reqs)
        t0 = time.perf_counter()
        outs = llm.generate([PROMPT] * n_reqs, sps, use_tqdm=False)
        dt = time.perf_counter() - t0
        n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        tpots.append(dt / n_tok * 1000)
        del first, outs
    return statistics.median(ttfts) * 1000, statistics.median(tpots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--vector-id", default=None,
                    help="steer with this vector id (needs HOTWIRE_VECTORS)")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--n-reqs", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--eager", action="store_true",
                    help="run the eager-mode condition instead of graph mode")
    args = ap.parse_args()

    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=4096,
              enable_prefix_caching=False, enforce_eager=args.eager)

    mode = "eager" if args.eager else "graphs"
    ttft, tpot = run(llm, args.n_reqs, args.max_tokens)
    print(f"[{mode}] idle/baseline: TTFT {ttft:8.1f} ms   TPOT {tpot:6.2f} ms/tok")

    if args.vector_id:
        xa = {"hotwire": json.dumps({"id": args.vector_id, "layer": args.layer,
                                     "scale": args.scale})}
        ttft, tpot = run(llm, args.n_reqs, args.max_tokens, xa)
        print(f"[{mode}] all-steered:   TTFT {ttft:8.1f} ms   TPOT {tpot:6.2f} ms/tok")


if __name__ == "__main__":
    main()
