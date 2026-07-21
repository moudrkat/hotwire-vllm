"""One-command verification on your hardware: python -m hotwire.verify

Loads a model (default Qwen/Qwen3-0.6B, override with --model), generates a
throwaway random steering vector sized to it, and checks:
  1. steering changes output          (chaos vector, silly scale)
  2. unsteered requests are untouched (byte-identical baseline after steering)
  3. mixed batch isolation            (steered + unsteered in one batch)
  4. decode cost, idle vs all-steered (quick TPOT comparison)

Prints a report block — please paste it into a GitHub issue, especially from
hardware or models the README table doesn't cover yet.
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model)
    cfg = getattr(cfg, "text_config", cfg)
    vec_dir = tempfile.mkdtemp(prefix="hotwire_verify_")
    torch.manual_seed(0)
    torch.save(torch.randn(cfg.num_hidden_layers, cfg.hidden_size),
               os.path.join(vec_dir, "chaos.pt"))
    os.environ["HOTWIRE_VECTORS"] = vec_dir

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enable_prefix_caching=False,
              tensor_parallel_size=args.tensor_parallel_size,
              trust_remote_code=True)

    layer = cfg.num_hidden_layers // 2
    spec = json.dumps({"id": "chaos", "layer": layer, "scale": 40.0})
    mild = json.dumps({"id": "chaos", "layer": layer, "scale": 1.0})
    prompt = "The capital of France is"

    def gen(extra=None, n=1, max_tokens=25):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            extra_args=extra or {})
        outs = llm.generate([prompt] * n, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    base = gen()[0]
    steered = gen({"hotwire": spec})[0]
    base_after = gen()[0]
    t1 = gen()[0]
    t2 = gen({"hotwire": spec})[0]  # separate calls; batch test below
    mixed = llm.generate(
        [prompt, prompt],
        [SamplingParams(temperature=0.0, max_tokens=25),
         SamplingParams(temperature=0.0, max_tokens=25,
                        extra_args={"hotwire": spec})],
        use_tqdm=False)
    mixed_plain, mixed_steered = (mixed[0].outputs[0].text,
                                  mixed[1].outputs[0].text)

    def tpot(extra=None, n=8, max_tokens=128, repeats=3):
        vals = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            outs = llm.generate([prompt] * n,
                                SamplingParams(temperature=0.0,
                                               max_tokens=max_tokens,
                                               extra_args=extra or {}),
                                use_tqdm=False)
            dt = time.perf_counter() - t0
            vals.append(dt / sum(len(o.outputs[0].token_ids) for o in outs) * 1000)
        return statistics.median(vals)

    tpot_idle = tpot()
    tpot_steered = tpot({"hotwire": mild})

    checks = {
        "steering_changes_output": steered != base,
        "unsteered_untouched": base_after == base,
        "mixed_batch_isolated": mixed_steered != mixed_plain
                                and mixed_plain == base,
    }
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    import vllm

    print("\n" + "=" * 62)
    print("hotwire verify report — paste this into a GitHub issue")
    print("=" * 62)
    print(f"model: {args.model}  (layers={cfg.num_hidden_layers}, "
          f"hidden={cfg.hidden_size}, tp={args.tensor_parallel_size})")
    print(f"gpu: {gpu}")
    print(f"vllm: {vllm.__version__}  torch: {torch.__version__}  "
          f"python: {sys.version.split()[0]}")
    for name, ok in checks.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"decode TPOT idle: {tpot_idle:.2f} ms/tok   "
          f"all-steered: {tpot_steered:.2f} ms/tok")
    print("=" * 62)
    if not all(checks.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
