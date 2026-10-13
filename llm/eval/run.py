"""Run the five-benchmark evaluation for one model.

    python -m llm.eval.run --model merged/ties_ri/0.5 --family llama-3.1-8b-instruct \
        --split test --output_dir results/ties_ri_0.5

Benchmarks and harnesses (each runs in its own conda environment):
    instruction  IFEval   lm-evaluation-harness     (env: ri)
    math         GSM8K    lm-evaluation-harness     (env: ri)   0-shot CoT, boxed answer
    multilingual ARC-ml   lm-evaluation-harness     (env: ri)   arc_de/es/fr/ru
    coding       MBPP+    bigcode-evaluation-harness (env: bigcode)
    safety       XSTest   safety-eval                (env: safety-eval)

--split val runs only the 100 held-out examples per benchmark (used to select
merging coefficients); --split test runs the full set, from which
llm.eval.score reads the test partition. Per-example outputs are saved, and
llm.eval.score computes all metrics from them.
"""
import argparse
import os
import subprocess

from llm.config import DOMAINS, REPO_ROOT, family_cfg, models_cfg, splits_cfg

TASKS_DIR = os.path.join(REPO_ROOT, "llm", "eval", "tasks")
BIGCODE_DIR = os.path.join(REPO_ROOT, "third_party", "bigcode-evaluation-harness")
SAFETY_DIR = os.path.join(REPO_ROOT, "third_party", "safety-eval")

LM_EVAL_TASKS = {
    "instruction": ["ifeval"],
    "math": ["gsm8k_cot_zeroshot_boxed"],
    "multilingual": ["arc_de", "arc_es", "arc_fr", "arc_ru"],
}
BATCH_SIZE = {"instruction": 16, "math": 16, "multilingual": 64, "coding": 10, "safety": 8}


def _model_ref(model):
    """Absolute path for a local checkpoint (harnesses run from their own
    directory); Hub ids are passed through unchanged."""
    return os.path.abspath(model) if os.path.exists(model) else model


def _env(env, cmd):
    return ["conda", "run", "--no-capture-output", "-n", env] + cmd


def lm_eval(model, domain, split, out, env):
    cmd = ["lm_eval", "--model", "hf", "--model_args", f"pretrained={model},dtype=bfloat16",
           "--tasks", ",".join(LM_EVAL_TASKS[domain]), "--include_path", TASKS_DIR,
           "--device", "cuda:0", "--batch_size", str(BATCH_SIZE[domain]),
           "--log_samples", "--output_path", os.path.join(out, domain)]
    if split == "val":
        cmd += ["--limit", "100"]          # covers the first 25 per ARC-ml language too
    return _env(env, cmd), None


def mbppplus(model, split, out, env):
    cfg = splits_cfg()["coding"][split]
    os.makedirs(os.path.join(out, "coding"), exist_ok=True)
    cmd = ["accelerate", "launch", "main.py", "--model", _model_ref(model),
           "--tasks", "mbppplus", "--temperature", "0.2", "--n_samples", "10",
           "--max_length_generation", "512", "--precision", "bf16",
           "--batch_size", str(BATCH_SIZE["coding"]), "--allow_code_execution",
           "--limit_start", str(cfg["start"]), "--limit", str(cfg["end"] - cfg["start"]),
           "--save_generations", "--save_generations_path", os.path.abspath(os.path.join(out, "coding", "generations.json")),
           "--metric_output_path", os.path.abspath(os.path.join(out, "coding", f"metrics_{split}.json"))]
    return _env(env, cmd), BIGCODE_DIR


def xstest(model, chat_template, out, env):
    d = os.path.abspath(os.path.join(out, "safety"))
    os.makedirs(d, exist_ok=True)
    cmd = ["python", "evaluation/eval.py", "generators", "--model_name_or_path", _model_ref(model),
           "--use_vllm", "--model_input_template_path_or_name", chat_template, "--tasks", "xstest",
           "--report_output_path", os.path.join(d, "xstest_eval.json"),
           "--save_individual_results_path", os.path.join(d, "xstest_generation.json"),
           "--batch_size", str(BATCH_SIZE["safety"])]
    return _env(env, cmd), SAFETY_DIR


def main():
    ap = argparse.ArgumentParser(description="Five-benchmark evaluation")
    ap.add_argument("--model", required=True, help="local checkpoint directory or Hub id")
    ap.add_argument("--family", required=True, choices=sorted(models_cfg()["families"]))
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--benchmarks", nargs="+", default=DOMAINS, choices=DOMAINS,
                    help="domains to evaluate (default: all five)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ri_env", default="ri")
    ap.add_argument("--bigcode_env", default="bigcode")
    ap.add_argument("--safety_env", default="safety-eval")
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    template = family_cfg(a.family)["chat_template"]

    jobs = []
    for d in a.benchmarks:
        if d in LM_EVAL_TASKS:
            jobs.append((d, *lm_eval(a.model, d, a.split, a.output_dir, a.ri_env)))
        elif d == "coding":
            jobs.append((d, *mbppplus(a.model, a.split, a.output_dir, a.bigcode_env)))
        elif d == "safety":
            # XSTest always runs all 450 prompts; the split is applied when scoring.
            jobs.append((d, *xstest(a.model, template, a.output_dir, a.safety_env)))
    with open(os.path.join(a.output_dir, "run_info.txt"), "w") as f:
        f.write(f"model={a.model}\nfamily={a.family}\nsplit={a.split}\n")
    for d, cmd, cwd in jobs:
        print(f"[eval] {d}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=cwd, check=True)


if __name__ == "__main__":
    main()
