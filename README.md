# Learning Not to Interfere: Adapting Experts for Model Merging

Official code for *Learning Not to Interfere: Adapting Experts for Model Merging*, which introduces **Resolving Interference (RI)**.

Merging experts fine-tuned from a shared pretrained model often degrades their specialized capabilities, because each expert's parameter update also changes its predictions on other tasks. RI adapts each expert independently before merging, using only unlabeled inputs:

- on the expert's **own** task, the adapted expert is distilled toward the original expert, which preserves its specialization;
- on the **other** tasks, it is distilled toward the shared pretrained model, which restricts the task vector's influence outside its own domain.

```
L_RI = KL(expert ‖ student) on own-task inputs  +  α · mean_j KL(base ‖ student) on other-task inputs
```

The adapted experts can then be merged with any existing merging method. This repository contains the language-model experiments: RI for Llama-3.1-8B-Instruct and Gemma-2-9B-Instruct experts, five merging methods (Task Arithmetic, TIES, TSV-M, AdaMerging, AdaRank), and the five-benchmark evaluation. Vision code will follow in [`vision/`](vision/).

## Released artifacts

**RI-adapted experts.** Each was adapted for 500 RI steps from the corresponding [MergeBench](https://huggingface.co/MergeBench) expert.

| Domain | Llama-3.1-8B-Instruct | Gemma-2-9B-Instruct |
|---|---|---|
| Instruction following | [`Llama-3.1-8B-Instruct_instruction_RI`](https://huggingface.co/hoffman-lab/Llama-3.1-8B-Instruct_instruction_RI) | [`gemma-2-9b-it_instruction_RI`](https://huggingface.co/hoffman-lab/gemma-2-9b-it_instruction_RI) |
| Mathematical reasoning | [`Llama-3.1-8B-Instruct_math_RI`](https://huggingface.co/hoffman-lab/Llama-3.1-8B-Instruct_math_RI) | [`gemma-2-9b-it_math_RI`](https://huggingface.co/hoffman-lab/gemma-2-9b-it_math_RI) |
| Multilingual reasoning | [`Llama-3.1-8B-Instruct_multilingual_RI`](https://huggingface.co/hoffman-lab/Llama-3.1-8B-Instruct_multilingual_RI) | [`gemma-2-9b-it_multilingual_RI`](https://huggingface.co/hoffman-lab/gemma-2-9b-it_multilingual_RI) |
| Code generation | [`Llama-3.1-8B-Instruct_coding_RI`](https://huggingface.co/hoffman-lab/Llama-3.1-8B-Instruct_coding_RI) | [`gemma-2-9b-it_coding_RI`](https://huggingface.co/hoffman-lab/gemma-2-9b-it_coding_RI) |
| Safety | [`Llama-3.1-8B-Instruct_safety_RI`](https://huggingface.co/hoffman-lab/Llama-3.1-8B-Instruct_safety_RI) | [`gemma-2-9b-it_safety_RI`](https://huggingface.co/hoffman-lab/gemma-2-9b-it_safety_RI) |

**Expert completions.** [`ri-expert-completions`](https://huggingface.co/datasets/hoffman-lab/ri-expert-completions) contains each original expert's completions on its own benchmark, which RI uses as distillation targets.

The base models ([`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it)) and the original experts (`MergeBench/Llama-3.1-8B-Instruct_*`, `MergeBench/gemma-2-9b-it_*`) are loaded directly from their Hugging Face repositories. All model IDs are listed in [`llm/configs/models.yaml`](llm/configs/models.yaml).

## Setup

```bash
git clone --recursive https://github.com/pramesh39/learning-not-to-interfere.git
cd learning-not-to-interfere
```

The three evaluation harnesses have incompatible dependencies, so each one gets its own conda environment.

```bash
# RI adaptation, merging, and IFEval / GSM8K / ARC-ml (lm-evaluation-harness)
conda env create -f environments/ri.yml
conda activate ri
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"   # used by IFEval

# MBPP+ (bigcode-evaluation-harness)
conda env create -f environments/bigcode.yml
conda run -n bigcode pip install -e third_party/bigcode-evaluation-harness

# XSTest (safety-eval). The pinned fork adds the Gemma-2 chat template and
# removes a duplicated BOS token from the Llama-3 template.
conda env create -f environments/safety-eval.yml
conda run -n safety-eval pip install -e third_party/safety-eval
```

Llama-3.1 and Gemma-2 are gated on the Hugging Face Hub. Request access to both, then run `huggingface-cli login`.

All commands below are run from the repository root in the `ri` environment. Supported families are `llama-3.1-8b-instruct` and `gemma-2-9b-it`.

## Usage

### 1. RI adaptation

```bash
python -m llm.ri.train --family llama-3.1-8b-instruct --domain math --output_dir runs/ri
# or all five experts:
bash llm/scripts/train_ri.sh llama-3.1-8b-instruct runs/ri
```

The defaults are the paper's settings: KL divergence, α = 1, 500 steps, effective batch size 32, learning rate 1e-5, AdamW. Training data comes from each benchmark's test partition (see [Evaluation protocol](#evaluation-protocol)), and ground-truth labels are never used.

**Expert completions are on by default.** For each domain, RI distills on the prompt followed by the expert's completion. Completions are looked up in this order:

1. the local cache (`--completions_dir`, default `./expert_completions`);
2. the [published dataset](https://huggingface.co/datasets/hoffman-lab/ri-expert-completions);
3. generation with the original expert, if neither of the above exists.

`--completions_source {auto,local,hub,generate}` forces one of these sources, and `--no_expert_completions` trains on prompts alone. To fetch or generate completions ahead of training:

```bash
python -m llm.ri.completions --family gemma-2-9b-it --domain all                     # download
python -m llm.ri.completions --family gemma-2-9b-it --domain all --source generate   # regenerate
```

### 2. Merging

**Task Arithmetic, TIES and TSV-M.** Each method computes the merged task vector once and writes one checkpoint per scaling coefficient λ.

```bash
# one coefficient
python -m llm.merging.static --family llama-3.1-8b-instruct --experts ri \
    --method ties --coefficients 0.5 --output_dir merged/ties_ri
# the paper's grid (llm/configs/merging.yaml)
python -m llm.merging.static --family llama-3.1-8b-instruct --experts ri \
    --method ties --sweep --output_dir merged/ties_ri
```

**AdaMerging and AdaRank** learn their merging coefficients by entropy minimization on unlabeled test-partition prompts. The model is spread across all visible GPUs.

```bash
python -m llm.merging.adaptive --family llama-3.1-8b-instruct --experts ri \
    --method adamerging --output_dir merged/adamerging_ri
```

`--experts` takes `vanilla` (the original MergeBench experts), `ri` (the released RI experts), or five comma-separated local paths in the order instruction, math, multilingual, coding, safety. For example, you can pass the RI experts trained in step 1.

### 3. Evaluation

```bash
python -m llm.eval.run --model merged/ties_ri/0.5 --family llama-3.1-8b-instruct \
    --split test --output_dir results/ties_ri_0.5
python -m llm.eval.score metrics --results results/ties_ri_0.5 --split test
```

`llm.eval.run` runs each harness in its own environment (`--ri_env`, `--bigcode_env` and `--safety_env` default to `ri`, `bigcode` and `safety-eval`) and saves the per-example outputs. `llm.eval.score` computes every metric from those saved outputs.

### Reproducing the main results

```bash
F=llama-3.1-8b-instruct                       # or gemma-2-9b-it
bash llm/scripts/eval_experts.sh $F           # "Individual" row and the Norm. Avg. reference
for E in vanilla ri; do
  for M in task_arithmetic ties tsvm; do
    bash llm/scripts/merge_select_eval.sh $F $M $E    # sweep, select on val, evaluate on test
  done
  for M in adamerging adarank; do
    bash llm/scripts/adaptive_eval.sh $F $M $E
  done
done
bash llm/scripts/make_table.sh $F
```

To reproduce RI end to end rather than use the released experts, run `train_ri.sh` first. Then pass the five resulting checkpoints to `--experts` as comma-separated paths, in place of `ri`.

## Evaluation protocol

| Domain | Benchmark | Metric |
|---|---|---|
| Instruction following | IFEval | prompt-level strict accuracy |
| Code generation | MBPP+ | pass@1 (10 samples, temperature 0.2) |
| Mathematical reasoning | GSM8K | flexible numeric match, 0-shot chain of thought |
| Safety | XSTest | overall accuracy |
| Multilingual reasoning | ARC-ml (de, es, fr, ru) | normalized accuracy |

**Partitions.** The first 100 examples of each benchmark form a validation partition, which is used only to select merging coefficients. ARC-ml's validation partition takes the first 25 examples of each language. Everything else is the test partition, used for RI adaptation, for AdaMerging/AdaRank, and for all reported results. Boundaries are in [`llm/configs/splits.yaml`](llm/configs/splits.yaml).

**Coefficient selection.** For Task Arithmetic, TIES and TSV-M, λ is the grid value with the highest validation average. The grids are in [`llm/configs/merging.yaml`](llm/configs/merging.yaml); TIES keeps top-K fixed at 20%.

**GSM8K scoring.** lm-eval's built-in answer filters mark a correct answer wrong whenever its format differs from the expected pattern. Answers are therefore extracted from the saved responses, in priority order: a `\boxed{}` answer, then "the answer is …", then the last number. They are compared numerically: `$30`, `30.0` and `30` are equal, fractions are evaluated, and an unparseable response counts as wrong. See [`llm/eval/gsm8k_flexible.py`](llm/eval/gsm8k_flexible.py).

**Norm. Avg.** is the mean over benchmarks of the merged model's score divided by the corresponding original expert's score, times 100. Both the vanilla and the RI rows are normalized by the original experts.

## Notes on reproducibility

- **Expert completions.** The published completions are exactly the ones used to train the released RI experts, so the default (download) reproduces the paper's training data. Regenerating them runs the same procedure but will not match byte for byte, for two reasons:
  - The Llama-3.1 instruction, math and coding completions were taken from evaluation runs of the experts, and the MBPP+ ones were sampled at temperature 0.2. Regeneration decodes all domains greedily.
  - **Known issue:** the published Gemma-2 safety completions were generated from XSTest prompts formatted with the Llama-3 chat template rather than Gemma's. RI training and evaluation use Gemma's template. Generation reproduces the published completions by default (`completions_chat_template` in `llm/configs/models.yaml`); this will be corrected in a future version.
- **Adaptive merging.** AdaMerging and AdaRank are seeded here (`--seed`, default 0). The paper's runs were not seeded, so results can differ slightly from the reported numbers.
- **Precision.** Merged checkpoints are written in bfloat16, the precision every evaluation loads them in. For Llama, Task Arithmetic, AdaMerging and AdaRank are computed in float16, as in the paper.

## Citation

```bibtex
@article{learning-not-to-interfere,
  title  = {Learning Not to Interfere: Adapting Experts for Model Merging},
  author = {TODO},
  year   = {2026}
}
```
