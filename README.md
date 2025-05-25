# 🚀 Awesome Finetuing And Evaluation

**Purpose** — give engineers and product teams a single place to turn a general‑purpose foundation model into a production‑ready vertical solution through systematic evaluation and parameter‑efficient fine‑tuning.

## 🚀 Quick links
[Papers](resources/papers/) · [Tutorials](resources/tutorials/) · [Projects](resources/projects/) · [Benchmarks](resources/benchmarks/) · [Datasets](resources/datasets/)
· [Models](resources/models/)
---

## Why evaluation + fine‑tuning?

| Stage | What it is | Why you care |
|-------|-----------|--------------|
| **Evaluate** | Measure how the base model or system performs on your exact tasks. | Reveals the gaps you must close before shipping. |
| **Fine‑tune** | Update a small subset of weights with curated domain data. | Injects expertise fast without retraining from scratch and lets you upgrade to stronger base models later. |
| **Repeat** | Evaluate → fine‑tune → evaluate. | Continuous gains, backed by hard evidence. |

Foundation models are great starting points, but real products demand **vertical localisation**: prompt design, RAG, PEFT‑style fine‑tuning, and relentless testing. This repo packages the resources and examples that make that loop repeatable.

---

## Who is this for?

* **Engineers** — grab runnable notebooks and scripts.
* **Product managers** — skim curated links to gauge scope and risk.
* **Builders** — fork code that already works on real data.

Have a paper, tutorial, or project on fine‑tuning or evaluation? Add its file to the correct `resources/` folder **and** insert a row in the Fine‑tuning or Evaluation table, then open a PR so everyone can find it.

---

## Fine‑tuning resources

| Name | Kind | Location | Brief |
|------|------|----------|-------|
| **LLaMA‑Factory** | OSS project | <https://github.com/hiyouga/LLaMA-Factory> | End‑to‑end PEFT/LoRA/QLoRA fine‑tuning toolkit for Llama‑family models |
| *SDXL Lora notebook* | Jupyter .ipynb | [`notebooks/finetune/01_azuki_lora.ipynb`](notebooks/01_lora_quickstart.ipynb) | Minimal walk‑through: prepare data, run LoRA, show result |
| *LoRA* | Paper / PDF | [`resources/papers/LoRA.pdf`](resources/papers/LoRA.pdf) | LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS |
| *Azuki dataset (sample placeholder)* | Dataset | <https://huggingface.co/datasets/Chris1/azuki> | 100 K images of Azuki NFT |
| Let's fine-tune Qwen 3 (100% locally) | Tutorial | https://x.com/_avichawla/status/1918917563825938645 by [@_avichawla (X)](https://x.com/_avichawla) | X thread (TODO: convert to notebook) |

## Evaluation resources

| Name | Kind | Location | Brief |
|------|------|----------|-------|
| **Evidently** | OSS project | <https://github.com/evidentlyai/evidently> | Open‑source data / model quality monitoring & LLM evaluation dashboards |
| **pydantic-eval** | OSS project | https://ai.pydantic.dev/evals/ | super light-weight toolset to buids evaluation with LLM for AI |
| **instructor(.llm_validator)** | OSS project | https://github.com/567-labs/instructor ([blog](https://python.useinstructor.com/blog/2025/05/20/understanding-semantic-validation-with-structured-outputs/)) | Beyond rule-based validation (e.g. `a filed has correct datatype 'str'`), semantic validation uses an LLM to interpret and evaluate text against natural language criteria ([blog](https://python.useinstructor.com/blog/2025/05/20/understanding-semantic-validation-with-structured-outputs/#beyond-rule-based-validation)).|
| *ChatEval demo* | Jupyter .ipynb | [`notebooks/evaluation/01_evidently_llm_eval.ipynb`](notebooks/evaluation/01_evidently_llm_eval.ipynb) | Build an interactive eval dashboard in 5 mins |
| *A Survey on Evaluation of Large Language Models* | Paper / PDF | [`resources/papers/A_Survey_on_Evaluation_of_LLMs.pdf`](resources/papers/A_Survey_on_Evaluation_of_LLMs.pdf) | A comprehensive survey on LLM evaluation and pipelines |
| *Finance‑QA benchmark* | Dataset | <https://huggingface.co/datasets/AIR-Bench/qa_finance_en> | 27.9K Q&A pairs for financial reasoning tests |

---

## Repo layout

```
awesome-finetuning-evaluation/
├─ resources/   # links to papers, tutorials, OSS
├─ notebooks/   # Colab & Jupyter demos
├─ scripts/     # CLI + Docker for local jobs
└─ docs/        # deeper guides & diagrams
```
Each folder ships its own `README.md` index.

---

## Resource map

| Path | Purpose |
|------|---------|
| `resources/papers/` | Seminal & recent papers on human‑intelligence evaluation, RLHF, PEFT, and vertical fine‑tuning |
| `resources/tutorials/` | Hands‑on guides, blog posts, and course notes that walk through evaluation loops and LoRA/QLoRA workflows |
| `resources/projects/` | Open‑source reference implementations, CLI tools, and libraries you can fork or vendor |
| `resources/benchmarks/` | Domain‑specific test sets, leaderboards, and evaluation configs to quantify progress |
| `resources/datasets/` | Curated public datasets suitable for fine‑tuning and continual training |

Make a PR any time you add a file: new resource doc **plus** an index update in the table above.

---

## Quick start

```bash
# clone and explore
git clone https://github.com/codatta/awesome-finetuning-and-evaluation.git
cd awesome-finetuning-and-evaluation

# open the zero‑to‑LoRA demo
jupyter lab notebooks/00_quickstart.ipynb
```

Prefer Colab? Click the badge at the top of each notebook.

---

## Contributing

### 1 · Add a new resource

1. **Create a branch**
   ```bash
   git checkout -b add-<topic>
   ```
2. **Add the file** in the correct `resources/` sub‑folder.
   - Use **`YYYY-author-keywords.md`** for Markdown stubs.
   - If the resource ships code, place it in `scripts/` or `notebooks/` **and** add a short stub in `resources/projects/` that links back.
3. **Update navigation tables**
   - Insert a row in **Fine‑tuning resources** *or* **Evaluation resources** above.
   - If the resource belongs to a new category, add it to the **Resource map**.
4. **Commit & push**
   ```bash
   git add <files>
   git commit -m "docs(<section>): add <name>"
   git push --set-upstream origin add-<topic>
   ```
5. **Open a PR** – CI will run link checks and Markdown linting.

### 2 · Improve docs or code

Feel free to tweak wording, fix typos, or extend scripts. Follow the same **branch → commit → PR** flow.

Full guidelines live in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

Made with 🧠 + ❤️ by [**Codatta**](https://codatta.ai). If this repo helps you, please ⭐ it so others can find it too.
