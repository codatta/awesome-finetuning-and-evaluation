# Contributing Guide

Thanks for improving **awesome‑finetuning‑evaluation**. Follow the steps below and open a PR. We review quickly and value clear, minimal additions.

---
## 1 · Set up
```bash
git clone https://github.com/codatta/awesome-finetuning-and-evaluation.git
cd awesome-finetuning-and-evaluation
poetry install          # or pip install -r requirements.txt
pre-commit install      # hook for linting
```
---
## 2 · Branching
| Goal              | Branch name example            |
| ----------------- | ------------------------------ |
| New resource      | `add-finance-qa-benchmark`     |
| Doc tweak         | `docs-fix-typo-readme`         |
| Feature / script  | `feat-lora-docker-runner`      |

Create from **`main`**:

```bash
git checkout -b add-<topic>
```
---

## 3 · Commit style
Use [Conventional Commits](https://www.conventionalcommits.org):

```
docs(readme): clarify quick‑start section
feat(script): add LoRA trainer wrapper
fix(ci): pin nbconvert version
```
Limit subject lines to 72 chars.

---

## 4 · Adding a resource
Step 1. **Place the file**
   - Markdown stub → `resources/<category>/YYYY-author-keywords.md`
   - Notebook / script → `notebooks/` or `scripts/`
   - Large assets (datasets, models) should be linked, not stored.

Step 2. **Update README tables**
   - Insert a row in **Fine‑tuning resources** or **Evaluation resources**.
   - Add a new sub‑folder under `resources/` if needed and list it in the **Resource map**.

Step 3. **Run checks**
   ```bash
   make lint        # ruff + black
   make test        # pytest (if tests added)
   ```

Step 4. **Commit & push**
   ```bash
   git add <files>
   git commit -m "docs(<section>): add <name>"
   git push --set-upstream origin <branch>
   ```

Step 5. **Open a PR**
   CI will verify links and lint Markdown.

---

## 5 · Code quality

| Tool      | Purpose            | Command          |
| --------- | ------------------ | ---------------- |
| **black** | code formatting    | `make format`    |
| **ruff**  | linting            | `make lint`      |
| **pytest**| unit tests         | `make test`      |

Notebooks run through **nbQA** black to stay clean.

---

## 6 · Code of Conduct

We follow the [Contributor Covenant](CODE_OF_CONDUCT.md). Be respectful.

---

## 7 · License

Your contributions are licensed under the repo’s MIT License.
