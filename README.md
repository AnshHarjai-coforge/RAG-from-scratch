# RAG from scratch

A local retrieval pipeline over the policy files in `data/policies`. It chunks each file, embeds the chunks with Qwen, and stores them in Chroma. A question is answered from a hybrid keyword and vector search, then a cross-encoder rerank, then Claude on Microsoft Foundry.

Run every command from the repository root.

## Setup

Use Python 3.12 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The first index or question downloads `Qwen/Qwen3-Embedding-0.6B` and `cross-encoder/ms-marco-MiniLM-L-6-v2`.

Indexing does not call Foundry. `scripts/ask.py` and `scripts/evaluate.py` do. Set these in `.env` before those two commands:

- `ANTHROPIC_FOUNDRY_API_KEY`
- `ANTHROPIC_FOUNDRY_RESOURCE`
- `ANTHROPIC_MODEL` (optional; the default is `claude-opus-5`)

The vector store is written to `data/chroma`, which git ignores.

## Index the policies

```bash
python scripts/index_policies.py
```

This reads `data/policies`, stores the chunks, and prints the closest chunks for one remote-work question. A markdown file whose opening front matter says `status: superseded` is removed from the store and left out of the index.

## Ask a question

Rebuild the store, then answer:

```bash
python scripts/ask.py --index "How many days per week may an employee work remotely?"
```

Answer from the chunks already stored on disk:

```bash
python scripts/ask.py "What is the expense report deadline?"
```

The command prints the vector hits, the hybrid hits, the reranked hits, the answer, and the source file and section for each cited chunk.

## Score the fixed questions

```bash
python scripts/evaluate.py
```

This indexes the policies again, then scores the questions in `src/rag/eval_cases.py`. It prints retrieval recall and answer accuracy. The process exits 0 when every case matches the expected source and phrase, and exits 1 otherwise.

## Tests and lint

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
ruff format --check .
```

## Settings

`.env.example` lists the defaults: chunk size, overlap, `TOP_K`, the policy directory, the Chroma directory, the collection name, and the model names. `docs/diagnosis.md` records why a remote-work question can cite both the current policy and the superseded 2024 file.
