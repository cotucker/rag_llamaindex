# RAG LlamaIndex Terminal

An interactive terminal app for asking rich, context‑aware questions over **your own documents** using Retrieval‑Augmented Generation (RAG), [LlamaIndex](https://www.llamaindex.ai/), ChromaDB, and modern LLMs.

- 🧠 Build a local knowledge base from PDFs, Word, Markdown, text, CSV, Excel, and images
- 🔍 Ask questions in natural language and get grounded answers with sources
- 🗂️ Target specific files with `@filename.ext` filters
- 🔄 Detect, update, or fully rebuild your vector index as documents change
- 🖼️ Extract and caption images from PDFs and standalone image files

---

## Features

- **Multi‑format ingestion**
  - PDF (`.pdf`) – including images on pages
  - Text (`.txt`)
  - Markdown (`.md`)
  - Word (`.docx`)
  - CSV (`.csv`)
  - Excel (`.xlsx`)
  - Images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`)
- **Smart chunking** using `SemanticSplitterNodeParser` and sentence‑transformer embeddings
- **Vector store** powered by **ChromaDB** with cosine similarity search
- **RAG pipeline** built on **LlamaIndex**
- **LLMs**
  - Document QA: Cerebras via `llama-index-llms-cerebras`
  - Image captioning: Google Gemini (`gemini-2.5-flash-lite`)
- **Rich TUI** using `rich` for a pleasant terminal chat experience

---

## Project Structure

```text path=null start=null
.
├── config.toml          # Main project configuration (LLM, embeddings, vector store, domain path)
├── data/                # Your documents live here (PDF, TXT, MD, DOCX, CSV, XLSX, images, ...)
├── main.py              # Entry point that launches the chat CLI
├── pyproject.toml       # Python project configuration & dependencies
├── src/
│   ├── chat_cli.py      # Rich-powered interactive chat interface
│   ├── config.py        # Pydantic settings loader for config.toml
│   ├── doc_parser.py    # Document loaders & cleaners for all supported formats
│   ├── image_captioning.py  # Gemini-based image captioning utility
│   └── rag.py           # RAG pipeline, vector store, and KB management
└── tests/
    └── test_chunking.py # Tests for semantic chunking
```

---

## Installation

### 1. Clone the repo

```bash path=null start=null
git clone <your-repo-url>
cd rag_llamaindex
```

### 2. Create and activate a virtual environment (recommended)

```bash path=null start=null
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
```

### 3. Install dependencies

This project is managed via `pyproject.toml`.

```bash path=null start=null
pip install -e .
# or
uv sync
```

> If you are not using `pip install -e .`, you can instead run:
>
> ```bash path=null start=null
> pip install -r requirements.txt
> ```
>
> (Generate `requirements.txt` from `pyproject.toml` if needed.)

---

## Configuration

All runtime configuration is read from `config.toml` via `src/config.py`.

A typical `config.toml` might look like:

```toml path=null start=null
[llm]
provider = "cerebras"
model_name = "llama3.1-8b"
temperature = 0.1
api_key_env_var = "CEREBRAS_API_KEY"

[vector_store]
collection_name = "rag-docs"
path = "./.chroma"
top_k = 5

[embedding]
model_name = "sentence-transformers/all-MiniLM-L12-v2"

[domain]
# Folder where your documents live
domain_path = "./data"
```

### Environment variables

You must expose the following API keys as environment variables:

- `CEREBRAS_API_KEY` – used by the Cerebras LLM in `src/rag.py`
- `GEMINI_API_KEY` – used by Google Gemini in `src/image_captioning.py`

Example for a Unix shell:

```bash path=null start=null
export CEREBRAS_API_KEY="<your-cerebras-key>"
export GEMINI_API_KEY="<your-gemini-key>"
```

Optionally, you can also use a `.env` file; `dotenv` is used in the code to load it.

```bash path=null start=null
echo "CEREBRAS_API_KEY=..." >> .env
echo "GEMINI_API_KEY=..." >> .env
```

---

## Usage

### 1. Prepare your documents

Place any of the supported document types into the folder configured as `domain.domain_path` in `config.toml` (by default `data/`). For example:

```text path=null start=null
data/
├── handbook.pdf
├── notes.txt
├── spec.md
├── report.docx
├── metrics.csv
├── budget.xlsx
└── diagram.png
```

### 2. Launch the terminal chat

You can either run the module or call `main.py` directly:

```bash path=null start=null
python main.py
# or
python -m src.chat_cli
# or
uv run main.py
```

You will see a banner similar to:

```text path=null start=null
🔮 RAG AI Chat Terminal
Ask questions about your documents. Type 'exit' to quit.
```

### 3. Ask questions

- Just type your question and press Enter:

  ```text path=null start=null
  What are the key risks called out in the handbook?
  ```

- Target specific documents using `@filename.ext`:

  ```text path=null start=null
  Summarize the conclusions. @report.docx
  ```

- Available commands inside the chat:
  - `help`, `h`, `?` – show help
  - `update`, `upd` – detect and apply changes to existing documents
  - `rebuild`, `rb` – fully rebuild the vector index from scratch
  - `clear`, `cls` – clear the screen
  - `exit`, `quit`, `q` – exit the chat

The app also displays a **"Knowledge sources"** tree with the files and chunks that contributed to each answer, so you can quickly see where information came from.

---

## Knowledge Base Lifecycle

The RAG pipeline is managed in `src/rag.py` and supports incremental updates.

### Initial index build

When the app starts, `initialize_index()` will:

1. Connect to a ChromaDB persistent database at `vector_store.path`.
2. If a collection already has data, load a `VectorStoreIndex` from it.
3. If the collection is empty, it will read documents from `domain.domain_path`,
   parse and chunk them, and create a new index.

### Detecting and applying updates

- `check_for_updates()` compares the current state of files in `domain.domain_path`
  with a JSON state file (`kb_state.json` under `vector_store.path`).
- If there are new, modified, or deleted files, you’ll be asked whether to
  update the DB.
- `update_knowledge_base()` then:
  - Deletes chunks for removed/modified files
  - Re‑parses and inserts chunks for new/modified files

### Full rebuild

If you want to start from scratch:

- Use the `rebuild` / `rb` command inside the chat
- Or run `python -m src.rag` directly

This deletes the Chroma collection and recreates the index from all current documents.

---

## Document & Image Processing

`src/doc_parser.py` handles ingestion of different file types:

- PDFs: text extraction via PyMuPDF (`pymupdf`), plus image descriptions per page
- Markdown: converted to HTML with `markdown`, then stripped to clean text
- DOCX: paragraphs and tables are extracted
- CSV/Excel: loaded with `pandas` and converted to string
- Plain text: lightly cleaned (remove excessive whitespace, normalize characters)
- Images: passed to `caption_image()` to generate a structured caption

`src/image_captioning.py` uses Google Gemini to return a structured `Image` object with:

- `image_type`
- `image_name`
- `image_description`

These captions are then appended to the document text as additional context.

---

## Development

### Running tests

In progress...

---

## Troubleshooting

- **`CEREBRAS_API_KEY environment variable is not set.`**
  - Ensure `CEREBRAS_API_KEY` is exported or present in `.env` and that `config.toml` points to the correct env var via `api_key_env_var`.

- **`GEMINI_API_KEY` errors** or image caption failures
  - Make sure `GEMINI_API_KEY` is defined and valid; image captioning will be skipped if an error occurs.

- **No documents indexed / "No documents to index"**
  - Check that files exist under the folder pointed to by `domain.domain_path` in `config.toml` (commonly `data/`).

- **Slow or large index**
  - Adjust `vector_store.top_k` or Chroma HNSW parameters in `src/rag.py`.
  - Remove unnecessary large files from `data/`.

---

## License

Apache-2.0
