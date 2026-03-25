# claude-augment

An MCP (Model Context Protocol) server that gives Claude Code semantic search over any codebase — **without sending your code to any external API**. It uses local Ollama embeddings and ChromaDB to index your repo on-disk, then lets Claude retrieve only the relevant chunks when answering questions.

---

## Why this saves tokens

When you ask Claude about your codebase, the naive approach is to paste entire files into the context window. That burns tokens fast:

| Approach | Tokens consumed (typical 50k-line repo) |
|---|---|
| Paste all files into context | ~200,000–500,000 tokens per session |
| This MCP server (top-5 chunks) | ~1,000–3,000 tokens per query |

**Savings: 99%+ reduction in tokens per codebase query.**

Instead of Claude reading every file, it issues a semantic search and receives only the most relevant 5 code chunks (configurable). For a large repo this translates directly into lower cost and faster responses.

---

## How it works

1. **Indexing** — your repo's files are chunked (512 tokens, 50-token overlap) and embedded locally using `nomic-embed-text` via Ollama. Embeddings are stored in a persistent ChromaDB database on disk.
2. **Search** — when Claude needs to understand your code, it calls `search_codebase` with a natural language query. The query is embedded and the top-k most similar chunks are returned.
3. **Auto-index** — the MCP server detects the current working directory and auto-indexes it on the first search if it hasn't been indexed yet.

Everything runs **100% locally** — no code leaves your machine.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with:
  - `nomic-embed-text` model (for embeddings)
  - `llama3.2` model (used by LlamaIndex internally)

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/claude-augment.git
cd claude-augment
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Register the MCP server with Claude Code

Add the following to your Claude Code MCP config (`~/.claude/claude_desktop_config.json` or via `claude mcp add`):

```json
{
  "mcpServers": {
    "codebase-search": {
      "command": "python",
      "args": ["/absolute/path/to/claude-augment/mcp_server.py"]
    }
  }
}
```

Or using the CLI:

```bash
claude mcp add codebase-search python /absolute/path/to/claude-augment/mcp_server.py
```

### 4. Start using it

Open any project in your terminal and start Claude Code. On your first search query, the server will auto-index the current repo. After that, Claude will automatically search your codebase semantically before answering questions about it.

---

## MCP Tools exposed

| Tool | Description |
|---|---|
| `search_codebase` | Semantic search over the indexed repo. Auto-indexes on first use. |
| `reindex_repo` | Re-index the current repo from scratch (run after major changes). |
| `index_status` | Show how many chunks are indexed for the current repo. |
| `list_indexed_repos` | List all repos indexed so far and their chunk counts. |

---

## Manual indexing (optional)

You can pre-index a repo manually before opening it in Claude:

```bash
python index_repo.py /path/to/your/repo
# With a custom collection name:
python index_repo.py /path/to/your/repo --collection my_project
```

---

## Supported file types

`.py` `.ts` `.tsx` `.js` `.jsx` `.go` `.java` `.rs` `.cpp` `.c` `.h` `.rb` `.php` `.swift` `.kt` `.cs` `.md` `.txt` `.yaml` `.yml` `.toml` `.json` `.sh` `.sql` `.ipynb`
