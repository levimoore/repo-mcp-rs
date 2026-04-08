# repo-mcp

A local MCP server written in Rust that indexes your code repos with embeddings
and exposes them to VSCode agents — so they use targeted context instead of
reading whole files.

## How it reduces tokens

Without repo-mcp, an agent reads entire files to answer a question.
With repo-mcp, it calls `semantic_search("authentication flow")` and gets back
only the 3-5 relevant functions — typically 95%+ token reduction.

## Quick Install (prebuilt binary)

```bash
# Download the binary
curl -L https://github.com/levimoore/repo-mcp-rs/releases/download/v0.1.0/repo-mcp-arm64 \
  -o /usr/local/bin/repo-mcp

# Make it executable
chmod +x /usr/local/bin/repo-mcp

# Remove Gatekeeper quarantine (required for unsigned binaries)
xattr -dr com.apple.quarantine /usr/local/bin/repo-mcp

# Verify it works
repo-mcp --help
```

## Prerequisites

- Rust (https://rustup.rs)

## Install (from source)

```bash
git clone <repo>
cd repo-mcp
cargo build --release
sudo cp target/release/repo-mcp /usr/local/bin/
```

The embedding model downloads automatically on first `repo-mcp index` and is
cached at `~/.cache/huggingface`. After that, everything runs fully offline.

## Usage

```bash
# 1. Register your repo
repo-mcp add ~/projects/myapp

# 2. Build the index (downloads embedding model on first run)
repo-mcp index myapp

# 3. Start the server
repo-mcp start
```

```bash
# Other commands
repo-mcp list                  # show repos + index status
repo-mcp status                # is the server running?
repo-mcp stop                  # stop the server
repo-mcp index                 # re-index all repos
repo-mcp remove myapp          # unregister
repo-mcp config set port 7777
repo-mcp config show
```

## Connect to VSCode

Add to your VSCode `settings.json`:

```json
{
  "mcp": {
    "servers": {
      "repo-mcp": {
        "type": "sse",
        "url": "http://localhost:3742/sse"
      }
    }
  }
}
```

## Tools (per repo)

For a repo registered as `myapp`:

| Tool | What it does | Token cost |
|------|-------------|------------|
| `myapp__semantic_search` | Meaning-based search — "find auth logic" | ✦ lowest |
| `myapp__find_symbol` | Exact symbol lookup by name | ✦ lowest |
| `myapp__get_outline` | File structure, no bodies | ✦ low |
| `myapp__read_lines` | Specific line range | ✦ low |
| `myapp__read_file` | Full file (smart truncation at 200 lines) | medium |
| `myapp__get_tree` | Directory structure | low |
| `myapp__search` | ripgrep text/regex search | low |

## Choosing an embedding model

repo-mcp uses [fastembed](https://github.com/Anush008/fastembed-rs) for local,
offline embeddings. The model is configurable — you can set a persistent default
or override it for a single index run.

The model name is the PascalCase enum variant name from fastembed's
`EmbeddingModel` and is matched **case-insensitively**, so `BGESmallENV15`,
`bgesmallenv15`, and `BGESMALLENV15` all work.

### Recommended models for code search

| Model | Dims | Download | Notes |
|-------|------|----------|-------|
| `JinaEmbeddingsV2BaseCode` | 768 | ~320 MB | **Best for code.** Trained on code + natural language pairs. 8192-token context window handles large functions without truncation. |
| `BGESmallENV15` | 384 | ~130 MB | Good general-purpose retrieval. Outperforms MiniLM on MTEB benchmarks. Fast and memory-efficient. |
| `AllMiniLML6V2` | 384 | ~90 MB | **Default.** Solid general baseline. Widely used, very fast. Switch to `JinaEmbeddingsV2BaseCode` for better code search. |
| `NomicEmbedTextV15` | 768 | ~275 MB | Strong NL retrieval with 8192-token context. Good if your codebase has heavy inline documentation. |
| `BGELargeENV15` | 1024 | ~1.3 GB | Highest quality general retrieval in the list. Significantly slower and larger — worth it only for large, infrequently re-indexed repos. |

The default model is `AllMiniLML6V2` — a fast, reliable baseline. For dedicated
code search, switching to `JinaEmbeddingsV2BaseCode` is recommended.
`BGESmallENV15` is a good middle ground if you want better retrieval without
the larger download.

The full list of supported models is available in the
[fastembed docs](https://docs.rs/fastembed/latest/fastembed/enum.EmbeddingModel.html).

### Setting the model

**Persist a model in config** (used for all future `index` and `start` runs):

```bash
repo-mcp config set embedding_model JinaEmbeddingsV2BaseCode
```

**Override for a single index run** without changing the config:

```bash
repo-mcp index --model JinaEmbeddingsV2BaseCode
repo-mcp index myapp --model BGESmallENV15
```

**Check the current model:**

```bash
repo-mcp config show
repo-mcp status        # also shows the configured model when the server is running
```

### ⚠️ Changing models requires a full re-index

Embedding vectors are only comparable within the same model. If you change the
model, you **must** re-index before starting the server, otherwise semantic
search will produce incorrect results:

```bash
repo-mcp config set embedding_model JinaEmbeddingsV2BaseCode
repo-mcp index        # re-index everything with the new model
repo-mcp start
```

## What's in the index

- SQLite database at `~/.repo-mcp/index.db`
- Every function, class, method, type, and interface is a chunk
- Each chunk has a vector embedding whose dimension depends on the chosen model
- Files are watched and re-indexed on save automatically

## Building a distributable binary

```bash
cargo build --release
# → target/release/repo-mcp  (~15-20 MB static binary)
```

The binary is self-contained. The only runtime requirements on a target machine:
- macOS (arm64 or x86_64)
- Internet access on first run (to download the embedding model)