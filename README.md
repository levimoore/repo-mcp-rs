# repo-mcp

A local MCP server written in Rust that indexes your code repos with embeddings
and exposes them to VSCode agents — so they use targeted context instead of
reading whole files.

## How it reduces tokens

Without repo-mcp, an agent reads entire files to answer a question.
With repo-mcp, it calls `semantic_search("authentication flow")` and gets back
only the 3-5 relevant functions — typically 95%+ token reduction.

## Prerequisites

- Rust (https://rustup.rs)

## Install

```bash
git clone <repo>
cd repo-mcp
cargo build --release
sudo cp target/release/repo-mcp /usr/local/bin/
```

The embedding model (~90 MB, all-MiniLM-L6-v2) downloads automatically on
first `repo-mcp index` and is cached at `~/.repo-mcp/models/`.
After that, everything runs fully offline.

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

## What's in the index

- SQLite database at `~/.repo-mcp/index.db`
- Every function, class, method, type, and interface is a chunk
- Each chunk has a 384-dimensional embedding (all-MiniLM-L6-v2 via ONNX)
- Files are watched and re-indexed on save automatically

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

## Building a distributable binary

```bash
cargo build --release
# → target/release/repo-mcp  (~15-20 MB static binary)
```

The binary is self-contained. The only runtime requirements on a target machine:
- macOS (arm64 or x86_64)
- Internet access on first run (to download the embedding model)
