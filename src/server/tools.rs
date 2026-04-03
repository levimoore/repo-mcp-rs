use anyhow::Result;
use serde_json::{json, Value};
use std::{path::Path, sync::Arc};
use tokio::sync::RwLock;

use crate::{config::RepoConfig, db::{Database, VectorStore}, indexer::Embedder};

// ─── App state ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub db:       Arc<Database>,
    pub embedder: Embedder,
    pub vectors:  Arc<RwLock<VectorStore>>,
    pub repos:    Vec<RepoConfig>,
}

// ─── Tool descriptors ─────────────────────────────────────────────────────────

pub struct Tool {
    pub name:         String,
    pub description:  String,
    pub input_schema: Value,
}

pub fn all_tools(repos: &[RepoConfig]) -> Vec<Tool> {
    let mut tools = Vec::new();
    for r in repos {
        let n = &r.name;
        tools.extend([
            Tool {
                name: format!("{n}__semantic_search"),
                description: format!(
                    "Search {n} by meaning using the pre-built embedding index. \
                     Unlike the built-in search, this operates only on {n}'s indexed code, \
                     returns symbol-level chunks with exact file paths and line ranges, \
                     and understands code concepts ('error handling', 'database connection') \
                     not just keywords. Use this first when you don't know where something lives."
                ),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of the code you're looking for"
                        },
                        "limit": {
                            "type": "number",
                            "description": "Max results to return (default: 5)"
                        }
                    },
                    "required": ["query"]
                }),
            },
            Tool {
                name: format!("{n}__find_symbol"),
                description: format!(
                    "Instantly retrieve the full definition of a named function, class, method, \
                     or type from {n}'s index. Returns the exact code block and its location. \
                     Use when you know the name — eliminates the grep → open file → find it pattern."
                ),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Exact or case-insensitive symbol name"
                        }
                    },
                    "required": ["name"]
                }),
            },
            Tool {
                name: format!("{n}__find_references"),
                description: format!(
                    "Find all functions and methods in {n} that call or reference a given symbol. \
                     Returns the calling code with file and line ranges — not just grep matches, \
                     but the full surrounding context. Use before modifying a symbol to understand \
                     its blast radius."
                ),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Symbol name to find references to"
                        },
                        "limit": {
                            "type": "number",
                            "description": "Max results (default: 10)"
                        }
                    },
                    "required": ["symbol"]
                }),
            },
            Tool {
                name: format!("{n}__get_outline"),
                description: format!(
                    "Get the full structure of a file in {n}: every symbol with its kind, \
                     complete signature (including parameters and return type), and line range — \
                     without the body. Use this to understand a file before deciding what to read. \
                     Much cheaper than opening the file."
                ),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to repo root"
                        }
                    },
                    "required": ["path"]
                }),
            },
            Tool {
                name: format!("{n}__get_file_summary"),
                description: format!(
                    "Get a one-line cached description of what a file in {n} does, plus a \
                     list of all its top-level symbols. Zero cost — generated at index time. \
                     Use when exploring unfamiliar parts of the codebase to decide which files \
                     are worth reading."
                ),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to repo root"
                        }
                    },
                    "required": ["path"]
                }),
            },
            Tool {
                name: format!("{n}__read_file"),
                description: format!(
                    "Read a file from {n} with line numbers. Files over 200 lines are \
                     automatically truncated — you get the first 100 lines plus an index-powered \
                     symbol map of the rest (names, kinds, line ranges) so you can make a \
                     targeted follow-up read."
                ),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to repo root"
                        }
                    },
                    "required": ["path"]
                }),
            },
        ]);
    }
    tools
}

// ─── Dispatch ─────────────────────────────────────────────────────────────────

pub async fn execute_tool(state: &AppState, tool_name: &str, args: &Value) -> Result<Value> {
    let Some((prefix, verb)) = tool_name.rsplit_once("__") else {
        return Ok(tool_error("Unknown tool format"));
    };
    let Some(repo) = state.repos.iter().find(|r| r.name == prefix) else {
        return Ok(tool_error(&format!("No repo named '{}'", prefix)));
    };

    match verb {
        "semantic_search"  => semantic_search(state, repo, args).await,
        "find_symbol"      => find_symbol(state, repo, args).await,
        "find_references"  => find_references(state, repo, args).await,
        "get_outline"      => get_outline(state, repo, args).await,
        "get_file_summary" => get_file_summary(state, repo, args).await,
        "read_file"        => read_file(state, repo, args).await,
        _                  => Ok(tool_error(&format!("Unknown tool '{}'", tool_name))),
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn tool_text(text: impl Into<String>) -> Value {
    json!({ "content": [{ "type": "text", "text": text.into() }] })
}

fn tool_error(msg: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": format!("Error: {}", msg) }], "isError": true })
}

fn numbered(lines: &[&str], start: usize) -> String {
    lines.iter()
        .enumerate()
        .map(|(i, l)| format!("{:5}  {}", start + i, l))
        .collect::<Vec<_>>()
        .join("\n")
}

fn safe_path(repo_path: &str, rel: &str) -> Result<std::path::PathBuf> {
    let full = Path::new(repo_path).join(rel).canonicalize()
        .unwrap_or_else(|_| Path::new(repo_path).join(rel));
    if !full.starts_with(repo_path) {
        anyhow::bail!("Path '{}' escapes repo root", rel);
    }
    Ok(full)
}

// ─── semantic_search ──────────────────────────────────────────────────────────

async fn semantic_search(state: &AppState, repo: &RepoConfig, args: &Value) -> Result<Value> {
    let query = args["query"].as_str().unwrap_or("").to_string();
    let limit = args["limit"].as_u64().unwrap_or(5) as usize;

    if query.is_empty() { return Ok(tool_error("query is required")); }

    let embeddings = state.embedder.embed(vec![query.clone()]).await?;
    let q_emb = &embeddings[0];

    let candidates = state.vectors.read().await.search(q_emb, limit * 4);

    let mut output = Vec::new();
    for (chunk_id, _score) in candidates {
        let Some(chunk) = state.db.get_chunk(chunk_id)? else { continue };
        if chunk.repo_name != repo.name { continue; }

        // Show signature if available, otherwise show first few lines of content
        let display = chunk.signature.as_deref().unwrap_or_else(|| {
            chunk.content.lines().next().unwrap_or("")
        });

        output.push(format!(
            "── {path}:{start}-{end}  [{kind}] {name}\n   {sig}",
            path  = chunk.rel_path,
            start = chunk.start_line,
            end   = chunk.end_line,
            kind  = chunk.kind,
            name  = chunk.name.as_deref().unwrap_or(""),
            sig   = display,
        ));

        if output.len() >= limit { break; }
    }

    if output.is_empty() {
        return Ok(tool_text(format!(
            "No matches for '{}' in {}. Try find_symbol if you know the name.",
            query, repo.name
        )));
    }

    Ok(tool_text(output.join("\n\n")))
}

// ─── find_symbol ──────────────────────────────────────────────────────────────

async fn find_symbol(state: &AppState, repo: &RepoConfig, args: &Value) -> Result<Value> {
    let name = args["name"].as_str().unwrap_or("");
    if name.is_empty() { return Ok(tool_error("name is required")); }

    let chunks = state.db.find_symbol(&repo.name, name)?;

    if chunks.is_empty() {
        return Ok(tool_text(format!(
            "Symbol '{}' not found in {}. Try semantic_search for fuzzy matching.",
            name, repo.name
        )));
    }

    let mut parts = Vec::new();
    for c in &chunks {
        parts.push(format!(
            "── {path}:{start}-{end}  [{kind}]\n{content}",
            path    = c.rel_path,
            start   = c.start_line,
            end     = c.end_line,
            kind    = c.kind,
            content = c.content,
        ));
    }

    Ok(tool_text(parts.join("\n\n")))
}

// ─── find_references ──────────────────────────────────────────────────────────

async fn find_references(state: &AppState, repo: &RepoConfig, args: &Value) -> Result<Value> {
    let symbol = args["symbol"].as_str().unwrap_or("");
    let limit  = args["limit"].as_u64().unwrap_or(10) as usize;

    if symbol.is_empty() { return Ok(tool_error("symbol is required")); }

    let chunks = state.db.find_references(&repo.name, symbol)?;

    if chunks.is_empty() {
        return Ok(tool_text(format!(
            "No references to '{}' found in {}.",
            symbol, repo.name
        )));
    }

    let total = chunks.len();
    let _shown = chunks.len().min(limit);

    let mut parts = vec![format!(
        "{} reference(s) to '{}' in {}{}:\n",
        total, symbol, repo.name,
        if total > limit { format!(" (showing first {})", limit) } else { String::new() }
    )];

    for c in chunks.into_iter().take(limit) {
        let caller = c.name.as_deref().unwrap_or("(anonymous)");
        parts.push(format!(
            "── {path}:{start}-{end}  [{kind}] {caller}\n   {sig}",
            path   = c.rel_path,
            start  = c.start_line,
            end    = c.end_line,
            kind   = c.kind,
            caller = caller,
            sig    = c.signature.as_deref().unwrap_or(""),
        ));
    }

    Ok(tool_text(parts.join("\n")))
}

// ─── get_outline ──────────────────────────────────────────────────────────────

async fn get_outline(state: &AppState, repo: &RepoConfig, args: &Value) -> Result<Value> {
    let path = args["path"].as_str().unwrap_or("");
    if path.is_empty() { return Ok(tool_error("path is required")); }

    let chunks = state.db.chunks_for_file(&repo.name, path)?;

    if chunks.is_empty() {
        let full = safe_path(&repo.path, path)?;
        if !full.exists() {
            return Ok(tool_error(&format!("File not found: {}", path)));
        }
        return Ok(tool_text(format!(
            "{} — not indexed yet. Run 'repo-mcp index' to index it.",
            path
        )));
    }

    let mut lines = vec![
        format!("{}  ({} symbols)\n{}", path, chunks.len(), "─".repeat(60)),
        format!("{:<8}  {:<12}  {}", "LINES", "KIND", "SIGNATURE"),
        "─".repeat(60),
    ];

    for c in &chunks {
        let sig = c.signature.as_deref()
            .or(c.name.as_deref())
            .unwrap_or("(anonymous)");
        lines.push(format!(
            "{:<8}  {:<12}  {}",
            format!("{}-{}", c.start_line, c.end_line),
            c.kind,
            sig,
        ));
    }

    Ok(tool_text(lines.join("\n")))
}

// ─── get_file_summary ─────────────────────────────────────────────────────────

async fn get_file_summary(state: &AppState, repo: &RepoConfig, args: &Value) -> Result<Value> {
    let path = args["path"].as_str().unwrap_or("");
    if path.is_empty() { return Ok(tool_error("path is required")); }

    let summary = state.db.get_file_summary(&repo.name, path)?;
    let chunks  = state.db.chunks_for_file(&repo.name, path)?;

    if summary.is_none() && chunks.is_empty() {
        let full = safe_path(&repo.path, path)?;
        if !full.exists() {
            return Ok(tool_error(&format!("File not found: {}", path)));
        }
        return Ok(tool_text(format!(
            "{} — not indexed yet. Run 'repo-mcp index'.",
            path
        )));
    }

    let mut out = vec![format!("{}:", path)];

    if let Some(s) = summary {
        out.push(format!("  {}", s));
    }

    if !chunks.is_empty() {
        out.push(String::new());
        out.push("  Symbols:".to_string());

        // Group by kind
        let mut by_kind: std::collections::BTreeMap<&str, Vec<&str>> =
            std::collections::BTreeMap::new();
        for c in &chunks {
            if let Some(name) = c.name.as_deref() {
                by_kind.entry(&c.kind).or_default().push(name);
            }
        }

        for (kind, names) in &by_kind {
            out.push(format!("    {} — {}", kind, names.join(", ")));
        }
    }

    Ok(tool_text(out.join("\n")))
}

// ─── read_file ────────────────────────────────────────────────────────────────

const FULL_THRESHOLD: usize = 200;
const HEAD_LINES:     usize = 100;

async fn read_file(state: &AppState, repo: &RepoConfig, args: &Value) -> Result<Value> {
    let rel = args["path"].as_str().unwrap_or("");
    if rel.is_empty() { return Ok(tool_error("path is required")); }

    let full = safe_path(&repo.path, rel)?;
    let source = match std::fs::read_to_string(&full) {
        Ok(s)  => s,
        Err(e) => return Ok(tool_error(&format!("Cannot read {}: {}", rel, e))),
    };

    let lines: Vec<&str> = source.lines().collect();

    if lines.len() <= FULL_THRESHOLD {
        return Ok(tool_text(numbered(&lines, 1)));
    }

    // Large file: head + index-powered symbol map of the tail
    let head      = numbered(&lines[..HEAD_LINES], 1);
    let remaining = lines.len() - HEAD_LINES;

    // Pull tail symbols from the index
    let tail_symbols: Vec<String> = state.db
        .chunks_for_file(&repo.name, rel)
        .unwrap_or_default()
        .into_iter()
        .filter(|c| c.start_line as usize > HEAD_LINES)
        .map(|c| {
            let sig = c.signature.as_deref()
                .or(c.name.as_deref())
                .unwrap_or("(anonymous)");
            format!("  {:5}-{:<5}  [{:<10}]  {}", c.start_line, c.end_line, c.kind, sig)
        })
        .collect();

    let tail_section = if tail_symbols.is_empty() {
        "  (no indexed symbols in remaining lines)".to_string()
    } else {
        tail_symbols.join("\n")
    };

    let summary = format!(
        "\n── {} more lines ({} total) ──────────────────────────────\n\
         Symbols in remaining lines (use find_symbol or read_lines for details):\n{}",
        remaining, lines.len(), tail_section
    );

    Ok(tool_text(format!("{}{}", head, summary)))
}
