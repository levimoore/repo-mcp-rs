pub mod chunker;

use anyhow::Result;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::{path::Path, sync::Arc, time::SystemTime};
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn};
use walkdir::WalkDir;

use crate::{
    config::RepoConfig,
    db::{Database, VectorStore},
};
use chunker::{chunk_file, chunk_text_file, detect_language};

// ─── Skip lists ──────────────────────────────────────────────────────────────

const SKIP_DIRS: &[&str] = &[
    ".git", "node_modules", ".next", ".nuxt", "dist", "build", "out",
    "__pycache__", ".cache", ".turbo", "coverage", "target", ".cargo",
    ".venv", "venv", "env", ".mypy_cache", ".pytest_cache",
];

const SKIP_EXTS: &[&str] = &[
    "lock", "sum", "png", "jpg", "jpeg", "gif", "svg", "ico",
    "woff", "woff2", "ttf", "eot", "pdf", "zip", "gz", "tar",
    "mp4", "mp3", "webm", "wav", "db", "sqlite", "bin", "exe",
];

const MAX_FILE_BYTES: u64 = 500_000;

fn should_skip_dir(name: &str) -> bool {
    SKIP_DIRS.contains(&name)
}

fn should_skip_file(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        if SKIP_EXTS.contains(&ext.to_lowercase().as_str()) { return true; }
    }
    if let Ok(meta) = path.metadata() {
        if meta.len() > MAX_FILE_BYTES { return true; }
    }
    false
}

// ─── Embedding service ────────────────────────────────────────────────────────

struct EmbedRequest {
    texts:    Vec<String>,
    response: tokio::sync::oneshot::Sender<Result<Vec<Vec<f32>>>>,
}

#[derive(Clone)]
pub struct Embedder {
    tx: tokio::sync::mpsc::Sender<EmbedRequest>,
}

impl Embedder {
    pub fn init() -> Result<Self> {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<EmbedRequest>(32);

        std::thread::spawn(move || {
            use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

            println!("  Loading embedding model…");
            println!("  (Downloads ~90 MB on first run to ~/.cache/huggingface)");

            let mut model = TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                    .with_show_download_progress(true),
            )
            .expect("Failed to load embedding model");

            println!("  Embedding model ready ✓\n");

            while let Some(req) = rx.blocking_recv() {
                let texts: Vec<&str> = req.texts.iter().map(String::as_str).collect();
                let result = model
                    .embed(texts, None)
                    .map_err(|e| anyhow::anyhow!("Embed error: {}", e));
                let _ = req.response.send(result);
            }
        });

        Ok(Self { tx })
    }

    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(EmbedRequest { texts, response: resp_tx })
            .await
            .map_err(|_| anyhow::anyhow!("Embedder thread has stopped"))?;
        resp_rx
            .await
            .map_err(|_| anyhow::anyhow!("Embedder dropped the response"))?
    }
}

// ─── File summary extraction ──────────────────────────────────────────────────

/// Extract a one-line description of a file from its leading comments/docstrings
/// and its top-level symbol names.
fn extract_file_summary(source: &str, rel_path: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();

    // Try to find a leading doc comment or file-level comment
    let mut comment_lines: Vec<String> = Vec::new();
    for line in lines.iter().take(20) {
        let t = line.trim();
        // Skip shebangs and blank lines
        if t.is_empty() || t.starts_with("#!") { continue; }

        // Common doc/comment patterns
        if let Some(rest) = t.strip_prefix("///").or_else(|| t.strip_prefix("//!")) {
            comment_lines.push(rest.trim().to_string());
        } else if let Some(rest) = t.strip_prefix("//") {
            comment_lines.push(rest.trim().to_string());
        } else if let Some(rest) = t.strip_prefix('#') {
            comment_lines.push(rest.trim().to_string());
        } else if t.starts_with("\"\"\"") || t.starts_with("'''") {
            // Python docstring — grab first meaningful line
            let inner = t.trim_matches('"').trim_matches('\'').trim();
            if !inner.is_empty() { comment_lines.push(inner.to_string()); break; }
        } else if !t.starts_with("/*") {
            // Hit real code — stop
            break;
        }
    }

    // First non-empty comment line is the summary
    let comment_summary = comment_lines.into_iter()
        .find(|l| !l.is_empty() && l.len() > 3);

    if let Some(s) = comment_summary {
        // Truncate long comments
        if s.len() > 120 {
            return format!("{}…", &s[..120]);
        }
        return s;
    }

    // Fallback: list top-level symbols from the file name
    let ext = rel_path.rsplit('.').next().unwrap_or("");
    format!("{} file", ext.to_uppercase())
}

/// Extract the clean signature (first meaningful line of a symbol's content).
/// Strips opening braces, colons, and normalizes whitespace.
fn extract_signature(content: &str) -> Option<String> {
    let first = content.lines().next()?.trim();
    if first.is_empty() { return None; }
    // Strip trailing { or :
    let sig = first
        .trim_end_matches('{')
        .trim_end_matches(':')
        .trim_end();
    Some(sig.to_string())
}

// ─── Core indexing logic ──────────────────────────────────────────────────────

async fn index_file(
    db:        &Arc<Database>,
    embedder:  &Embedder,
    vectors:   &Arc<RwLock<VectorStore>>,
    repo_id:   i64,
    repo_name: &str,
    repo_path: &Path,
    file_path: &Path,
) -> Result<()> {
    let rel = file_path.strip_prefix(repo_path)
        .unwrap_or(file_path)
        .to_string_lossy()
        .to_string();

    let mtime = file_path
        .metadata().ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    if let Some(stored) = db.file_mtime(repo_id, &rel)? {
        if stored == mtime { return Ok(()); }
    }

    let source = match std::fs::read_to_string(file_path) {
        Ok(s)  => s,
        Err(_) => return Ok(()),
    };

    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let chunks = if let Some(lang) = detect_language(ext) {
        match chunk_file(&source, &lang) {
            Ok(c) if !c.is_empty() => c,
            _ => chunk_text_file(&source, 60, 10),
        }
    } else {
        chunk_text_file(&source, 60, 10)
    };

    if chunks.is_empty() { return Ok(()); }

    let file_id = db.upsert_file(repo_id, &rel, mtime)?;

    // Generate and store the file summary
    let summary = extract_file_summary(&source, &rel);
    db.set_file_summary(file_id, &summary)?;

    // Remove old vectors
    let old_ids = db.file_chunk_ids(file_id)?;
    db.delete_file_chunks(file_id)?;
    vectors.write().await.remove_by_ids(&old_ids);

    // Insert new chunks with signatures
    let mut chunk_ids   = Vec::with_capacity(chunks.len());
    let mut chunk_texts = Vec::with_capacity(chunks.len());

    for c in &chunks {
        let signature = extract_signature(&c.content);
        let id = db.insert_chunk(
            file_id, repo_name, &c.kind,
            c.name.as_deref(),
            c.start_line, c.end_line,
            &c.content,
            signature.as_deref(),
        )?;
        chunk_ids.push(id);
        chunk_texts.push(c.content.clone());
    }

    // Embed and store
    let embeddings = embedder.embed(chunk_texts).await?;
    let mut vs = vectors.write().await;
    for (id, emb) in chunk_ids.iter().zip(embeddings.into_iter()) {
        db.set_embedding(*id, &emb)?;
        vs.insert(*id, emb);
    }

    Ok(())
}

pub async fn index_repo_into(
    db:       &Arc<Database>,
    embedder: &Embedder,
    vectors:  &Arc<RwLock<VectorStore>>,
    repo:     &RepoConfig,
) -> Result<()> {
    let repo_path = Path::new(&repo.path);
    db.upsert_repo(&repo.name, &repo.path)?;
    let repo_id = db.repo_id(&repo.name)?
        .ok_or_else(|| anyhow::anyhow!("Repo not found after upsert"))?;

    let mut files = Vec::new();

    for entry in WalkDir::new(repo_path)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            if e.file_type().is_dir() {
                let name = e.file_name().to_string_lossy();
                !should_skip_dir(&name) && !name.starts_with('.')
            } else {
                !should_skip_file(e.path())
            }
        })
    {
        let entry = entry?;
        if entry.file_type().is_file() { files.push(entry.into_path()); }
    }

    let total = files.len();
    info!("  {} files to scan in '{}'", total, repo.name);

    for (i, file_path) in files.iter().enumerate() {
        if (i + 1) % 100 == 0 || i + 1 == total {
            info!("  [{}/{}] indexing…", i + 1, total);
        }
        if let Err(e) = index_file(db, embedder, vectors, repo_id, &repo.name, repo_path, file_path).await {
            warn!("  ⚠ {}: {}", file_path.display(), e);
        }
    }

    // Embed any remaining un-embedded chunks
    let pending = db.unembedded_chunks(&repo.name)?;
    if !pending.is_empty() {
        info!("  Embedding {} new chunks…", pending.len());
        let ids:  Vec<i64>   = pending.iter().map(|(id, _)| *id).collect();
        let texts: Vec<String> = pending.into_iter().map(|(_, t)| t).collect();
        let embeddings = embedder.embed(texts).await?;
        let mut vs = vectors.write().await;
        for (id, emb) in ids.iter().zip(embeddings.into_iter()) {
            db.set_embedding(*id, &emb)?;
            vs.insert(*id, emb);
        }
    }

    let count = db.chunk_count(&repo.name)?;
    info!("  ✓ '{}' indexed — {} chunks", repo.name, count);
    Ok(())
}

// ─── File watcher ─────────────────────────────────────────────────────────────

pub async fn start_watcher(
    db:      Arc<Database>,
    embedder: Embedder,
    vectors:  Arc<RwLock<VectorStore>>,
    repos:    Vec<RepoConfig>,
) -> Result<()> {
    let (tx, mut rx) = mpsc::channel::<notify::Result<Event>>(256);

    let mut watcher = RecommendedWatcher::new(
        move |res| { let _ = tx.blocking_send(res); },
        notify::Config::default(),
    )?;

    for repo in &repos {
        let path = Path::new(&repo.path);
        if path.exists() {
            watcher.watch(path, RecursiveMode::Recursive)?;
            info!("  Watching '{}'", repo.path);
        }
    }

    tokio::spawn(async move {
        let _watcher = watcher;

        while let Some(event_result) = rx.recv().await {
            let event = match event_result {
                Ok(e)  => e,
                Err(e) => { warn!("Watch error: {}", e); continue; }
            };

            if !matches!(event.kind,
                EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
            ) { continue; }

            for path in &event.paths {
                if should_skip_file(path) || path.is_dir() { continue; }

                let Some(repo) = repos.iter().find(|r| path.starts_with(&r.path)) else { continue };
                let repo_path = Path::new(&repo.path);
                let Ok(Some(repo_id)) = db.repo_id(&repo.name) else { continue };

                info!("  ↻ re-indexing: {}", path.display());
                let _ = index_file(&db, &embedder, &vectors, repo_id, &repo.name, repo_path, path).await;
            }
        }
    });

    Ok(())
}
