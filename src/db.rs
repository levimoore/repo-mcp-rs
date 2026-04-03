use anyhow::Result;
use rusqlite::{Connection, params};
use std::sync::Mutex;

// ─── Schema ──────────────────────────────────────────────────────────────────

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS repos (
    id          INTEGER PRIMARY KEY,
    name        TEXT    UNIQUE NOT NULL,
    path        TEXT    NOT NULL,
    added_at    INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY,
    repo_id     INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    rel_path    TEXT    NOT NULL,
    mtime       INTEGER NOT NULL,
    summary     TEXT,
    indexed_at  INTEGER NOT NULL DEFAULT (unixepoch()),
    UNIQUE(repo_id, rel_path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    repo_name   TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    name        TEXT,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    content     TEXT    NOT NULL,
    signature   TEXT,
    embedding   BLOB
);

CREATE INDEX IF NOT EXISTS idx_chunks_repo   ON chunks(repo_name);
CREATE INDEX IF NOT EXISTS idx_chunks_name   ON chunks(name);
CREATE INDEX IF NOT EXISTS idx_files_repo    ON files(repo_id);
"#;

// Migrations applied once — errors ignored (column already exists).
const MIGRATIONS: &[&str] = &[
    "ALTER TABLE files ADD COLUMN summary TEXT",
    "ALTER TABLE chunks ADD COLUMN signature TEXT",
];

// ─── Public types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Chunk {
    pub _id:        i64,
    pub repo_name:  String,
    pub kind:       String,
    pub name:       Option<String>,
    pub start_line: u32,
    pub end_line:   u32,
    pub content:    String,
    pub signature:  Option<String>,
    pub rel_path:   String,
}

// ─── In-memory vector store ───────────────────────────────────────────────────

pub struct VectorStore {
    chunk_ids:  Vec<i64>,
    embeddings: Vec<Vec<f32>>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self { chunk_ids: vec![], embeddings: vec![] }
    }

    pub fn insert(&mut self, chunk_id: i64, embedding: Vec<f32>) {
        self.chunk_ids.push(chunk_id);
        self.embeddings.push(normalize(embedding));
    }

    pub fn remove_by_ids(&mut self, ids: &[i64]) {
        let id_set: std::collections::HashSet<i64> = ids.iter().copied().collect();
        let mut i = 0;
        while i < self.chunk_ids.len() {
            if id_set.contains(&self.chunk_ids[i]) {
                self.chunk_ids.swap_remove(i);
                self.embeddings.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(i64, f32)> {
        let q = normalize(query.to_vec());
        let mut scores: Vec<(i64, f32)> = self.chunk_ids.iter()
            .zip(self.embeddings.iter())
            .map(|(&id, emb)| (id, dot(&q, emb)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    pub fn len(&self) -> usize { self.chunk_ids.len() }
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 { v.iter_mut().for_each(|x| *x /= norm); }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn embedding_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_embedding(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ─── Database ─────────────────────────────────────────────────────────────────

pub struct Database {
    conn: Mutex<Connection>,
}

impl Database {
    pub fn open() -> Result<Self> {
        let path = crate::config::Config::db_path();
        std::fs::create_dir_all(path.parent().unwrap())?;
        let conn = Connection::open(&path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        conn.execute_batch(SCHEMA)?;
        for migration in MIGRATIONS {
            let _ = conn.execute_batch(migration);
        }
        Ok(Self { conn: Mutex::new(conn) })
    }

    // ── Repos ────────────────────────────────────────────────────────────────

    pub fn upsert_repo(&self, name: &str, path: &str) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO repos(name, path) VALUES(?1, ?2)
             ON CONFLICT(name) DO UPDATE SET path=excluded.path",
            params![name, path],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn repo_id(&self, name: &str) -> Result<Option<i64>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id FROM repos WHERE name = ?1")?;
        let mut rows = stmt.query(params![name])?;
        Ok(rows.next()?.map(|r| r.get(0).unwrap()))
    }

    pub fn chunk_count(&self, repo_name: &str) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE repo_name = ?1",
            params![repo_name],
            |r| r.get(0),
        )?;
        Ok(count)
    }

    // ── Files ────────────────────────────────────────────────────────────────

    pub fn upsert_file(&self, repo_id: i64, rel_path: &str, mtime: i64) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO files(repo_id, rel_path, mtime) VALUES(?1, ?2, ?3)
             ON CONFLICT(repo_id, rel_path) DO UPDATE SET mtime=excluded.mtime, indexed_at=unixepoch()",
            params![repo_id, rel_path, mtime],
        )?;
        let file_id: i64 = conn.query_row(
            "SELECT id FROM files WHERE repo_id=?1 AND rel_path=?2",
            params![repo_id, rel_path],
            |r| r.get(0),
        )?;
        Ok(file_id)
    }

    pub fn file_mtime(&self, repo_id: i64, rel_path: &str) -> Result<Option<i64>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT mtime FROM files WHERE repo_id=?1 AND rel_path=?2"
        )?;
        let mut rows = stmt.query(params![repo_id, rel_path])?;
        Ok(rows.next()?.map(|r| r.get(0).unwrap()))
    }

    pub fn file_chunk_ids(&self, file_id: i64) -> Result<Vec<i64>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id FROM chunks WHERE file_id=?1")?;
        let ids = stmt.query_map(params![file_id], |r| r.get(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ids)
    }

    pub fn delete_file_chunks(&self, file_id: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM chunks WHERE file_id=?1", params![file_id])?;
        Ok(())
    }

    pub fn set_file_summary(&self, file_id: i64, summary: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE files SET summary=?1 WHERE id=?2",
            params![summary, file_id],
        )?;
        Ok(())
    }

    pub fn get_file_summary(&self, repo_name: &str, rel_path: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT f.summary FROM files f
             JOIN repos r ON r.id = f.repo_id
             WHERE r.name=?1 AND f.rel_path=?2"
        )?;
        let mut rows = stmt.query(params![repo_name, rel_path])?;
        Ok(rows.next()?.and_then(|r| r.get(0).ok()))
    }

    // ── Chunks ───────────────────────────────────────────────────────────────

    pub fn insert_chunk(
        &self,
        file_id: i64,
        repo_name: &str,
        kind: &str,
        name: Option<&str>,
        start_line: u32,
        end_line: u32,
        content: &str,
        signature: Option<&str>,
    ) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO chunks(file_id, repo_name, kind, name, start_line, end_line, content, signature)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
            params![file_id, repo_name, kind, name, start_line, end_line, content, signature],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn set_embedding(&self, chunk_id: i64, embedding: &[f32]) -> Result<()> {
        let blob = embedding_to_blob(embedding);
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE chunks SET embedding=?1 WHERE id=?2",
            params![blob, chunk_id],
        )?;
        Ok(())
    }

    pub fn load_all_vectors(&self) -> Result<VectorStore> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
        )?;
        let mut store = VectorStore::new();
        let rows = stmt.query_map([], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
        })?;
        for row in rows {
            let (id, blob) = row?;
            store.insert(id, blob_to_embedding(&blob));
        }
        Ok(store)
    }

    pub fn get_chunk(&self, id: i64) -> Result<Option<Chunk>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT c.id, c.repo_name, c.kind, c.name, c.start_line, c.end_line,
                    c.content, c.signature, f.rel_path
             FROM chunks c
             JOIN files f ON f.id = c.file_id
             WHERE c.id = ?1"
        )?;
        let mut rows = stmt.query(params![id])?;
        Ok(rows.next()?.map(|r| Chunk {
            _id:        r.get(0).unwrap(),
            repo_name:  r.get(1).unwrap(),
            kind:       r.get(2).unwrap(),
            name:       r.get(3).unwrap(),
            start_line: r.get::<_, u32>(4).unwrap(),
            end_line:   r.get::<_, u32>(5).unwrap(),
            content:    r.get(6).unwrap(),
            signature:  r.get(7).unwrap(),
            rel_path:   r.get(8).unwrap(),
        }))
    }

    pub fn find_symbol(&self, repo_name: &str, name: &str) -> Result<Vec<Chunk>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT c.id, c.repo_name, c.kind, c.name, c.start_line, c.end_line,
                    c.content, c.signature, f.rel_path
             FROM chunks c
             JOIN files f ON f.id = c.file_id
             WHERE c.repo_name=?1 AND LOWER(c.name)=LOWER(?2)"
        )?;
        let rows = stmt.query_map(params![repo_name, name], |r| Ok(Chunk {
            _id:        r.get(0)?,
            repo_name:  r.get(1)?,
            kind:       r.get(2)?,
            name:       r.get(3)?,
            start_line: r.get(4)?,
            end_line:   r.get(5)?,
            content:    r.get(6)?,
            signature:  r.get(7)?,
            rel_path:   r.get(8)?,
        }))?.collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Find chunks that reference a symbol — content contains the name but the
    /// chunk is not itself the definition of that name.
    pub fn find_references(&self, repo_name: &str, symbol: &str) -> Result<Vec<Chunk>> {
        let conn = self.conn.lock().unwrap();
        let pattern = format!("%{}%", symbol);
        let mut stmt = conn.prepare(
            "SELECT c.id, c.repo_name, c.kind, c.name, c.start_line, c.end_line,
                    c.content, c.signature, f.rel_path
             FROM chunks c
             JOIN files f ON f.id = c.file_id
             WHERE c.repo_name=?1
               AND c.content LIKE ?2
               AND (c.name IS NULL OR LOWER(c.name) != LOWER(?3))
             ORDER BY f.rel_path, c.start_line"
        )?;
        let rows = stmt.query_map(params![repo_name, pattern, symbol], |r| Ok(Chunk {
            _id:        r.get(0)?,
            repo_name:  r.get(1)?,
            kind:       r.get(2)?,
            name:       r.get(3)?,
            start_line: r.get(4)?,
            end_line:   r.get(5)?,
            content:    r.get(6)?,
            signature:  r.get(7)?,
            rel_path:   r.get(8)?,
        }))?.collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn chunks_for_file(&self, repo_name: &str, rel_path: &str) -> Result<Vec<Chunk>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT c.id, c.repo_name, c.kind, c.name, c.start_line, c.end_line,
                    c.content, c.signature, f.rel_path
             FROM chunks c
             JOIN files f ON f.id = c.file_id
             WHERE c.repo_name=?1 AND f.rel_path=?2
             ORDER BY c.start_line"
        )?;
        let rows = stmt.query_map(params![repo_name, rel_path], |r| Ok(Chunk {
            _id:        r.get(0)?,
            repo_name:  r.get(1)?,
            kind:       r.get(2)?,
            name:       r.get(3)?,
            start_line: r.get(4)?,
            end_line:   r.get(5)?,
            content:    r.get(6)?,
            signature:  r.get(7)?,
            rel_path:   r.get(8)?,
        }))?.collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn unembedded_chunks(&self, repo_name: &str) -> Result<Vec<(i64, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, content FROM chunks WHERE repo_name=?1 AND embedding IS NULL"
        )?;
        let rows = stmt.query_map(params![repo_name], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?.collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }
}
