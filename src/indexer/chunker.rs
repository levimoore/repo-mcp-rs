use anyhow::Result;
use tree_sitter::{Node, Parser};

const MAX_CHUNK_BYTES: usize = 8_000; // truncate content sent to embedder

#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub kind:       String,
    pub name:       Option<String>,
    pub start_line: u32,
    pub end_line:   u32,
    pub content:    String,
}

// ─── Language detection ───────────────────────────────────────────────────────

pub enum Language {
    TypeScript,
    Tsx,
    JavaScript,
    Python,
    Rust,
    Go,
}

pub fn detect_language(ext: &str) -> Option<Language> {
    match ext.to_lowercase().as_str() {
        "ts" | "gts"        => Some(Language::TypeScript),
        "tsx"               => Some(Language::Tsx),
        "js" | "mjs" | "cjs" | "jsx" | "gjs" => Some(Language::JavaScript),
        "py" | "pyi"        => Some(Language::Python),
        "rs"                => Some(Language::Rust),
        "go"                => Some(Language::Go),
        _                   => None,
    }
}

fn ts_language(lang: &Language) -> tree_sitter::Language {
    match lang {
        Language::TypeScript => tree_sitter_typescript::language_typescript(),
        Language::Tsx        => tree_sitter_typescript::language_tsx(),
        Language::JavaScript => tree_sitter_javascript::language(),
        Language::Python     => tree_sitter_python::language(),
        Language::Rust       => tree_sitter_rust::language(),
        Language::Go         => tree_sitter_go::language(),
    }
}

// ─── Node classification ──────────────────────────────────────────────────────

fn classify(kind: &str) -> Option<&'static str> {
    match kind {
        // JS/TS
        "function_declaration"
        | "generator_function_declaration"    => Some("function"),
        "class_declaration"                   => Some("class"),
        "method_definition"                   => Some("method"),
        "interface_declaration"               => Some("interface"),
        "type_alias_declaration"              => Some("type"),
        "enum_declaration"                    => Some("enum"),
        // Python
        "function_definition"                 => Some("function"),
        "class_definition"                    => Some("class"),
        // Rust
        "function_item"                       => Some("function"),
        "struct_item"                         => Some("struct"),
        "enum_item"                           => Some("enum"),
        "impl_item"                           => Some("impl"),
        "trait_item"                          => Some("trait"),
        "type_item"                           => Some("type"),
        // Go
        "method_declaration"                  => Some("method"),
        "type_declaration"                    => Some("type"),
        _                                     => None,
    }
}

/// Pull the symbol name from common field names across languages.
fn extract_name<'a>(node: &Node<'a>, source: &'a [u8]) -> Option<String> {
    // Try "name" field (most languages)
    if let Some(n) = node.child_by_field_name("name") {
        if let Ok(t) = n.utf8_text(source) {
            return Some(t.to_string());
        }
    }
    // Rust impl blocks: "impl Foo" or "impl Bar for Foo"
    if node.kind() == "impl_item" {
        // Grab the first type node as the impl target
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "type_identifier" || child.kind() == "generic_type" {
                if let Ok(t) = child.utf8_text(source) {
                    return Some(format!("impl {}", t));
                }
            }
        }
    }
    None
}

/// Extract top-level symbol chunks, then recurse into class/impl bodies
/// one level deep to also capture methods.
pub fn chunk_file(source: &str, lang: &Language) -> Result<Vec<CodeChunk>> {
    let ts_lang = ts_language(lang);
    let mut parser = Parser::new();
    parser.set_language(&ts_lang)?;

    let tree = parser
        .parse(source.as_bytes(), None)
        .ok_or_else(|| anyhow::anyhow!("tree-sitter parse returned None"))?;

    let source_bytes = source.as_bytes();
    let mut chunks = Vec::new();
    let root = tree.root_node();

    collect_chunks(root, source_bytes, source, &mut chunks, 0);

    Ok(chunks)
}

fn collect_chunks(
    node: Node,
    source_bytes: &[u8],
    source: &str,
    out: &mut Vec<CodeChunk>,
    depth: usize,
) {
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        let kind = child.kind();

        if let Some(chunk_kind) = classify(kind) {
            let name = extract_name(&child, source_bytes);
            let start = child.start_position().row as u32 + 1;
            let end   = child.end_position().row as u32 + 1;

            // Grab raw content, cap at MAX_CHUNK_BYTES for the embedding
            let raw = child.utf8_text(source_bytes).unwrap_or("").to_string();
            let content = if raw.len() > MAX_CHUNK_BYTES {
                format!("{}…[truncated]", &raw[..MAX_CHUNK_BYTES])
            } else {
                raw
            };

            out.push(CodeChunk { kind: chunk_kind.to_string(), name, start_line: start, end_line: end, content });

            // Recurse one level into containers to pick up methods / inner fns
            // (depth guard keeps it from blowing up on deeply nested code)
            if depth < 1 && matches!(kind,
                "class_declaration" | "class_definition" | "impl_item" | "trait_item"
            ) {
                collect_chunks(child, source_bytes, source, out, depth + 1);
            }
        }
    }
}

// ─── Fallback: sliding-window chunker for unsupported file types ──────────────

pub fn chunk_text_file(source: &str, window: usize, overlap: usize) -> Vec<CodeChunk> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let step = window.saturating_sub(overlap).max(1);
    let mut start = 0;

    while start < lines.len() {
        let end = (start + window).min(lines.len());
        let content = lines[start..end].join("\n");
        chunks.push(CodeChunk {
            kind: "block".to_string(),
            name: None,
            start_line: start as u32 + 1,
            end_line: end as u32,
            content,
        });
        if end == lines.len() { break; }
        start += step;
    }

    chunks
}
