pub mod tools;

use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::HashMap, convert::Infallible, sync::Arc};
use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::CorsLayer;
use tracing::info;

use crate::config::Config;
use tools::{all_tools, execute_tool, AppState};

// ─── JSON-RPC types ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct RpcRequest {
    id:     Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct RpcResponse {
    jsonrpc: &'static str,
    id:      Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result:  Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error:   Option<RpcError>,
}

#[derive(Debug, Serialize)]
struct RpcError {
    code:    i32,
    message: String,
}

type SessionSender = mpsc::UnboundedSender<Result<Event, Infallible>>;

// ─── Shared server state ──────────────────────────────────────────────────────

#[derive(Clone)]
struct ServerState {
    sessions: Arc<RwLock<HashMap<String, SessionSender>>>,
    app:      AppState,
}

// ─── Entry point ──────────────────────────────────────────────────────────────

pub async fn start(config: Config) -> Result<()> {
    use crate::{
        db::{Database, VectorStore},
        indexer::{index_repo_into, start_watcher, Embedder},
    };
    use std::sync::Arc;

    let db = Arc::new(Database::open()?);
    let vectors = Arc::new(RwLock::new(VectorStore::new()));

    println!("\n  repo-mcp starting…\n");

    // Init embedder (downloads model on first run)
    let embedder = Embedder::init()?;

    // Index all repos (skips unchanged files via mtime)
    for repo in &config.repos {
        println!("  Indexing '{}' at {}", repo.name, repo.path);
        if let Err(e) = index_repo_into(&db, &embedder, &vectors, repo).await {
            eprintln!("  ⚠ Index error for '{}': {}", repo.name, e);
        }
    }

    // Load vectors for any already-indexed chunks
    let existing = db.load_all_vectors()?;
    if existing.len() > 0 {
        let mut vs = vectors.write().await;
        *vs = existing;
    }

    println!("\n  ✓ {} repos ready\n", config.repos.len());

    // Start file watcher
    start_watcher(
        Arc::clone(&db),
        embedder.clone(),
        Arc::clone(&vectors),
        config.repos.clone(),
    ).await?;

    // Build app state
    let app_state = AppState {
        db:      Arc::clone(&db),
        embedder,
        vectors: Arc::clone(&vectors),
        repos:   config.repos.clone(),
    };

    let state = ServerState {
        sessions: Arc::new(RwLock::new(HashMap::new())),
        app:      app_state,
    };

    let router = Router::new()
        .route("/health", get(health))
        .route("/sse",    get(sse_handler))
        .route("/message",post(message_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("127.0.0.1:{}", config.port);

    println!("  Listening on http://{}\n", addr);
    println!("  SSE endpoint: http://{}/sse", addr);
    println!("  Health:       http://{}/health\n", addr);
    println!("  Add to VSCode settings.json:");
    println!("  {{");
    println!("    \"mcp\": {{");
    println!("      \"servers\": {{");
    println!("        \"repo-mcp\": {{");
    println!("          \"type\": \"sse\",");
    println!("          \"url\": \"http://localhost:{}/sse\"", config.port);
    println!("        }}");
    println!("      }}");
    println!("    }}");
    println!("  }}\n");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

async fn health(State(state): State<ServerState>) -> Json<Value> {
    let repos: Vec<_> = state.app.repos.iter().map(|r| {
        let count = state.app.db.chunk_count(&r.name).unwrap_or(0);
        json!({ "name": r.name, "path": r.path, "chunks": count })
    }).collect();

    let vector_count = state.app.vectors.read().await.len();

    Json(json!({
        "status":  "ok",
        "repos":   repos,
        "vectors": vector_count,
    }))
}

#[derive(Deserialize)]
struct SessionParams {
    #[serde(rename = "sessionId")]
    session_id: String,
}

async fn sse_handler(
    State(state): State<ServerState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let session_id = uuid::Uuid::new_v4().to_string();
    let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();

    // Send endpoint URL as first event
    let endpoint_event = Event::default()
        .event("endpoint")
        .data(format!("/message?sessionId={}", session_id));
    let _ = tx.send(Ok(endpoint_event));

    state.sessions.write().await.insert(session_id.clone(), tx);

    info!("New session: {}", session_id);

    let stream = UnboundedReceiverStream::new(rx);
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn message_handler(
    State(state): State<ServerState>,
    Query(params): Query<SessionParams>,
    Json(request): Json<RpcRequest>,
) -> StatusCode {
    let sessions = state.sessions.read().await;
    let sender = match sessions.get(&params.session_id) {
        Some(s) => s.clone(),
        None    => return StatusCode::BAD_REQUEST,
    };
    drop(sessions);

    tokio::spawn(handle_rpc(request, sender, state));

    StatusCode::ACCEPTED
}

// ─── MCP protocol ─────────────────────────────────────────────────────────────

async fn handle_rpc(req: RpcRequest, sender: SessionSender, state: ServerState) {
    let id = req.id.unwrap_or(Value::Null);

    let response = match req.method.as_str() {
        "initialize" => RpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "repo-mcp", "version": "0.1.0" }
            })),
            error: None,
        },

        "initialized" | "notifications/initialized" => return, // no response

        "ping" => RpcResponse {
            jsonrpc: "2.0", id, result: Some(json!({})), error: None,
        },

        "tools/list" => {
            let tools = all_tools(&state.app.repos);
            let list: Vec<_> = tools.iter().map(|t| json!({
                "name":        t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            })).collect();
            RpcResponse {
                jsonrpc: "2.0", id,
                result: Some(json!({ "tools": list })),
                error: None,
            }
        }

        "tools/call" => {
            let params = req.params.unwrap_or(json!({}));
            let tool_name = params["name"].as_str().unwrap_or("").to_string();
            let args = params.get("arguments").cloned().unwrap_or(json!({}));

            let result = execute_tool(&state.app, &tool_name, &args).await;

            match result {
                Ok(v) => RpcResponse {
                    jsonrpc: "2.0", id, result: Some(v), error: None,
                },
                Err(e) => RpcResponse {
                    jsonrpc: "2.0", id, result: None,
                    error: Some(RpcError { code: -32603, message: e.to_string() }),
                },
            }
        }

        _ => RpcResponse {
            jsonrpc: "2.0", id, result: None,
            error: Some(RpcError {
                code: -32601,
                message: format!("Method not found: {}", req.method),
            }),
        },
    };

    let event = Event::default()
        .event("message")
        .data(serde_json::to_string(&response).unwrap_or_default());

    let _ = sender.send(Ok(event));
}
