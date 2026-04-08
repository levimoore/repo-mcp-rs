mod config;
mod daemon;
mod db;
mod indexer;
mod server;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use config::{Config, RepoConfig};
use daemon::{clear_pid, is_running, read_pid, write_pid};

#[derive(Parser)]
#[command(
    name = "repo-mcp",
    about = "Local MCP server — indexes your repos so VSCode agents use fewer tokens",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create the config file at ~/.repo-mcp/config.json
    Init,

    /// Register a local repo
    Add {
        path: PathBuf,
        /// Namespace alias (default: directory name)
        #[arg(long)]
        name: Option<String>,
    },

    /// Unregister a repo by name
    Remove { name: String },

    /// Show registered repos and their index status
    List,

    /// Build or refresh the index (run this after adding a repo)
    Index {
        /// Index only this repo (default: all)
        name: Option<String>,
        /// Embedding model to use for this run (overrides config).
        /// E.g. AllMiniLML6V2, BGESmallENV15, NomicEmbedTextV15
        #[arg(long, value_name = "MODEL")]
        model: Option<String>,
    },

    /// Start the MCP server (foreground)
    Start,

    /// Stop the running server
    Stop,

    /// Show server status
    Status,

    /// Manage config values
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// Set a config value (e.g. `config set port 7777` or `config set embedding_model BGESmallENV15`)
    Set { key: String, value: String },
    /// Print the current config
    Show,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init => cmd_init(),
        Commands::Add { path, name } => cmd_add(path, name),
        Commands::Remove { name } => cmd_remove(name),
        Commands::List => cmd_list(),
        Commands::Index { name, model } => cmd_index(name, model).await,
        Commands::Start => cmd_start().await,
        Commands::Stop => cmd_stop(),
        Commands::Status => cmd_status(),
        Commands::Config { command } => match command {
            ConfigCommand::Set { key, value } => cmd_config_set(key, value),
            ConfigCommand::Show => cmd_config_show(),
        },
    }
}

// ─── Commands ─────────────────────────────────────────────────────────────────

fn cmd_init() -> Result<()> {
    let config = Config::load()?;
    config.save()?;
    println!("✓ Config ready at {}", Config::path().display());
    println!("  Next: repo-mcp add <path-to-repo>");
    Ok(())
}

fn cmd_add(path: PathBuf, name: Option<String>) -> Result<()> {
    let path = path
        .canonicalize()
        .map_err(|_| anyhow::anyhow!("Path does not exist: {}", path.display()))?;

    let name = name.unwrap_or_else(|| {
        path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect()
    });

    let mut config = Config::load()?;

    if config.repos.iter().any(|r| r.name == name) {
        anyhow::bail!(
            "Repo '{}' already registered.\n  Use --name <alias> to choose a different namespace.",
            name
        );
    }

    config.repos.push(RepoConfig {
        name: name.clone(),
        path: path.to_string_lossy().to_string(),
    });
    config.save()?;

    println!("✓ Added '{}'  →  {}", name, path.display());
    println!("  Build the index:  repo-mcp index {}", name);
    Ok(())
}

fn cmd_remove(name: String) -> Result<()> {
    let mut config = Config::load()?;
    let before = config.repos.len();
    config.repos.retain(|r| r.name != name);

    if config.repos.len() == before {
        anyhow::bail!(
            "No repo named '{}'. Run 'repo-mcp list' to see what's registered.",
            name
        );
    }

    config.save()?;
    println!("✓ Removed '{}'", name);
    Ok(())
}

fn cmd_list() -> Result<()> {
    let config = Config::load()?;

    if config.repos.is_empty() {
        println!("No repos registered.");
        println!("  repo-mcp add <path>");
        return Ok(());
    }

    let db = db::Database::open()?;

    println!("{:<24}  {:<40}  {}", "NAME", "PATH", "INDEX");
    println!("{}", "─".repeat(72));
    for r in &config.repos {
        let exists = std::path::Path::new(&r.path).exists();
        let chunks = db.chunk_count(&r.name).unwrap_or(0);
        let status = if !exists {
            "⚠ path missing".to_string()
        } else if chunks == 0 {
            "not indexed  →  run: repo-mcp index".to_string()
        } else {
            format!("{} chunks indexed", chunks)
        };
        println!("{:<24}  {:<40}  {}", r.name, r.path, status);
    }

    Ok(())
}

async fn cmd_index(name: Option<String>, model: Option<String>) -> Result<()> {
    let config = Config::load()?;

    let repos: Vec<&RepoConfig> = match &name {
        Some(n) => {
            let r = config
                .repos
                .iter()
                .find(|r| &r.name == n)
                .ok_or_else(|| anyhow::anyhow!("No repo named '{}'", n))?;
            vec![r]
        }
        None => config.repos.iter().collect(),
    };

    if repos.is_empty() {
        anyhow::bail!("No repos registered. Run: repo-mcp add <path>");
    }

    // --model flag overrides the persisted config value for this run only
    let model_name = model.as_deref().unwrap_or(&config.embedding_model);

    let db = std::sync::Arc::new(db::Database::open()?);
    let embedder = indexer::Embedder::init(model_name)?;
    let vectors = std::sync::Arc::new(tokio::sync::RwLock::new(db::VectorStore::new()));

    for repo in repos {
        println!("\n  Indexing '{}'…", repo.name);
        indexer::index_repo_into(&db, &embedder, &vectors, repo).await?;
    }

    println!("\n✓ Done. Start the server: repo-mcp start");
    Ok(())
}

async fn cmd_start() -> Result<()> {
    if let Some(pid) = read_pid() {
        if is_running(pid) {
            println!("Already running (PID {}). Use 'repo-mcp stop' first.", pid);
            return Ok(());
        }
        let _ = clear_pid();
    }

    let config = Config::load()?;

    if config.repos.is_empty() {
        anyhow::bail!("No repos registered.\n  repo-mcp add <path>  then  repo-mcp index");
    }

    write_pid(std::process::id())?;

    // Handle Ctrl+C cleanly
    let cleanup = move || {
        let _ = clear_pid();
        std::process::exit(0);
    };
    ctrlc::set_handler(cleanup)?;

    server::start(config).await?;

    Ok(())
}

fn cmd_stop() -> Result<()> {
    match read_pid() {
        Some(pid) if is_running(pid) => {
            std::process::Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .status()?;
            let _ = clear_pid();
            println!("✓ Stopped (PID {})", pid);
        }
        _ => {
            println!("Not running.");
            let _ = clear_pid();
        }
    }
    Ok(())
}

fn cmd_status() -> Result<()> {
    let config = Config::load()?;
    match read_pid() {
        Some(pid) if is_running(pid) => {
            println!("Status:  running");
            println!("PID:     {}", pid);
            println!("Port:    {}", config.port);
            println!("Model:   {}", config.embedding_model);
            println!(
                "Repos:   {}",
                config
                    .repos
                    .iter()
                    .map(|r| r.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!("Health:  http://localhost:{}/health", config.port);
        }
        _ => {
            println!("Status:  stopped");
        }
    }
    Ok(())
}

fn cmd_config_set(key: String, value: String) -> Result<()> {
    use fastembed::EmbeddingModel;
    use std::str::FromStr as _;

    let mut config = Config::load()?;
    match key.as_str() {
        "port" => {
            let port: u16 = value.parse().map_err(|_| anyhow::anyhow!("Invalid port"))?;
            config.port = port;
            config.save()?;
            println!("✓ port = {}", port);
        }
        "embedding_model" => {
            // Validate the model name before saving
            EmbeddingModel::from_str(&value)
                .map_err(|e| anyhow::anyhow!("Unknown embedding model '{}': {}\n  Run `repo-mcp index --model <MODEL>` to see available names.", value, e))?;
            config.embedding_model = value.clone();
            config.save()?;
            println!("✓ embedding_model = {}", value);
            println!("  Re-index your repos to apply: repo-mcp index");
        }
        _ => anyhow::bail!("Unknown key '{}'. Valid keys: port, embedding_model", key),
    }
    Ok(())
}

fn cmd_config_show() -> Result<()> {
    let config = Config::load()?;
    println!("{}", serde_json::to_string_pretty(&config)?);
    Ok(())
}
