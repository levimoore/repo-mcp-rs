use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoConfig {
    pub name: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub repos: Vec<RepoConfig>,
}

fn default_port() -> u16 {
    3742
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: default_port(),
            repos: vec![],
        }
    }
}

impl Config {
    pub fn dir() -> PathBuf {
        dirs::home_dir()
            .expect("Could not find home directory")
            .join(".repo-mcp")
    }

    pub fn path() -> PathBuf {
        Self::dir().join("config.json")
    }

    pub fn load() -> Result<Self> {
        let path = Self::path();
        std::fs::create_dir_all(Self::dir())?;

        if !path.exists() {
            return Ok(Config::default());
        }

        let text = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn save(&self) -> Result<()> {
        std::fs::create_dir_all(Self::dir())?;
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(Self::path(), text)?;
        Ok(())
    }

    pub fn db_path() -> PathBuf {
        Self::dir().join("index.db")
    }
}
