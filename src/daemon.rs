use anyhow::Result;
use std::path::PathBuf;

fn pid_path() -> PathBuf {
    crate::config::Config::dir().join("daemon.pid")
}

pub fn write_pid(pid: u32) -> Result<()> {
    std::fs::create_dir_all(crate::config::Config::dir())?;
    std::fs::write(pid_path(), pid.to_string())?;
    Ok(())
}

pub fn read_pid() -> Option<u32> {
    std::fs::read_to_string(pid_path())
        .ok()?
        .trim()
        .parse()
        .ok()
}

pub fn clear_pid() -> Result<()> {
    let p = pid_path();
    if p.exists() {
        std::fs::remove_file(p)?;
    }
    Ok(())
}

/// Check if a PID is alive by sending signal 0 (macOS/Linux).
pub fn is_running(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
