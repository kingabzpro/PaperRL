//! Alpaca paper credentials for the CLI and the desktop dashboard.
//!
//! Resolution order is the standard environment variables first, then the file
//! the desktop app saves, so a shell or CI export can never be shadowed by an
//! older file. The file is created owner-only on Unix; on Windows it inherits
//! the user-profile ACL. Nothing here can reach the live-money endpoint:
//! `paperrl-alpaca` hardcodes the paper API.

use std::path::{Path, PathBuf};

use paperrl_alpaca::Credentials;
use paperrl_core::{Error, Result};

/// File name inside the app data directory.
pub const CREDENTIALS_FILE: &str = "credentials.json";

/// Keys as persisted on disk. `Debug` redacts both halves so a stray log line
/// or panic message cannot print a usable key.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SavedCredentials {
    pub key_id: String,
    pub secret_key: String,
}

impl std::fmt::Debug for SavedCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SavedCredentials")
            .field("key_id", &mask(&self.key_id))
            .field("secret_key", &"<redacted>")
            .finish()
    }
}

/// Where a usable key pair came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialSource {
    /// `APCA_API_KEY_ID` and `APCA_API_SECRET`; wins over the saved file.
    Environment,
    /// `credentials.json` in the app data directory, used when the environment
    /// variables are absent.
    Saved,
    /// Neither is available.
    Missing,
}

/// Shows only the last four characters so the UI can identify a key without
/// displaying it.
pub fn mask(value: &str) -> String {
    let trimmed = value.trim();
    let count = trimmed.chars().count();
    if count == 0 {
        return String::new();
    }
    if count <= 4 {
        return "…".to_string();
    }
    let tail: String = trimmed.chars().skip(count - 4).collect();
    format!("…{tail}")
}

/// Path of the credentials file inside the app data directory.
pub fn credentials_path(home: &Path) -> PathBuf {
    home.join(CREDENTIALS_FILE)
}

/// Reads saved keys. A missing, empty, or unparsable file yields `None`, so a
/// half-written file can never produce a client with one blank half.
pub fn load_credentials(home: &Path) -> Option<SavedCredentials> {
    let raw = std::fs::read_to_string(credentials_path(home)).ok()?;
    let saved: SavedCredentials = serde_json::from_str(&raw).ok()?;
    let key_id = saved.key_id.trim().to_string();
    let secret_key = saved.secret_key.trim().to_string();
    if key_id.is_empty() || secret_key.is_empty() {
        return None;
    }
    Some(SavedCredentials { key_id, secret_key })
}

/// Writes both keys to `<home>/credentials.json` with owner-only permissions,
/// replacing whatever was saved before.
pub fn save_credentials(home: &Path, key_id: &str, secret_key: &str) -> Result<PathBuf> {
    let key_id = key_id.trim();
    let secret_key = secret_key.trim();
    if key_id.is_empty() || secret_key.is_empty() {
        return Err(Error::MissingCredentials(
            "both the Alpaca paper key id and the secret are required".into(),
        ));
    }
    std::fs::create_dir_all(home)?;
    let path = credentials_path(home);
    let json = serde_json::to_string_pretty(&SavedCredentials {
        key_id: key_id.to_string(),
        secret_key: secret_key.to_string(),
    })
    .map_err(|e| Error::Other(e.to_string()))?;
    std::fs::write(&path, json)?;
    restrict_permissions(&path)?;
    Ok(path)
}

/// Removes saved keys; the environment variables still apply afterwards.
/// Returns whether a file was actually deleted.
pub fn clear_credentials(home: &Path) -> std::io::Result<bool> {
    match std::fs::remove_file(credentials_path(home)) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

/// Which source a usable key pair would come from right now. The environment
/// wins, so a shell or CI export is never shadowed by an older saved file.
pub fn credential_source(home: &Path) -> CredentialSource {
    let env_keys = std::env::var("APCA_API_KEY_ID")
        .ok()
        .filter(|v| !v.trim().is_empty());
    let env_secret = std::env::var("APCA_API_SECRET")
        .ok()
        .filter(|v| !v.trim().is_empty());
    match (env_keys, env_secret) {
        (Some(_), Some(_)) => CredentialSource::Environment,
        _ if load_credentials(home).is_some() => CredentialSource::Saved,
        _ => CredentialSource::Missing,
    }
}

/// Credentials for an API call: `APCA_API_KEY_ID` and `APCA_API_SECRET` first,
/// then the keys saved from the desktop Settings tab.
pub fn resolve_credentials(home: &Path) -> Result<Credentials> {
    if credential_source(home) == CredentialSource::Environment {
        return Credentials::from_env();
    }
    if let Some(saved) = load_credentials(home) {
        return Credentials::from_parts(Some(saved.key_id), Some(saved.secret_key));
    }
    Err(Error::MissingCredentials(
        "no Alpaca paper keys: set APCA_API_KEY_ID and APCA_API_SECRET, or save them in Settings"
            .into(),
    ))
}

#[cfg(unix)]
fn restrict_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the tests that read or write process environment variables.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn env_keys_present() -> bool {
        std::env::var("APCA_API_KEY_ID").is_ok_and(|v| !v.trim().is_empty())
            && std::env::var("APCA_API_SECRET").is_ok_and(|v| !v.trim().is_empty())
    }

    #[test]
    fn round_trip_save_load_resolve_clear() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        assert!(load_credentials(home).is_none());
        let env_present = env_keys_present();
        assert_eq!(
            credential_source(home),
            if env_present { CredentialSource::Environment } else { CredentialSource::Missing }
        );

        let path = save_credentials(home, " PKTEST1234 ", " secret-value ").unwrap();
        assert_eq!(path, home.join(CREDENTIALS_FILE));
        let saved = load_credentials(home).unwrap();
        assert_eq!(saved.key_id, "PKTEST1234");
        assert_eq!(saved.secret_key, "secret-value");
        assert_eq!(
            credential_source(home),
            if env_present { CredentialSource::Environment } else { CredentialSource::Saved }
        );
        if !env_present {
            let creds = resolve_credentials(home).unwrap();
            assert_eq!(creds.key_id, "PKTEST1234");
            assert_eq!(creds.secret_key, "secret-value");
        }

        assert!(clear_credentials(home).unwrap());
        assert!(!clear_credentials(home).unwrap());
        assert!(load_credentials(home).is_none());
    }

    #[test]
    fn environment_variables_win_over_the_saved_file() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        save_credentials(dir.path(), "FILETEST1234", "file-secret").unwrap();

        // SAFETY: ENV_LOCK serializes every test that touches these variables,
        // and no test or library call reads them concurrently.
        unsafe {
            std::env::set_var("APCA_API_KEY_ID", " ENVTEST5678 ");
            std::env::set_var("APCA_API_SECRET", "env-secret");
        }
        let resolved = resolve_credentials(dir.path());
        assert_eq!(credential_source(dir.path()), CredentialSource::Environment);
        unsafe {
            std::env::remove_var("APCA_API_KEY_ID");
            std::env::remove_var("APCA_API_SECRET");
        }

        let creds = resolved.unwrap();
        assert_eq!(creds.key_id, "ENVTEST5678");
        assert_eq!(creds.secret_key, "env-secret");
        // With the variables gone, the saved file applies again.
        assert_eq!(credential_source(dir.path()), CredentialSource::Saved);
        let fallback = resolve_credentials(dir.path()).unwrap();
        assert_eq!(fallback.key_id, "FILETEST1234");
        assert_eq!(fallback.secret_key, "file-secret");
    }

    #[test]
    fn blank_keys_are_rejected_and_blank_files_ignored() {
        let dir = tempfile::tempdir().unwrap();
        assert!(save_credentials(dir.path(), "  ", "secret").is_err());
        assert!(save_credentials(dir.path(), "key", "").is_err());
        std::fs::write(
            credentials_path(dir.path()),
            r#"{"key_id":"","secret_key":"x"}"#,
        )
        .unwrap();
        assert!(load_credentials(dir.path()).is_none());
        std::fs::write(credentials_path(dir.path()), "not json").unwrap();
        assert!(load_credentials(dir.path()).is_none());
    }

    #[test]
    fn mask_keeps_only_the_tail() {
        assert_eq!(mask("PKABCDEF1234"), "…1234");
        assert_eq!(mask("abc"), "…");
        assert_eq!(mask("   "), "");
    }

    #[test]
    fn debug_output_redacts_the_secret() {
        let saved = SavedCredentials {
            key_id: "PKTEST1234".into(),
            secret_key: "super-secret".into(),
        };
        let rendered = format!("{saved:?}");
        assert!(!rendered.contains("super-secret"));
        assert!(!rendered.contains("PKTEST1234"));
        assert!(rendered.contains("…1234"));
    }

    #[cfg(unix)]
    #[test]
    fn saved_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = save_credentials(dir.path(), "key", "secret").unwrap();
        let mode = std::fs::metadata(path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}
