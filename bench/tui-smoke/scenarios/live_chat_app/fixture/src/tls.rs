use std::sync::Arc;
use tokio_rustls::TlsAcceptor;

/// TLS configuration loaded from `CHAT_TLS_CERT` / `CHAT_TLS_KEY` PEM files.
/// Returns `None` when neither variable is set (plaintext mode).
pub fn tls_acceptor_from_env() -> Option<std::io::Result<TlsAcceptor>> {
    let cert_path = std::env::var("CHAT_TLS_CERT").ok()?;
    let key_path = std::env::var("CHAT_TLS_KEY").ok()?;

    Some(load_acceptor(&cert_path, &key_path))
}

/// Build a `TlsAcceptor` from PEM-encoded cert and key files.
pub fn load_acceptor(cert_path: &str, key_path: &str) -> std::io::Result<TlsAcceptor> {
    let certs = load_certs(cert_path)?;
    let key = load_key(key_path)?;

    let config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| std::io::Error::other(format!("invalid TLS cert/key: {e}")))?;
    Ok(TlsAcceptor::from(Arc::new(config)))
}

fn load_certs(path: &str) -> std::io::Result<Vec<rustls::pki_types::CertificateDer<'static>>> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let certs: Result<Vec<_>, _> = rustls_pemfile::certs(&mut reader).collect();
    certs.map_err(|e| std::io::Error::other(format!("failed to parse certs in {path}: {e}")))
}

fn load_key(path: &str) -> std::io::Result<rustls::pki_types::PrivateKeyDer<'static>> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    rustls_pemfile::private_key(&mut reader)?
        .ok_or_else(|| std::io::Error::other(format!("no private key found in {path}")))
}
