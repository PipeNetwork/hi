//! Tenant-isolated, tree-addressed Rust repository intelligence.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

pub mod languages;
pub use languages::{LanguageId, LanguageRegistry, LanguageConfig, SymbolDef};

pub const INDEX_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IndexKey {
    pub tenant_id: String,
    pub repository_tree_hash: String,
    pub rust_toolchain: String,
    pub analyzer_version: String,
    pub schema_version: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradationKind {
    RustAnalyzerUnavailable,
    CargoMetadataUnavailable,
    FileUnreadable,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DegradationEvidence {
    pub kind: DegradationKind,
    pub detail: String,
    pub fallback: String,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IndexSummary {
    pub files: u64,
    pub crates: u64,
    pub symbols: u64,
    pub references: u64,
    pub features: u64,
    pub targets: u64,
    pub tests: u64,
    pub degradation: Vec<DegradationEvidence>,
}

pub struct RepositoryIndex {
    connection: Connection,
    key: IndexKey,
}

impl RepositoryIndex {
    pub fn open(cache_root: &Path, key: IndexKey) -> Result<Self> {
        validate_key(&key)?;
        let tenant = cache_root.join(&key.tenant_id);
        fs::create_dir_all(&tenant)?;
        let db = tenant.join(format!(
            "{}-v{}.sqlite",
            key.repository_tree_hash, key.schema_version
        ));
        let connection = Connection::open(db)?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        initialize(&connection)?;
        Ok(Self { connection, key })
    }

    pub fn index_rust_workspace(
        &mut self,
        root: &Path,
        cargo_metadata: Option<&serde_json::Value>,
        analyzer_available: bool,
    ) -> Result<IndexSummary> {
        let root = root.canonicalize().context("canonicalizing repository")?;
        let transaction = self.connection.transaction()?;
        transaction.execute("DELETE FROM files", [])?;
        transaction.execute("DELETE FROM symbols", [])?;
        transaction.execute("DELETE FROM refs", [])?;
        transaction.execute("DELETE FROM crates", [])?;
        transaction.execute("DELETE FROM features", [])?;
        transaction.execute("DELETE FROM targets", [])?;
        transaction.execute("DELETE FROM test_affinity", [])?;
        let mut summary = IndexSummary::default();
        let mut paths = Vec::new();
        collect_files(&root, &root, &mut paths)?;
        paths.sort();
        for relative in paths {
            let absolute = root.join(&relative);
            let bytes = match fs::read(&absolute) {
                Ok(bytes) => bytes,
                Err(error) => {
                    summary.degradation.push(DegradationEvidence {
                        kind: DegradationKind::FileUnreadable,
                        detail: format!("{}: {error}", relative.display()),
                        fallback: "skip unreadable file".into(),
                    });
                    continue;
                }
            };
            let generated = is_generated(&relative, &bytes);
            let language = relative.extension().and_then(|v| v.to_str()).unwrap_or("");
            transaction.execute("INSERT INTO files(path, language, bytes, generated, content_hash) VALUES (?1,?2,?3,?4,?5)", params![relative.to_string_lossy(), language, bytes.len() as u64, generated, blake3::hash(&bytes).to_hex().to_string()])?;
            summary.files += 1;
            if language == "rs" {
                let text = String::from_utf8_lossy(&bytes);
                let file_id = transaction.last_insert_rowid();
                for (line, name, kind) in rust_symbols(&text) {
                    transaction.execute(
                        "INSERT INTO symbols(file_id,name,kind,line) VALUES (?1,?2,?3,?4)",
                        params![file_id, name, kind, line],
                    )?;
                    summary.symbols += 1;
                }
                let tests = text
                    .lines()
                    .filter(|line| line.trim_start().starts_with("#[test]"))
                    .count() as u64;
                if tests > 0 {
                    transaction.execute(
                        "INSERT INTO test_affinity(test_path,code_path,weight) VALUES (?1,?2,?3)",
                        params![
                            relative.to_string_lossy(),
                            relative.to_string_lossy(),
                            tests
                        ],
                    )?;
                    summary.tests += tests;
                }
            }
        }
        if let Some(metadata) = cargo_metadata {
            index_cargo(&transaction, metadata, &mut summary)?;
        } else {
            summary.degradation.push(DegradationEvidence {
                kind: DegradationKind::CargoMetadataUnavailable,
                detail: "cargo metadata was not supplied".into(),
                fallback: "Cargo.toml lexical retrieval".into(),
            });
        }
        if !analyzer_available {
            summary.degradation.push(DegradationEvidence {
                kind: DegradationKind::RustAnalyzerUnavailable,
                detail: "rust-analyzer index unavailable".into(),
                fallback: "lexical symbols and Cargo graph".into(),
            });
        }
        transaction.execute("INSERT OR REPLACE INTO metadata(id, tenant, tree_hash, toolchain, analyzer, schema_version, summary_json) VALUES (1,?1,?2,?3,?4,?5,?6)", params![self.key.tenant_id, self.key.repository_tree_hash, self.key.rust_toolchain, self.key.analyzer_version, self.key.schema_version, serde_json::to_string(&summary)?])?;
        transaction.commit()?;
        Ok(summary)
    }

    /// Index a multi-language repository using tree-sitter for symbol extraction.
    ///
    /// Unlike [`index_rust_workspace`], this method uses tree-sitter grammars
    /// to extract symbols from Rust, Python, Go, JavaScript, and TypeScript files.
    /// It does not index the Cargo graph or test affinity — use it alongside
    /// `index_rust_workspace` for Rust-specific metadata, or standalone for
    /// polyglot repos.
    pub fn index_polyglot(&mut self, root: &Path) -> Result<IndexSummary> {
        let root = root.canonicalize().context("canonicalizing repository")?;
        let registry = LanguageRegistry::new();
        let transaction = self.connection.transaction()?;
        transaction.execute("DELETE FROM files", [])?;
        transaction.execute("DELETE FROM symbols", [])?;
        transaction.execute("DELETE FROM refs", [])?;
        let mut summary = IndexSummary::default();
        let mut paths = Vec::new();
        collect_files(&root, &root, &mut paths)?;
        paths.sort();
        for relative in paths {
            let absolute = root.join(&relative);
            let bytes = match fs::read(&absolute) {
                Ok(bytes) => bytes,
                Err(error) => {
                    summary.degradation.push(DegradationEvidence {
                        kind: DegradationKind::FileUnreadable,
                        detail: format!("{}: {error}", relative.display()),
                        fallback: "skip unreadable file".into(),
                    });
                    continue;
                }
            };
            let language = relative.extension().and_then(|v| v.to_str()).unwrap_or("");
            let generated = is_generated(&relative, &bytes);
            transaction.execute("INSERT INTO files(path, language, bytes, generated, content_hash) VALUES (?1,?2,?3,?4,?5)", params![relative.to_string_lossy(), language, bytes.len() as u64, generated, blake3::hash(&bytes).to_hex().to_string()])?;
            summary.files += 1;

            // Use tree-sitter for supported languages.
            if let Some(config) = registry.for_file_path(&relative) {
                let text = String::from_utf8_lossy(&bytes);
                let file_id = transaction.last_insert_rowid();
                for sym in config.extract_symbols(&text) {
                    transaction.execute(
                        "INSERT INTO symbols(file_id,name,kind,line) VALUES (?1,?2,?3,?4)",
                        params![file_id, sym.name, sym.kind, sym.line],
                    )?;
                    summary.symbols += 1;
                }
            }
        }
        transaction.execute("INSERT OR REPLACE INTO metadata(id, tenant, tree_hash, toolchain, analyzer, schema_version, summary_json) VALUES (1,?1,?2,?3,?4,?5,?6)", params![self.key.tenant_id, self.key.repository_tree_hash, self.key.rust_toolchain, self.key.analyzer_version, self.key.schema_version, serde_json::to_string(&summary)?])?;
        transaction.commit()?;
        Ok(summary)
    }

    pub fn symbol_locations(&self, query: &str, limit: u32) -> Result<Vec<(PathBuf, u32, String)>> {
        ensure!(
            !query.is_empty() && limit > 0 && limit <= 1000,
            "invalid symbol query"
        );
        let mut statement = self.connection.prepare("SELECT f.path,s.line,s.kind FROM symbols s JOIN files f ON f.id=s.file_id WHERE s.name LIKE ?1 ORDER BY s.name,f.path LIMIT ?2")?;
        let rows = statement.query_map(params![format!("%{query}%"), limit], |row| {
            Ok((
                PathBuf::from(row.get::<_, String>(0)?),
                row.get(1)?,
                row.get(2)?,
            ))
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }
}

fn validate_key(key: &IndexKey) -> Result<()> {
    ensure!(
        key.schema_version == INDEX_SCHEMA_VERSION,
        "unsupported repository index schema"
    );
    ensure!(
        !key.tenant_id.is_empty()
            && key
                .tenant_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b)),
        "invalid tenant id"
    );
    ensure!(
        key.repository_tree_hash.len() == 64
            && key
                .repository_tree_hash
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "invalid tree hash"
    );
    ensure!(
        !key.rust_toolchain.is_empty() && !key.analyzer_version.is_empty(),
        "index versions are required"
    );
    Ok(())
}

fn initialize(connection: &Connection) -> Result<()> {
    connection.execute_batch("CREATE TABLE IF NOT EXISTS metadata(id INTEGER PRIMARY KEY,tenant TEXT NOT NULL,tree_hash TEXT NOT NULL,toolchain TEXT NOT NULL,analyzer TEXT NOT NULL,schema_version INTEGER NOT NULL,summary_json TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY,path TEXT NOT NULL UNIQUE,language TEXT NOT NULL,bytes INTEGER NOT NULL,generated INTEGER NOT NULL,content_hash TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS symbols(id INTEGER PRIMARY KEY,file_id INTEGER NOT NULL REFERENCES files(id),name TEXT NOT NULL,kind TEXT NOT NULL,line INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS refs(id INTEGER PRIMARY KEY,from_symbol INTEGER,to_name TEXT,kind TEXT);
      CREATE TABLE IF NOT EXISTS crates(id INTEGER PRIMARY KEY,name TEXT NOT NULL,version TEXT,manifest_path TEXT,features_json TEXT,dependencies_json TEXT);
      CREATE TABLE IF NOT EXISTS features(crate_name TEXT NOT NULL,name TEXT NOT NULL,members_json TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS targets(crate_name TEXT NOT NULL,name TEXT NOT NULL,kind_json TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS diagnostics(path TEXT,line INTEGER,severity TEXT,message TEXT);
      CREATE TABLE IF NOT EXISTS trait_impls(trait_name TEXT,type_name TEXT,path TEXT,line INTEGER);
      CREATE TABLE IF NOT EXISTS ownership(path TEXT,owner TEXT,commit_hash TEXT,changed_at INTEGER);
      CREATE TABLE IF NOT EXISTS test_affinity(test_path TEXT,code_path TEXT,weight INTEGER);")?;
    Ok(())
}

fn collect_files(root: &Path, dir: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        ensure!(!kind.is_symlink(), "repository index rejects symlinks");
        let relative = entry.path().strip_prefix(root)?.to_owned();
        if kind.is_dir() {
            if !matches!(
                relative.file_name().and_then(|v| v.to_str()),
                Some(".git" | "target")
            ) {
                collect_files(root, &entry.path(), output)?;
            }
        } else if kind.is_file() {
            output.push(relative);
        }
    }
    Ok(())
}

fn is_generated(path: &Path, bytes: &[u8]) -> bool {
    path.components().any(|c| c.as_os_str() == "generated")
        || String::from_utf8_lossy(&bytes[..bytes.len().min(512)]).contains("@generated")
}

fn rust_symbols(text: &str) -> Vec<(u32, String, &'static str)> {
    let prefixes = [
        ("fn ", "function"),
        ("struct ", "struct"),
        ("enum ", "enum"),
        ("trait ", "trait"),
        ("mod ", "module"),
    ];
    let mut output = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line
            .trim_start()
            .strip_prefix("pub ")
            .unwrap_or(line.trim_start());
        for (prefix, kind) in prefixes {
            if let Some(rest) = line.strip_prefix(prefix) {
                let name = rest
                    .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or("");
                if !name.is_empty() {
                    output.push((index as u32 + 1, name.into(), kind));
                }
            }
        }
    }
    output
}

fn index_cargo(
    tx: &rusqlite::Transaction<'_>,
    metadata: &serde_json::Value,
    summary: &mut IndexSummary,
) -> Result<()> {
    for package in metadata
        .get("packages")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        let name = package
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let version = package.get("version").and_then(|v| v.as_str());
        let manifest = package.get("manifest_path").and_then(|v| v.as_str());
        let features = package.get("features").cloned().unwrap_or_default();
        let dependencies = package.get("dependencies").cloned().unwrap_or_default();
        tx.execute("INSERT INTO crates(name,version,manifest_path,features_json,dependencies_json) VALUES (?1,?2,?3,?4,?5)", params![name, version, manifest, features.to_string(), dependencies.to_string()])?;
        summary.crates += 1;
        if let Some(map) = features.as_object() {
            for (feature, members) in map {
                tx.execute(
                    "INSERT INTO features(crate_name,name,members_json) VALUES (?1,?2,?3)",
                    params![name, feature, members.to_string()],
                )?;
                summary.features += 1;
            }
        }
        for target in package
            .get("targets")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
        {
            tx.execute(
                "INSERT INTO targets(crate_name,name,kind_json) VALUES (?1,?2,?3)",
                params![
                    name,
                    target
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown"),
                    target.get("kind").cloned().unwrap_or_default().to_string()
                ],
            )?;
            summary.targets += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn indexes_symbols_cargo_graph_and_explicit_degradation() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        fs::create_dir(&repo).unwrap();
        fs::write(
            repo.join("lib.rs"),
            "pub trait Run {}\npub fn execute() {}\n#[test]\nfn test_it() {}",
        )
        .unwrap();
        let key = IndexKey {
            tenant_id: "tenant-a".into(),
            repository_tree_hash: "a".repeat(64),
            rust_toolchain: "1.80".into(),
            analyzer_version: "none".into(),
            schema_version: 1,
        };
        let mut index = RepositoryIndex::open(&tmp.path().join("cache"), key).unwrap();
        let summary = index.index_rust_workspace(&repo, None, false).unwrap();
        assert!(summary.symbols >= 3);
        assert_eq!(summary.degradation.len(), 2);
        assert_eq!(index.symbol_locations("execute", 10).unwrap().len(), 1);
    }

    #[test]
    fn index_polyglot_extracts_symbols_from_multiple_languages() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        fs::create_dir(&repo).unwrap();
        // Rust file
        fs::write(
            repo.join("main.rs"),
            "fn rust_func() {}\nstruct RustStruct {}",
        )
        .unwrap();
        // Python file
        fs::write(
            repo.join("app.py"),
            "def python_func():\n    pass\nclass PythonClass:\n    pass\n",
        )
        .unwrap();
        // Go file
        fs::write(
            repo.join("main.go"),
            "package main\nfunc go_func() {}\ntype GoType struct { x int }\n",
        )
        .unwrap();
        // Unsupported file (should be indexed as a file but with no symbols)
        fs::write(repo.join("README.md"), "# Hello\n").unwrap();

        let key = IndexKey {
            tenant_id: "tenant-poly".into(),
            repository_tree_hash: "b".repeat(64),
            rust_toolchain: "1.80".into(),
            analyzer_version: "none".into(),
            schema_version: 1,
        };
        let mut index = RepositoryIndex::open(&tmp.path().join("cache"), key).unwrap();
        let summary = index.index_polyglot(&repo).unwrap();
        // 4 files indexed
        assert_eq!(summary.files, 4, "should index all 4 files");
        // Symbols from Rust (2) + Python (2) + Go (2) = 6
        assert!(
            summary.symbols >= 6,
            "should extract symbols from all supported languages, got {}",
            summary.symbols
        );
        // Verify cross-language symbol lookup
        let rust_syms = index.symbol_locations("rust_func", 10).unwrap();
        assert!(!rust_syms.is_empty(), "should find Rust symbol");
        let py_syms = index.symbol_locations("python_func", 10).unwrap();
        assert!(!py_syms.is_empty(), "should find Python symbol");
        let go_syms = index.symbol_locations("go_func", 10).unwrap();
        assert!(!go_syms.is_empty(), "should find Go symbol");
    }
}
