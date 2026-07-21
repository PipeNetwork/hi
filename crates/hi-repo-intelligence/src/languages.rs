//! Multi-language support via tree-sitter.
//!
//! Provides language detection by file extension and tree-sitter query
//! definitions for symbol extraction (functions, classes, methods, imports).
//!
//! Inspired by grok-build's `xai-codebase-graph` language registry pattern.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Supported languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LanguageId {
    Rust,
    Python,
    Go,
    JavaScript,
    TypeScript,
}

impl LanguageId {
    pub fn as_str(self) -> &'static str {
        match self {
            LanguageId::Rust => "rust",
            LanguageId::Python => "python",
            LanguageId::Go => "go",
            LanguageId::JavaScript => "javascript",
            LanguageId::TypeScript => "typescript",
        }
    }
}

/// Configuration for a tree-sitter language: grammar, extensions, and symbol
/// extraction queries.
pub struct LanguageConfig {
    pub id: LanguageId,
    pub extensions: Vec<&'static str>,
    pub grammar: fn() -> tree_sitter::Language,
    /// Tree-sitter query for extracting definition symbols.
    pub definition_query: &'static str,
}

impl LanguageConfig {
    /// Parse a source file and extract symbol definitions.
    pub fn extract_symbols(&self, source: &str) -> Vec<SymbolDef> {
        let language = (self.grammar)();
        let mut parser = tree_sitter::Parser::new();
        if parser.set_language(&language).is_err() {
            return Vec::new();
        }
        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => return Vec::new(),
        };
        let query = match tree_sitter::Query::new(&language, self.definition_query) {
            Ok(q) => q,
            Err(_) => return Vec::new(),
        };
        let mut cursor = tree_sitter::QueryCursor::new();
        let mut matches = cursor.matches(&query, tree.root_node(), source.as_bytes());

        let mut symbols = Vec::new();
        use tree_sitter::StreamingIterator;
        while let Some(m) = matches.next() {
            for cap in m.captures {
                if let Some(name) = cap.node.utf8_text(source.as_bytes()).ok() {
                    let kind = cap.index;
                    let line = cap.node.start_position().row as u32 + 1;
                    let symbol_kind = query.capture_names().get(kind as usize).copied().unwrap_or("unknown");
                    symbols.push(SymbolDef {
                        name: name.to_string(),
                        kind: symbol_kind.to_string(),
                        line,
                    });
                }
            }
        }
        symbols
    }
}

/// A symbol definition extracted from source.
#[derive(Debug, Clone)]
pub struct SymbolDef {
    pub name: String,
    pub kind: String,
    pub line: u32,
}

/// Registry of all supported languages, with lookup by extension.
pub struct LanguageRegistry {
    by_extension: HashMap<String, Arc<LanguageConfig>>,
}

impl LanguageRegistry {
    /// Create a new registry with all supported languages.
    pub fn new() -> Self {
        let configs: Vec<Arc<LanguageConfig>> = vec![
            Arc::new(rust_config()),
            Arc::new(python_config()),
            Arc::new(go_config()),
            Arc::new(javascript_config()),
            Arc::new(typescript_config()),
        ];

        let mut by_extension = HashMap::new();
        for config in &configs {
            for ext in &config.extensions {
                by_extension.insert(ext.to_string(), Arc::clone(config));
            }
        }

        Self {
            by_extension,
        }
    }

    /// Get a language config by file extension (without the leading dot).
    pub fn for_extension(&self, ext: &str) -> Option<Arc<LanguageConfig>> {
        self.by_extension.get(ext).cloned()
    }

    /// Get a language config for a file path.
    pub fn for_file_path(&self, path: &Path) -> Option<Arc<LanguageConfig>> {
        let ext = path.extension()?.to_str()?;
        self.for_extension(ext)
    }

    /// Check if a file path is supported.
    pub fn is_supported(&self, path: &Path) -> bool {
        self.for_file_path(path).is_some()
    }

    /// All supported file extensions.
    pub fn supported_extensions(&self) -> Vec<&str> {
        self.by_extension.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for LanguageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// --- Language configurations ---

fn rust_config() -> LanguageConfig {
    LanguageConfig {
        id: LanguageId::Rust,
        extensions: vec!["rs"],
        grammar: || tree_sitter_rust::LANGUAGE.into(),
        definition_query: r#"
            (function_item name: (identifier) @function)
            (function_signature_item name: (identifier) @function)
            (struct_item name: (type_identifier) @struct)
            (enum_item name: (type_identifier) @enum)
            (trait_item name: (type_identifier) @trait)
            (impl_item type: (type_identifier) @impl)
            (mod_item name: (identifier) @module)
        "#,
    }
}

fn python_config() -> LanguageConfig {
    LanguageConfig {
        id: LanguageId::Python,
        extensions: vec!["py"],
        grammar: || tree_sitter_python::LANGUAGE.into(),
        definition_query: r#"
            (function_definition name: (identifier) @function)
            (class_definition name: (identifier) @class)
        "#,
    }
}

fn go_config() -> LanguageConfig {
    LanguageConfig {
        id: LanguageId::Go,
        extensions: vec!["go"],
        grammar: || tree_sitter_go::LANGUAGE.into(),
        definition_query: r#"
            (function_declaration name: (identifier) @function)
            (method_declaration name: (field_identifier) @method)
            (type_declaration (type_spec name: (type_identifier) @type))
        "#,
    }
}

fn javascript_config() -> LanguageConfig {
    LanguageConfig {
        id: LanguageId::JavaScript,
        extensions: vec!["js", "jsx", "mjs", "cjs"],
        grammar: || tree_sitter_javascript::LANGUAGE.into(),
        definition_query: r#"
            (function_declaration name: (identifier) @function)
            (class_declaration name: (identifier) @class)
            (method_definition name: (property_identifier) @method)
            (variable_declarator name: (identifier) @variable)
        "#,
    }
}

fn typescript_config() -> LanguageConfig {
    LanguageConfig {
        id: LanguageId::TypeScript,
        extensions: vec!["ts", "tsx"],
        grammar: || tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        definition_query: r#"
            (function_declaration name: (identifier) @function)
            (class_declaration name: (type_identifier) @class)
            (method_definition name: (property_identifier) @method)
            (interface_declaration name: (type_identifier) @interface)
            (type_alias_declaration name: (type_identifier) @type)
        "#,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_detects_by_extension() {
        let reg = LanguageRegistry::new();
        assert!(reg.for_file_path(Path::new("src/main.rs")).is_some());
        assert!(reg.for_file_path(Path::new("app.py")).is_some());
        assert!(reg.for_file_path(Path::new("main.go")).is_some());
        assert!(reg.for_file_path(Path::new("index.js")).is_some());
        assert!(reg.for_file_path(Path::new("app.ts")).is_some());
        assert!(reg.for_file_path(Path::new("README.md")).is_none());
    }

    #[test]
    fn rust_extracts_functions_and_structs() {
        let config = rust_config();
        let source = r#"
            pub fn hello() {}
            struct Foo { x: i32 }
            trait Bar {}
        "#;
        let symbols = config.extract_symbols(source);
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"hello"), "found symbols: {symbols:?}");
        assert!(names.contains(&"Foo"));
        assert!(names.contains(&"Bar"));
    }

    #[test]
    fn python_extracts_functions_and_classes() {
        let config = python_config();
        let source = r#"
            def hello():
                pass

            class Foo:
                def method(self):
                    pass
        "#;
        let symbols = config.extract_symbols(source);
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"hello"));
        assert!(names.contains(&"Foo"));
        assert!(names.contains(&"method"));
    }

    #[test]
    fn go_extracts_functions_and_types() {
        let config = go_config();
        let source = r#"
            func main() {}
            type Foo struct { x int }
        "#;
        let symbols = config.extract_symbols(source);
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"main"));
        assert!(names.contains(&"Foo"));
    }

    #[test]
    fn javascript_extracts_functions_and_classes() {
        let config = javascript_config();
        let source = r#"
            function hello() {}
            class Foo {
                bar() {}
            }
        "#;
        let symbols = config.extract_symbols(source);
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"hello"));
        assert!(names.contains(&"Foo"));
    }
}
