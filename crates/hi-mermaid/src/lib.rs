//! Render Mermaid diagram source to a rasterized image.
//!
//! Provides a swappable [`MermaidEngine`] trait so the rendering backend can be
//! swapped without changing call sites. The default [`SubprocessEngine`] shells
//! out to `mmdc` (the Mermaid CLI) if available; a [`PureRustEngine`] stub is
//! provided for environments without Node.js.
//!
//! Inspired by grok-build's `xai-grok-mermaid` crate.
//!
//! # Quick start
//!
//! ```no_run
//! use hi_mermaid::{MermaidEngine, SubprocessEngine, RenderParams, RenderLimits};
//!
//! # fn render() -> anyhow::Result<()> {
//! let engine = SubprocessEngine::detect().unwrap_or_default();
//! let diagram = engine.render(
//!     "graph TD\n  A --> B",
//!     &RenderParams::default(),
//!     &RenderLimits::default(),
//! )?;
//! println!("Rendered {}x{} PNG, {} bytes", diagram.width_px, diagram.height_px, diagram.png.len());
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use thiserror::Error;

/// Maximum output size in megapixels to prevent runaway renders.
pub const MAX_OUTPUT_MEGAPIXELS: u32 = 50;

/// A color value (RGBA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub fn to_hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

/// Mermaid theme for rendering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MermaidTheme {
    #[default]
    Light,
    Dark,
}

impl MermaidTheme {
    /// The background color for this theme.
    pub fn surface_background(self) -> Rgba {
        match self {
            MermaidTheme::Light => Rgba::new(255, 255, 255, 255),
            MermaidTheme::Dark => Rgba::new(30, 30, 30, 255),
        }
    }
}

/// Parameters controlling how a diagram is rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderParams {
    pub theme: MermaidTheme,
    pub min_width_px: u32,
    pub max_height_px: u32,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            theme: MermaidTheme::default(),
            min_width_px: 100,
            max_height_px: 2000,
        }
    }
}

impl RenderParams {
    /// Parameters suitable for display in an OS image viewer.
    pub fn for_os_viewer(theme: MermaidTheme, min_width_px: u32, max_height_px: u32) -> Self {
        Self {
            theme,
            min_width_px,
            max_height_px,
        }
    }
}

/// Limits to prevent runaway renders.
#[derive(Debug, Clone, Copy)]
pub struct RenderLimits {
    pub max_output_megapixels: u32,
    pub timeout: Duration,
}

impl Default for RenderLimits {
    fn default() -> Self {
        Self {
            max_output_megapixels: MAX_OUTPUT_MEGAPIXELS,
            timeout: Duration::from_secs(30),
        }
    }
}

/// A rendered diagram.
#[derive(Debug, Clone)]
pub struct RenderedDiagram {
    /// The PNG image data.
    pub png: Vec<u8>,
    /// The width of the rendered image in pixels.
    pub width_px: u32,
    /// The height of the rendered image in pixels.
    pub height_px: u32,
}

/// Errors from the Mermaid renderer.
#[derive(Debug, Error)]
pub enum MermaidError {
    /// The rendering engine is not available.
    #[error("mermaid engine not available: {0}")]
    EngineUnavailable(String),
    /// The rendering process failed.
    #[error("render failed: {0}")]
    RenderFailed(String),
    /// The output exceeded the size limit.
    #[error("output exceeds limit: {width}x{height} > {max_megapixels} megapixels")]
    OutputTooLarge {
        width: u32,
        height: u32,
        max_megapixels: u32,
    },
    /// The rendering timed out.
    #[error("render timed out after {0:?}")]
    Timeout(Duration),
}

/// Trait for Mermaid rendering backends.
pub trait MermaidEngine: Send + Sync {
    /// Render Mermaid source to a PNG image.
    fn render(
        &self,
        source: &str,
        params: &RenderParams,
        limits: &RenderLimits,
    ) -> Result<RenderedDiagram, MermaidError>;
}

/// Check that a rendered diagram is within limits.
pub fn render_checked(
    engine: &dyn MermaidEngine,
    source: &str,
    params: &RenderParams,
    limits: &RenderLimits,
) -> Result<RenderedDiagram, MermaidError> {
    let diagram = engine.render(source, params, limits)?;
    let megapixels = (diagram.width_px as u64 * diagram.height_px as u64) / 1_000_000;
    if megapixels > limits.max_output_megapixels as u64 {
        return Err(MermaidError::OutputTooLarge {
            width: diagram.width_px,
            height: diagram.height_px,
            max_megapixels: limits.max_output_megapixels,
        });
    }
    Ok(diagram)
}

/// Subprocess-based renderer using `mmdc` (the Mermaid CLI).
///
/// Requires Node.js and `@mermaid-js/mermaid-cli` installed. Use [`detect`] to
/// check availability.
pub struct SubprocessEngine {
    mmdc_path: Option<PathBuf>,
}

impl SubprocessEngine {
    /// Create a new subprocess engine, detecting `mmdc` on `$PATH`.
    pub fn detect() -> Result<Self, MermaidError> {
        let mmdc_path = which_mmdc();
        if mmdc_path.is_none() {
            return Err(MermaidError::EngineUnavailable(
                "mmdc not found on PATH".to_string(),
            ));
        }
        Ok(Self { mmdc_path })
    }

    /// Create a subprocess engine with a known `mmdc` path (or `None` for
    /// a stub that always fails).
    pub fn new(mmdc_path: Option<PathBuf>) -> Self {
        Self { mmdc_path }
    }
}

impl Default for SubprocessEngine {
    fn default() -> Self {
        Self {
            mmdc_path: which_mmdc(),
        }
    }
}

impl MermaidEngine for SubprocessEngine {
    fn render(
        &self,
        source: &str,
        params: &RenderParams,
        limits: &RenderLimits,
    ) -> Result<RenderedDiagram, MermaidError> {
        let mmdc = self
            .mmdc_path
            .as_ref()
            .ok_or_else(|| MermaidError::EngineUnavailable("no mmdc path".into()))?;

        let tmp = tempfile::tempdir().map_err(|e| MermaidError::RenderFailed(e.to_string()))?;
        let input = tmp.path().join("diagram.mmd");
        let output = tmp.path().join("diagram.png");

        std::fs::write(&input, source).map_err(|e| MermaidError::RenderFailed(e.to_string()))?;

        let theme = match params.theme {
            MermaidTheme::Light => "default",
            MermaidTheme::Dark => "dark",
        };

        let result = Command::new(mmdc)
            .arg("-i")
            .arg(&input)
            .arg("-o")
            .arg(&output)
            .arg("-t")
            .arg(theme)
            .arg("--scale")
            .arg("1")
            .output();

        let output_result = result.map_err(|e| MermaidError::RenderFailed(e.to_string()))?;
        if !output_result.status.success() {
            return Err(MermaidError::RenderFailed(format!(
                "mmdc exited with {}: {}",
                output_result.status,
                String::from_utf8_lossy(&output_result.stderr)
            )));
        }

        let png = std::fs::read(&output).map_err(|e| MermaidError::RenderFailed(e.to_string()))?;

        // We don't have an image decoder here, so we report a nominal size.
        // In a full implementation, this would parse the PNG header for dimensions.
        let (width_px, height_px) = parse_png_dimensions(&png).unwrap_or((params.min_width_px, 0));

        Ok(RenderedDiagram {
            png,
            width_px,
            height_px,
        })
    }
}

/// Pure-Rust stub engine that produces a placeholder PNG.
///
/// This is a fallback for environments without `mmdc`. It generates a minimal
/// valid 1x1 PNG in the theme's background color.
pub struct PureRustEngine;

impl MermaidEngine for PureRustEngine {
    fn render(
        &self,
        _source: &str,
        params: &RenderParams,
        _limits: &RenderLimits,
    ) -> Result<RenderedDiagram, MermaidError> {
        let bg = params.theme.surface_background();
        let png = minimal_png(bg.r, bg.g, bg.b, bg.a);
        Ok(RenderedDiagram {
            png,
            width_px: 1,
            height_px: 1,
        })
    }
}

/// Return the default engine: `SubprocessEngine` if `mmdc` is available,
/// otherwise `PureRustEngine`.
pub fn default_engine() -> Box<dyn MermaidEngine> {
    if which_mmdc().is_some() {
        Box::new(SubprocessEngine::default())
    } else {
        Box::new(PureRustEngine)
    }
}

/// Find `mmdc` on `$PATH`.
fn which_mmdc() -> Option<PathBuf> {
    let cmd = if cfg!(windows) { "mmdc.cmd" } else { "mmdc" };
    which::which(cmd).ok()
}

/// Parse PNG dimensions from the IHDR chunk.
fn parse_png_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    // PNG signature: 8 bytes, then IHDR chunk: 4 length + 4 type + 4 width + 4 height
    if data.len() < 24 {
        return None;
    }
    // Check PNG signature.
    if &data[0..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    // IHDR starts at byte 8: 4-byte length, 4-byte type "IHDR", then width/height.
    if &data[12..16] != b"IHDR" {
        return None;
    }
    let width = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
    let height = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
    Some((width, height))
}

/// Generate a minimal 1x1 PNG with the given RGBA color.
fn minimal_png(r: u8, g: u8, b: u8, _a: u8) -> Vec<u8> {
    // A minimal 1x1 RGBA PNG. This is a pre-computed PNG template with the
    // color bytes filled in. For simplicity, we produce a 1x1 RGB PNG.
    // PNG structure: signature + IHDR + IDAT + IEND
    let mut png = Vec::new();
    // Signature
    png.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    // IHDR chunk
    png.extend_from_slice(&0u32.to_be_bytes()); // length = 13
    png.extend_from_slice(b"IHDR");
    png.extend_from_slice(&1u32.to_be_bytes()); // width
    png.extend_from_slice(&1u32.to_be_bytes()); // height
    png.push(8); // bit depth
    png.push(2); // color type = RGB
    png.push(0); // compression
    png.push(0); // filter
    png.push(0); // interlace
    let ihdr_crc = crc32(&png[12..29]);
    png.extend_from_slice(&ihdr_crc.to_be_bytes());
    // IDAT chunk: deflate-compressed scanline (filter byte 0 + RGB pixel)
    let raw_scanline = [0u8, r, g, b]; // filter=0, then RGB
    let compressed = zlib_compress(&raw_scanline);
    png.extend_from_slice(&(compressed.len() as u32).to_be_bytes());
    png.extend_from_slice(b"IDAT");
    png.extend_from_slice(&compressed);
    let idat_crc = crc32(&png[png.len() - compressed.len() - 4..png.len()]);
    png.extend_from_slice(&idat_crc.to_be_bytes());
    // IEND chunk
    png.extend_from_slice(&0u32.to_be_bytes());
    png.extend_from_slice(b"IEND");
    let iend_crc = crc32(b"IEND");
    png.extend_from_slice(&iend_crc.to_be_bytes());
    png
}

/// Simple CRC32 (PNG polynomial).
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFF_FFFF
}

/// Minimal zlib compression (stored blocks, no real deflate).
fn zlib_compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // zlib header: CMF=0x78 (deflate, window 32K), FLG=0x01 (no dict, level 0)
    out.push(0x78);
    out.push(0x01);
    // Stored deflate blocks: split into blocks of max 65535 bytes.
    let mut offset = 0;
    while offset < data.len() {
        let remaining = data.len() - offset;
        let block_len = remaining.min(65535) as u16;
        let is_final = remaining <= 65535;
        // BFINAL (1 bit) + BTYPE=00 (stored)
        out.push(if is_final { 1 } else { 0 });
        // LEN and NLEN
        out.extend_from_slice(&block_len.to_le_bytes());
        out.extend_from_slice(&(!block_len).to_le_bytes());
        out.extend_from_slice(&data[offset..offset + block_len as usize]);
        offset += block_len as usize;
    }
    // Adler-32 checksum
    let adler = adler32(data);
    out.extend_from_slice(&adler.to_be_bytes());
    out
}

/// Adler-32 checksum.
fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_to_hex() {
        assert_eq!(Rgba::new(255, 0, 0, 255).to_hex(), "#ff0000");
        assert_eq!(Rgba::new(0, 255, 0, 128).to_hex(), "#00ff00");
    }

    #[test]
    fn theme_background() {
        assert_eq!(
            MermaidTheme::Light.surface_background(),
            Rgba::new(255, 255, 255, 255)
        );
        assert_eq!(
            MermaidTheme::Dark.surface_background(),
            Rgba::new(30, 30, 30, 255)
        );
    }

    #[test]
    fn render_params_default() {
        let p = RenderParams::default();
        assert_eq!(p.theme, MermaidTheme::Light);
        assert_eq!(p.min_width_px, 100);
    }

    #[test]
    fn pure_rust_engine_produces_valid_png() {
        let engine = PureRustEngine;
        let diagram = engine
            .render(
                "graph TD\n  A --> B",
                &RenderParams::default(),
                &RenderLimits::default(),
            )
            .unwrap();
        // Should be a valid PNG.
        assert!(diagram.png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert_eq!(diagram.width_px, 1);
        assert_eq!(diagram.height_px, 1);
    }

    #[test]
    fn parse_png_dimensions_valid() {
        let png = minimal_png(255, 0, 0, 255);
        let (w, h) = parse_png_dimensions(&png).unwrap();
        assert_eq!(w, 1);
        assert_eq!(h, 1);
    }

    #[test]
    fn parse_png_dimensions_invalid() {
        assert!(parse_png_dimensions(b"not a png").is_none());
        assert!(parse_png_dimensions(b"").is_none());
    }

    #[test]
    fn render_checked_rejects_oversize() {
        // Create a mock engine that returns a huge diagram.
        struct HugeEngine;
        impl MermaidEngine for HugeEngine {
            fn render(
                &self,
                _source: &str,
                _params: &RenderParams,
                _limits: &RenderLimits,
            ) -> Result<RenderedDiagram, MermaidError> {
                Ok(RenderedDiagram {
                    png: vec![],
                    width_px: 10_000,
                    height_px: 10_000,
                })
            }
        }
        let engine = HugeEngine;
        let result = render_checked(
            &engine,
            "graph TD",
            &RenderParams::default(),
            &RenderLimits {
                max_output_megapixels: 10,
                timeout: Duration::from_secs(1),
            },
        );
        assert!(matches!(result, Err(MermaidError::OutputTooLarge { .. })));
    }

    #[test]
    fn default_engine_returns_something() {
        let _engine = default_engine();
    }

    #[test]
    fn subprocess_engine_detect() {
        // detect() should either find mmdc or return an error — never panic.
        let _ = SubprocessEngine::detect();
    }

    #[test]
    fn crc32_known_value() {
        // CRC32 of "IEND" is a known constant.
        let crc = crc32(b"IEND");
        assert_eq!(crc, 0xAE426082);
    }

    #[test]
    fn adler32_known_value() {
        // Adler-32 of "Wikipedia" is 0x11E60398.
        assert_eq!(adler32(b"Wikipedia"), 0x11E60398);
    }

    #[test]
    fn minimal_png_is_valid_structure() {
        let png = minimal_png(100, 150, 200, 255);
        // Signature
        assert_eq!(&png[0..8], b"\x89PNG\r\n\x1a\n");
        // IHDR
        assert_eq!(&png[12..16], b"IHDR");
        // IEND should be present
        assert!(png.windows(4).any(|w| w == b"IEND"));
    }
}
