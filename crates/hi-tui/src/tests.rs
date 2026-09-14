use super::*;
use ratatui::backend::TestBackend;

mod review_pane;
mod thinking;

pub(crate) fn dump(term: &Terminal<TestBackend>) -> String {
    let buf = term.backend().buffer();
    let mut out = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            out.push_str(buf[(x, y)].symbol());
        }
        out.push('\n');
    }
    out
}

pub(crate) fn test_app(provider: &str, model: &str) -> App {
    let mut app = App::new(provider, model);
    app.workspace_root = std::path::PathBuf::from("/workspace");
    app.timestamps_enabled = false;
    app
}
