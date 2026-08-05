use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn session_cache_hit(c: &mut Criterion) {
    let mut fixture = hi_tui::benchmark::SessionFixture::new(120, 24);
    c.bench_function("session_cache_hit_120x24", |b| {
        b.iter(|| {
            fixture.render_cache_hit();
            black_box(())
        })
    });
}

fn session_cache_rebuild(c: &mut Criterion) {
    let mut fixture = hi_tui::benchmark::SessionFixture::new(120, 24);
    c.bench_function("session_cache_rebuild_120x24", |b| {
        b.iter(|| {
            fixture.render_full_rebuild();
            black_box(())
        })
    });
}

fn session_stream_invalidation(c: &mut Criterion) {
    let mut fixture = hi_tui::benchmark::SessionFixture::new(40, 10);
    c.bench_function("session_stream_invalidation_40x10", |b| {
        b.iter(|| {
            fixture.render_rebuild();
            black_box(())
        })
    });
}

fn dashboard_frames(c: &mut Criterion) {
    let mut wide = hi_tui::benchmark::DashboardFixture::new(120, 24, 50);
    c.bench_function("dashboard_50_rows_120x24", |b| {
        b.iter(|| {
            wide.render();
            black_box(())
        })
    });

    let mut tiny = hi_tui::benchmark::DashboardFixture::new(24, 8, 50);
    c.bench_function("dashboard_50_rows_24x8", |b| {
        b.iter(|| {
            tiny.render();
            black_box(())
        })
    });
}

fn watch_frames(c: &mut Criterion) {
    let mut wide = hi_tui::benchmark::WatchFixture::new(120, 24, 50);
    c.bench_function("watch_50_rows_120x24", |b| {
        b.iter(|| {
            wide.render();
            black_box(())
        })
    });

    let mut tiny = hi_tui::benchmark::WatchFixture::new(24, 8, 50);
    c.bench_function("watch_50_rows_24x8", |b| {
        b.iter(|| {
            tiny.render();
            black_box(())
        })
    });
}

criterion_group!(
    tui_render,
    session_cache_hit,
    session_cache_rebuild,
    session_stream_invalidation,
    dashboard_frames,
    watch_frames
);
criterion_main!(tui_render);
