//! One profile-fallback resolver shared by startup and switching.

use super::{Cli, Config, Settings, profile_for_route, resolve_named_profile};

/// The fallback chain (excluding the primary) — `--fallback` flags first, then
/// the selected profile's `fallback` list, deduped. Profiles that don't resolve
/// (missing key/model) are skipped with a warning rather than blocking startup.
pub fn resolve_fallbacks(cli: &Cli, config: &Config) -> Vec<Settings> {
    let primary_name = cli.profile.as_ref().or(config.default_profile.as_ref());
    let primary_route_profile = profile_for_route(
        primary_name.and_then(|name| config.profiles.get(name)),
        cli.provider,
    );

    let mut names: Vec<String> = cli.fallback.clone();
    if let Some(list) = primary_route_profile.and_then(|profile| profile.fallback.as_ref()) {
        names.extend(list.iter().cloned());
    }

    resolve_fallback_names(primary_name.map(String::as_str), names, config)
}

/// Same profile fallback resolution for startup and interactive switches.
pub fn resolve_profile_fallbacks(config: &Config, primary_name: &str) -> Vec<Settings> {
    let names = config
        .profiles
        .get(primary_name)
        .and_then(|profile| profile.fallback.clone())
        .unwrap_or_default();
    resolve_fallback_names(Some(primary_name), names, config)
}

fn resolve_fallback_names(
    primary_name: Option<&str>,
    names: Vec<String>,
    config: &Config,
) -> Vec<Settings> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Some(name) = primary_name {
        seen.insert(name.to_string()); // don't fall back to the primary itself
    }

    let mut out = Vec::new();
    for name in names {
        if !seen.insert(name.clone()) {
            continue;
        }
        match resolve_named_profile(config, &name) {
            Ok(settings) => out.push(settings),
            Err(err) => {
                eprintln!("\x1b[33mwarning: skipping fallback profile '{name}': {err}\x1b[0m")
            }
        }
    }
    out
}
