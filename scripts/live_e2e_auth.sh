# Shared credential check for live e2e scripts.
#
# A run uses the same route as a normal `hi` launch: the saved default
# profile and its key in the private credential store. `HI_PROFILE` or
# `HI_PROVIDER` overrides that. An exported `PIPENETWORK_API_KEY` or
# `HI_API_KEY` still wins, matching hi's own resolution.
#
# Source this after `cd` to the repo root. Nothing here prints a key.

# 0 when hi can authenticate with the saved config or an exported key.
has_saved_hi_credential() {
  python3 - <<'PY'
import json, os
from pathlib import Path

def nonempty(name):
    return bool(os.environ.get(name, "").strip())

if nonempty("PIPENETWORK_API_KEY") or nonempty("HI_API_KEY"):
    raise SystemExit(0)

try:
    import tomllib
except ImportError:
    raise SystemExit(1)

base = Path(os.environ["XDG_CONFIG_HOME"]) if os.environ.get("XDG_CONFIG_HOME") else Path.home() / ".config"
config_path = base / "hi" / "config.toml"
auth_path = base / "hi" / "auth.json"
if not config_path.is_file():
    raise SystemExit(1)
try:
    config = tomllib.loads(config_path.read_text())
except Exception:
    raise SystemExit(1)

auth = {}
if auth_path.is_file():
    try:
        loaded = json.loads(auth_path.read_text())
        if isinstance(loaded, dict):
            auth = loaded
    except Exception:
        auth = {}

def store_has(reference):
    if not isinstance(reference, str) or not reference.startswith("auth-store://"):
        return False
    entry = auth.get(reference[len("auth-store://"):])
    return isinstance(entry, dict) and str(entry.get("access", "")).strip() != ""

def profile_ready(profile):
    if not isinstance(profile, dict):
        return False
    if str(profile.get("api_key") or "").strip():
        return True
    env_name = str(profile.get("api_key_env") or "").strip()
    if env_name and nonempty(env_name):
        return True
    ref = profile.get("api_key_ref")
    if store_has(ref):
        return True
    if isinstance(ref, str) and ref.startswith("env://"):
        return nonempty(ref[len("env://"):])
    return False

profiles = config.get("profiles") or {}
if not isinstance(profiles, dict):
    profiles = {}
chosen = os.environ.get("HI_PROFILE", "").strip()
if chosen:
    profile = profiles.get(chosen)
else:
    default = config.get("default_profile")
    profile = profiles.get(default) if isinstance(default, str) else None

if profile_ready(profile):
    raise SystemExit(0)

# A profile with no key of its own still works when `hi login <provider>`
# stored one under the provider id.
provider = os.environ.get("HI_PROVIDER", "").strip()
if not provider and isinstance(profile, dict):
    provider = str(profile.get("provider") or "openai")
if provider and store_has("auth-store://" + provider):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

# Prepend the optional route override. With neither set, hi uses the saved
# default profile, including its model and sealed key.
apply_live_route() {
  if [[ -n "${HI_PROFILE:-}" ]]; then
    HI_ARGS=(--profile "$HI_PROFILE" "${HI_ARGS[@]}")
  elif [[ -n "${HI_PROVIDER:-}" ]]; then
    HI_ARGS=(--provider "$HI_PROVIDER" "${HI_ARGS[@]}")
  fi
}

live_route_label() {
  if [[ -n "${HI_PROFILE:-}" ]]; then
    printf 'profile %s' "$HI_PROFILE"
  elif [[ -n "${HI_PROVIDER:-}" ]]; then
    printf 'provider %s' "$HI_PROVIDER"
  else
    printf 'saved default profile'
  fi
}

live_credential_label() {
  if [[ -n "${PIPENETWORK_API_KEY:-${HI_API_KEY:-}}" ]]; then
    printf 'env'
  else
    printf 'saved hi config'
  fi
}
