#!/usr/bin/env bash
# Install the pre-commit guard for this PUBLIC repo: the hygiene test must
# pass, and (when detect-secrets is installed) staged files must carry no
# secrets. Hooks are per-clone and untracked, so run this once after cloning.
set -euo pipefail
cd "$(dirname "$0")/.."
HOOK=.git/hooks/pre-commit
cat > "$HOOK" <<'HOOK_EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="${CLLOOP_PY:-python3}"
"$PY" -m pytest -q -p no:cacheprovider tests/test_public_hygiene.py \
  || { echo "pre-commit: public-hygiene guard FAILED — commit refused" >&2; exit 1; }
if command -v detect-secrets-hook >/dev/null 2>&1; then
  # *.dvc pointers are md5 content hashes by design — high-entropy, not secret
  git diff --cached --name-only --diff-filter=ACM -z \
    | xargs -0 -r detect-secrets-hook --exclude-files '\.dvc$' \
    || { echo "pre-commit: detect-secrets flagged a staged file" >&2; exit 1; }
fi
HOOK_EOF
chmod +x "$HOOK"
echo "installed $HOOK"
[ -f .hygiene-denylist.local ] || echo "note: no .hygiene-denylist.local — add one (untracked) for private terms"
