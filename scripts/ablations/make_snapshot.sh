#!/usr/bin/env bash
# Read-only code snapshot of a commit (the live working tree may be edited by another session).
# Usage: make_snapshot.sh <dest_dir> [commit=HEAD]
set -euo pipefail
DEST="$1"; COMMIT="${2:-HEAD}"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
mkdir -p "$DEST"
git -C "$REPO" archive "$COMMIT" | tar -x -C "$DEST"
git -C "$REPO" rev-parse "$COMMIT" > "$DEST/COMMIT"
echo "snapshot of $(cat "$DEST/COMMIT") -> $DEST"
