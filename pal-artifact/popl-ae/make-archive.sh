#!/usr/bin/env bash
# Pack the artifact for distribution: PAL, the artifact's scripts, the environment's Nix
# files, and the top-level files, as pal-artifact.tar.gz in the current directory.
#
#   popl-ae/make-archive.sh            (from the repository root)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$PWD/pal-artifact.tar.gz"
cd "$root"

files=(pal popl-ae examples unittests flake.nix flake.lock pyproject.toml sitecustomize.py README.md LICENSE
       Dockerfile .dockerignore
       # The Nix files flake.nix uses.
       nix/bngen-upstream.nix nix/dice-bayestensor nix/mlcuddidl.nix nix/rsdd
       nix/memo.nix nix/memo-python312-compat.patch nix/alea.jl.nix)
excludes=(
  --exclude=popl-ae/results
  --exclude='__pycache__' --exclude='*.pyc' --exclude=.pytest_cache
)
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='2026-10-01 00:00Z' \
    --transform 's,^,pal-artifact/,' "${excludes[@]}" -czf "$out" "${files[@]}"
tokens=$(tar -xzOf "$out" | grep -c 'gldt-[A-Za-z0-9]' || true)
if [ "$tokens" -gt 0 ]; then
  echo "error: the archive contains a deploy token" >&2
  exit 1
fi
echo "wrote $out ($(( $(stat -c %s "$out") / 1024 )) KB)"
