#!/usr/bin/env sh
# Runs k3d for this project. When the host Docker CLI can reach the daemon, the local k3d is
# used. Otherwise (Rancher Desktop on Windows with a broken Docker pipe) the official Linux k3d
# binary runs inside the Rancher Desktop VM, next to its Docker daemon: downloaded once,
# checksum-verified, cached outside the repo, and nothing is installed into the VM.
set -eu

K3D_VERSION=v5.9.0
repo_dir=$(cd "$(dirname "$0")/../.." && pwd)

if docker info >/dev/null 2>&1 && command -v k3d >/dev/null 2>&1; then
    cd "$repo_dir" && exec k3d "$@"
fi
command -v rdctl >/dev/null 2>&1 || { echo "Docker unreachable and rdctl not found" >&2; exit 1; }

# Host path (Git Bash) -> the same path as mounted in the VM: /c/Users/... -> /mnt/c/Users/...
to_vm() { printf '%s' "$1" | sed -E 's#^/?([A-Za-z]):?/#/mnt/\L\1/#'; }

cache_dir="$(cygpath -u "${LOCALAPPDATA:-$HOME/.cache}" 2>/dev/null || echo "$HOME/.cache")/telcoai/bin"
bin="$cache_dir/k3d-$K3D_VERSION-linux-amd64"
if [ ! -f "$bin" ]; then
    mkdir -p "$cache_dir"
    base="https://github.com/k3d-io/k3d/releases/download/$K3D_VERSION"
    curl -fsSL -o "$bin.tmp" "$base/k3d-linux-amd64"
    expected=$(curl -fsSL "$base/checksums.txt" | awk '$2 ~ /(^|\/)k3d-linux-amd64$/ {print $1}')
    actual=$(sha256sum "$bin.tmp" | awk '{print $1}')
    if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then
        rm -f "$bin.tmp"
        echo "k3d checksum mismatch (expected '$expected', got '$actual')" >&2
        exit 1
    fi
    mv "$bin.tmp" "$bin"
fi

MSYS_NO_PATHCONV=1 exec rdctl shell sh -c 'cd "$1" && shift && exec "$@"' sh \
    "$(to_vm "$repo_dir")" "$(to_vm "$bin")" "$@"
