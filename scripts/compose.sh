#!/usr/bin/env sh
# Runs `docker compose` for this project. When the host Docker CLI can't reach the daemon
# (e.g. Rancher Desktop's Windows pipe proxy is down), falls back to running compose from the
# docker:cli image inside the Rancher Desktop VM against its local socket.
set -eu

if docker info >/dev/null 2>&1; then
    exec docker compose "$@"
fi

command -v rdctl >/dev/null 2>&1 || { echo "Docker daemon unreachable and rdctl not found" >&2; exit 1; }

project_dir=$(cd "$(dirname "$0")/.." && pwd)
# /c/Users/... or C:/Users/... -> /mnt/c/Users/... (the Windows drive as mounted in the VM)
vm_dir=$(printf '%s' "$project_dir" | sed -E 's#^/?([A-Za-z]):?/#/mnt/\L\1/#')

# Stop Git Bash from rewriting the VM paths into Windows paths.
MSYS_NO_PATHCONV=1 exec rdctl shell docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$vm_dir:$vm_dir" -w "$vm_dir" \
    docker:29.8.1-cli docker compose "$@"
