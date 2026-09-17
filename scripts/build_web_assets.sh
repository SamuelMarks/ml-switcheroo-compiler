#!/usr/bin/env bash
# File: build_web_assets.sh
# Description: Builds wheel distributions for ml-switcheroo-compiler, ml-switcheroo-ir,
# ml-framework-snapshots, and zero-* ecosystem packages.
# Prefers local checkouts in parent directory if they exist; otherwise fetches from GitHub.
# Copies built wheels to docs/_static/, generates wheels.json manifest, and verifies SHA256 checksums.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

# Clean previous local dist and stale static wheels
rm -rf dist tmp_*_build
mkdir -p docs/_static
rm -f docs/_static/*.whl docs/_static/wheels.json docs/_static/checksums.sha256

echo "=== Building ml-switcheroo-compiler wheel (local) ==="
python3 -m build --wheel --no-isolation
cp dist/*.whl docs/_static/

build_wheel_for_repo() {
    local repo_name="$1"
    local git_org="${2:-SamuelMarks}"
    local git_url="https://github.com/${git_org}/${repo_name}.git"
    local local_path="../${repo_name}"

    echo "=== Processing dependency: ${repo_name} ==="
    if [ -d "${local_path}" ]; then
        echo "Found local checkout at ${local_path}. Building wheel from local source..."
        (cd "${local_path}" && rm -rf dist && python3 -m build --wheel --no-isolation)
        cp "${local_path}"/dist/*.whl docs/_static/
    else
        echo "Local checkout ${local_path} not found. Fetching from ${git_url}..."
        local tmp_dir="tmp_${repo_name}_build"
        rm -rf "${tmp_dir}" || true
        mkdir -p "${tmp_dir}"
        git clone --depth 1 "${git_url}" "${tmp_dir}/${repo_name}"
        (cd "${tmp_dir}/${repo_name}" && python3 -m build --wheel --no-isolation)
        cp "${tmp_dir}/${repo_name}"/dist/*.whl docs/_static/
        rm -rf "${tmp_dir}"
    fi
}

# Determine list of dependencies to process
if [ "$#" -gt 0 ]; then
    TARGET_REPOS=("$@")
else
    TARGET_REPOS=(
        "ml-switcheroo-ir"
        "ml-framework-snapshots"
        "zero-jax"
        "zero-pytorch"
        "zero-tensorflow"
        "zero-keras"
        "zero-mlx"
        "zero-optax"
        "zero-flax"
        "zero-chex"
        "zero-grain"
        "zero-orbax"
        "zero-pax"
        "zero-zoo"
    )
fi

for repo in "${TARGET_REPOS[@]}"; do
    build_wheel_for_repo "${repo}"
done

# Generate wheels.json manifest for runtime deeplinking
python3 -c "
import glob, json, os
static_dir = 'docs/_static'
whls = sorted([os.path.basename(p) for p in glob.glob(os.path.join(static_dir, '*.whl'))])
manifest_path = os.path.join(static_dir, 'wheels.json')
with open(manifest_path, 'w') as f:
    json.dump({'wheels': whls}, f, indent=2)
print(f'Generated {manifest_path} with {len(whls)} wheels:')
for w in whls:
    print(f'  - {w}')
"

# Generate and verify SHA256 checksums
(cd docs/_static && shasum -a 256 *.whl > checksums.sha256 && shasum -a 256 -c checksums.sha256)

echo "Web assets successfully built, placed in docs/_static/, and verified via SHA256."
