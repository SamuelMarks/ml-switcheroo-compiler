#!/usr/bin/env bash
# File: build_web_assets.sh
# Description: Builds the wheel distributions for ml-switcheroo-compiler and ml-switcheroo-ir
# and places them into the docs/_static directory for web deployment.

set -e

# Build the ml-switcheroo-compiler wheel
python3 -m build --wheel

# Copy compiler wheel to docs static
cp dist/*.whl docs/_static/

# Also fetch and build ml-switcheroo-ir
rm -rf tmp_ir_build || true
mkdir tmp_ir_build
cd tmp_ir_build
git clone https://github.com/SamuelMarks/ml-switcheroo-ir.git
cd ml-switcheroo-ir
python3 -m build --wheel
cp dist/*.whl ../../docs/_static/
cd ../../
rm -rf tmp_ir_build

echo "Web assets built and placed in docs/_static/"
