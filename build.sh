#!/bin/bash
# Render.com build script
set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Installing GLPK solver ==="
apt-get update -qq && apt-get install -y -qq glpk-utils

echo "=== Build complete ==="
