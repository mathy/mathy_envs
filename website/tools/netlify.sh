#!/bin/bash
set -e
sh tools/setup.sh
echo "Building Netlify site..."
../../.env/bin/mkdocs build
