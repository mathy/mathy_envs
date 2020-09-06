#!/usr/bin/env bash

set -e
. .env/bin/activate

echo "========================= mypy"
mypy mathy_envs
echo "========================= flake8"
flake8 mathy_envs tests
echo "========================= black"
black mathy_envs tests --check
echo "========================= pyright"
npx pyright mathy_envs tests