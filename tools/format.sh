#!/bin/sh -e
. .env/bin/activate

# Sort imports one per line, so autoflake can remove unused imports
isort mathy_envs tests --force-single-line-imports
autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place mathy_envs tests --exclude=__init__.py
isort mathy_envs tests
black mathy_envs tests