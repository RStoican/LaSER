#!/bin/bash
set -e

# If the command passed to the container starts with a hyphen (e.g., -t, --task), we assume the user wants to run
# main.py with those arguments
if [[ "${1#-}" != "$1" ]]; then
    set -- python main.py "$@"
fi

exec "$@"