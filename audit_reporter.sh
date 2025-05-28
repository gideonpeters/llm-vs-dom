#!/usr/bin/env bash
set -euo pipefail

#### CONFIGURATION ####
# (Feel free to tweak these paths/names if yours differ)
PROJECT_ROOT="$(pwd)"
IMAGE_NAME="lighthouse-reporter"

# Notebook → script
NOTEBOOK="$PROJECT_ROOT/scripts/audits_reporter.ipynb"
SCRIPT="$PROJECT_ROOT/scripts/audits_reporter.py"

# Docker context
DOCKERFILE="$PROJECT_ROOT/Dockerfile"

# Mount points
HOST_DATA_DIR="$PROJECT_ROOT/dataset"
HOST_RESULTS_DIR="$PROJECT_ROOT/results"
CONTAINER_DATA_DIR="/app/dataset"
CONTAINER_RESULTS_DIR="/app/results"

echo "🚀  Starting Lighthouse pipeline…"

# 1️⃣ Convert notebook to script
if [ -f "$NOTEBOOK" ]; then
  echo "📝  Converting $NOTEBOOK → $SCRIPT"
  jupyter nbconvert --to script "$NOTEBOOK" \
    --output "$(basename "$SCRIPT" .py)" \
    --output-dir "$(dirname "$SCRIPT")"
else
  echo "ℹ️   No notebook found at $NOTEBOOK, skipping conversion"
fi

# 2️⃣ Build the Docker image
echo "🐳  Building Docker image: $IMAGE_NAME"
docker build -f "$DOCKERFILE" -t "$IMAGE_NAME" "$PROJECT_ROOT"

# 3️⃣ Run the container
echo "▶️   Running container (mounting dataset & results)"
docker run --rm \
  -v "$HOST_DATA_DIR":"$CONTAINER_DATA_DIR" \
  -v "$HOST_RESULTS_DIR":"$CONTAINER_RESULTS_DIR" \
  "$IMAGE_NAME"

echo "✅  All done! Reports should be in $HOST_RESULTS_DIR"
