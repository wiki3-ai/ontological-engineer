#!/bin/bash
# =============================================================================
# Start MLflow Server for Local Development
# =============================================================================
# Usage: ./scripts/start_mlflow.sh
#
# This starts MLflow with:
#   - SQLite backend (persists runs, experiments, traces)
#   - Local artifact storage in ./mlflow-artifacts
#   - Accessible at http://localhost:5000
#
# For VS Code devcontainer: port 5000 should auto-forward
# =============================================================================

set -e

cd /workspaces/wiki3-kg-project

# Create directories if needed
mkdir -p mlflow-artifacts

echo "Starting MLflow server..."
echo "  Backend:   sqlite:///mlflow.sqlite"
echo "  Artifacts: ./mlflow-artifacts"
echo "  URL:       http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

mlflow server \
    --backend-store-uri sqlite:///mlflow.sqlite \
    --default-artifact-root ./mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
