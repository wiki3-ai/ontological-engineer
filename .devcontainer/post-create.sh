#!/bin/bash
set -e

echo "Installing Deno..."
curl -fsSL https://deno.land/install.sh | sh

# Add Deno to PATH for current session
export DENO_INSTALL="$HOME/.deno"
export PATH="$DENO_INSTALL/bin:$PATH"

echo "Installing Deno Jupyter kernel..."
deno jupyter --install --unstable

# echo "Upgrading JupyterLab..."
# pip install --no-cache-dir "jupyterlab>=4.5"

echo "Post-create setup complete!"
