#!/bin/bash
# Setup script for Connect Four AI project

echo "Setting up Connect Four AI environment..."

# Add Poetry to PATH if not already there
export PATH="$HOME/.local/bin:$PATH"

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "✅ Poetry found: $(poetry --version)"
fi

# Install dependencies
echo "📦 Installing dependencies..."
poetry install

# Run tests to verify everything works
echo "🧪 Running tests..."
poetry run pytest tests/ -v

echo "✅ Setup complete! You can now use:"
echo "  poetry run python example.py"
echo "  poetry run connect-four-ai"
echo "  poetry run pytest"
