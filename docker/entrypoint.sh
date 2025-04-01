#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Activate virtual environment if needed (uncomment if using venv)
# source /app/venv/bin/activate

# Run database migrations if applicable (for future enhancements)
# alembic upgrade head

# Start the FastAPI application with Uvicorn
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
