#!/bin/bash

# Exit on error
set -e

# Default to development mode
MODE="development"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --prod|--production)
      MODE="production"
      shift
      ;;
    --dev|--development)
      MODE="development"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Starting Tic-Tac-Toe in $MODE mode..."

if [ "$MODE" = "production" ]; then
  # Run in production mode using gunicorn
  echo "Starting production server..."
  gunicorn --bind 0.0.0.0:9000 --workers 4 app:app
else
  # Run in development mode with auto-reload
  echo "Starting development server..."
  python train_model.py
  python -m flask run --host=0.0.0.0 --port=9000 --debug
fi
