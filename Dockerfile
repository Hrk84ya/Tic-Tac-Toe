# Builder stage
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    # Security: Run as non-root user
    APP_USER=appuser \
    # Security: Set workdir permissions
    APP_HOME=/home/appuser/app

# Create non-root user and set up directory structure
RUN groupadd -r $APP_USER && \
    useradd -r -g $APP_USER -d /home/$APP_USER $APP_USER && \
    mkdir -p $APP_HOME && \
    chown -R $APP_USER:$APP_USER $APP_HOME

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder --chown=$APP_USER:$APP_USER /opt/venv /opt/venv

# Set working directory
WORKDIR $APP_HOME

# Copy application code
COPY --chown=$APP_USER:$APP_USER . .

# Switch to non-root user
USER $APP_USER

# Expose the port the app runs on
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/ || exit 1

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:9000", "--workers", "4", "--worker-class", "gthread", "--threads", "2", "app:app"]
