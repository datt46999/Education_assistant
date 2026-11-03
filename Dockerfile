FROM --platform=linux/arm64 ghcr.io/astral-sh/uv:python3.13-bookworm-slim
 
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"
 
# 1) install dependencies first (cached unless pyproject/uv.lock change)
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project
 
# 2) copy the code (.dockerignore keeps .env and .venv out)
COPY . .
 
# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser
 
EXPOSE 8080
# app.py = the AgentCore entrypoint (rename here if your file is agentcore_app.py)
CMD ["python", "app.py"]
 