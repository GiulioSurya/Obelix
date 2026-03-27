FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv sync --no-dev --extra litellm --extra serve --extra openshell --extra tracer --link-mode=copy

COPY examples/shell-demo/bash_server_sandboxed.py .

EXPOSE 8002
CMD ["uv", "run", "python", "bash_server_sandboxed.py"]
