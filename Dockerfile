FROM python:3.11-slim

ARG INSTALL_EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./

RUN if [ -n "$INSTALL_EXTRAS" ]; then \
        uv sync --frozen --no-dev --extra "$INSTALL_EXTRAS"; \
    else \
        uv sync --frozen --no-dev; \
    fi

COPY app_ui ./app_ui
COPY rerun_viz ./rerun_viz
COPY scripts ./scripts
COPY configs ./configs

CMD ["uv", "run", "python", "scripts/serve/serve_dashboard_app.py", "--app-port", "8080", "--viewer-port", "9090", "--grpc-port", "9876", "--config-dir", "/app/configs", "--outputs-dir", "/outputs"]
