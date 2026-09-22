# Run this in the same region as the GPU: the state is re-sent on every branch, so RTT dominates.
FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv venv /opt/venv && VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 so1
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY --chown=so1:so1 src ./src
COPY --chown=so1:so1 config.example.yaml ./config.example.yaml

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SO1_CONFIG=/app/config.yaml \
    SO1_HOST=0.0.0.0 \
    SO1_PORT=8080

USER so1
EXPOSE 8080
# The upstream probes run at startup, so readiness is what /health reports, not process liveness.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=8).status==200 else 1)"
CMD ["python", "-m", "so1"]
