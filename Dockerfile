# LuminaQuant live-trading image.
#
# Docker is out of scope for the supported runtime profile (see docs/DEPLOYMENT.md,
# which documents a uv-only systemd install); this image mirrors that documented
# `uv sync --extra optimize --extra live` + `lq live` flow for parity/experiments.
#
#   Build:  docker build -t lumina-quant .
#   Run:    docker run --env-file .env lumina-quant
#
# The ta-lib dependency installs from a manylinux wheel that bundles the TA-Lib C
# library (ta_lib.libs/libta-lib-*.so), pinned + SHA256-checksummed via uv.lock
# (`--frozen`), so there is no plain-HTTP source tarball to download and compile.

ARG PYTHON_VERSION=3.14

FROM python:${PYTHON_VERSION}-slim

# Pinned uv toolchain, copied from the official image (matches the local 0.11.21).
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    LQ_DISABLE_CONSOLE_LOG=1

WORKDIR /app

# Resolve + install dependencies first for cache-friendly rebuilds.  --frozen
# fails closed if uv.lock is out of sync with pyproject.toml and installs the
# exact checksummed wheels (including the bundled-C-library ta-lib wheel).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra live --extra optimize

# Copy the source and install the project itself against the locked env.
COPY . .
RUN uv sync --frozen --extra live --extra optimize

# Structured logs / crash-recovery state live under these (mount to persist).
RUN mkdir -p logs data

# Live trader entrypoint (matches DEPLOYMENT.md and run_bot.sh).  Override the
# command in docker-compose or `docker run ... <cmd>` for other lq subcommands.
ENV PATH="/opt/venv/bin:${PATH}"
CMD ["lq", "live"]
