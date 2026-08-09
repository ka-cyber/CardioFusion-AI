# CardioFusion-AI -- reproducible environment.
#
# NOTE: this Dockerfile was written and syntax-reviewed in an environment
# with no internet access, so `docker build` has NOT been run against it
# here. Build it yourself and report back if anything doesn't line up --
# the base image, package names, and paths are all standard, but "written
# carefully" and "verified by actually building" are different claims and
# this repo is honest about which one applies. See PRODUCTION_READINESS.md.

FROM python:3.11-slim AS base

# System deps needed by scipy/matplotlib wheels and by wfdb's use of libsndfile-ish tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user -- don't run as root in a container that might process
# uploaded signal data
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Default: run the test suite (fast feedback that the image is sound).
# Override with `docker run <image> python -m training.train --config ...`
# or `docker run -p 8000:8000 <image> uvicorn api.main:app --host 0.0.0.0`
CMD ["pytest", "tests/", "-v"]
