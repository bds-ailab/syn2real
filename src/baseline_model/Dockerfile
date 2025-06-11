FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

ARG REPO_URL

LABEL org.opencontainers.image.source="$REPO_URL"

ARG USER_NAME
ARG USER_ID
ARG GROUP_NAME
ARG GROUP_ID

# ENV DEBIAN_FRONTEND=noninteractive

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends --fix-missing \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    unzip \
    sudo \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy src code
COPY ./* /baseline_model/

# hadolint ignore=DL3046
RUN groupadd -g "${GROUP_ID}" "${GROUP_NAME}" \
    && echo "%${GROUP_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
    && useradd -m -u "${USER_ID}" -l "${USER_NAME}" -G "${GROUP_NAME}"

USER "${USER_NAME}":"${GROUP_NAME}"

# Install python packages
COPY --chown="${USER_NAME}":"${GROUP_NAME}" pyproject_baseline.toml poetry.lock ./
RUN python3 -m venv .venv && \
. .venv/bin/activate && \
pip install --no-cache-dir poetry && \
poetry install