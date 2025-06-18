FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

ARG USER_NAME
ARG USER_ID
ARG GROUP_NAME
ARG GROUP_ID

# ENV DEBIAN_FRONTEND=noninteractive

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends --fix-missing \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    sudo \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# hadolint ignore=DL3046
RUN groupadd -g "${GROUP_ID}" "${GROUP_NAME}" \
    && echo "%${GROUP_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
    && useradd -m -u "${USER_ID}" -l "${USER_NAME}" -G "${GROUP_NAME}"

USER "${USER_NAME}":"${GROUP_NAME}"

# Install python packages
COPY --chown="${USER_NAME}":"${GROUP_NAME}" ./pyproject_blip.toml /pyproject_blip.toml
# hadolint ignore=DL3013,SC1091
RUN pip install --user --upgrade pip && \
    pip install --user --default-timeout=100 --no-cache-dir -r /pyproject_blip.toml

# COPY --chown="${USER_NAME}":"${GROUP_NAME}" entrypoint.sh /entrypoint.sh

# Write your docker commands here