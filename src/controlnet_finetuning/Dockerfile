ARG REGISTRY

FROM ${REGISTRY}/pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

ARG https_proxy
ARG http_proxy
ARG SERVICE_ACC_NAME
ARG SERVICE_ACC_UID
ARG GROUP_NAME
ARG GROUP_ID
ARG SRC_PATH


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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# hadolint ignore=DL3046
RUN groupadd -g "${GROUP_ID}" "${GROUP_NAME}" \
    && echo "%${GROUP_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
    && useradd -m -u "${SERVICE_ACC_UID}" -l "${SERVICE_ACC_NAME}" -G "${GROUP_NAME}"

USER "${SERVICE_ACC_NAME}":"${GROUP_NAME}"

# Copy working dir to clone the git repo inside
COPY --chown="${SERVICE_ACC_NAME}":"${GROUP_NAME}" ./ControlNet /ControlNet

# Cloning ControlNet repository
RUN curl -L https://github.com/lllyasviel/ControlNet/archive/refs/heads/main.zip -o /ControlNet/controlnet.zip \
    && unzip -q /ControlNet/controlnet.zip -d /ControlNet/ \
    && mv /ControlNet/ControlNet-main /ControlNet/ControlNet \
    && rm /ControlNet/controlnet.zip
    
# Install python packages
COPY --chown="${SERVICE_ACC_NAME}":"${GROUP_NAME}" ./requirements_controlnet.txt /requirements_controlnet.txt
RUN pip install --user --upgrade pip && \
    pip install --user --default-timeout=100 --no-cache-dir -r /requirements_controlnet.txt

# COPY --chown="${SERVICE_ACC_NAME}":"${GROUP_NAME}" entrypoint.sh /entrypoint.sh