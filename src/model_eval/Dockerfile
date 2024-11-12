ARG REGISTRY

FROM ${REGISTRY}/pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

ARG https_proxy
ARG http_proxy
ARG SERVICE_ACC_NAME
ARG SERVICE_ACC_UID
ARG GROUP_NAME
ARG GROUP_ID
ARG SRC_PATH
ARG NEWUSERS_FILE_PATH
ARG MY_USER_ID
ARG MY_ACC_NAME

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

# Add all users from cvlab
# COPY "${NEWUSERS_FILE_PATH}" /new_users.txt
RUN groupadd -g "${GROUP_ID}" "${GROUP_NAME}" && \
    useradd -m -u "${MY_USER_ID}" -l "${MY_ACC_NAME}" -G "${GROUP_NAME}" && \
    useradd -m -u "${SERVICE_ACC_UID}" -l "${SERVICE_ACC_NAME}" -G "${GROUP_NAME}"

# Copy working dir to clone the git repo inside
COPY ./diffusers/* /diffusers/
COPY ./requirements_add.txt /requirements_add.txt
RUN rm /diffusers/README.md

# COPY ./segadapt/* /segadapt/ 
# RUN rm /segadapt/README.md

# Install python packages
RUN pip --no-cache-dir install --upgrade pip\
    && git clone https://github.com/huggingface/diffusers /diffusers/\
    && pip install --no-cache-dir -e /diffusers/.\
    && pip install --no-cache-dir -r /diffusers/examples/controlnet/requirements_sdxl.txt\
    && pip install --no-cache-dir -r /requirements_add.txt
    
# Copy source code
USER "${SERVICE_ACC_NAME}":"${GROUP_NAME}"
COPY --chown="${SERVICE_ACC_NAME}":"${GROUP_NAME}" ./* /sdxl/

ENTRYPOINT ["bash"]