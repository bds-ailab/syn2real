ARG REGISTRY

FROM ${REGISTRY}/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

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
RUN groupadd -g "${GROUP_ID}" "${GROUP_NAME}" && \
    useradd -m -u "${SERVICE_ACC_UID}" -l "${SERVICE_ACC_NAME}" -G "${GROUP_NAME}" && \
    useradd -m -u "${MY_USER_ID}" -l "${MY_ACC_NAME}" -G "${GROUP_NAME}"

COPY ./requirements.txt /requirements.txt 

RUN pip --no-cache-dir install --upgrade pip\
    && pip install --no-cache-dir --requirement /requirements.txt

# Copy source code
USER "${SERVICE_ACC_NAME}":"${GROUP_NAME}"
COPY --chown="${SERVICE_ACC_NAME}":"${GROUP_NAME}" ./* /gan/ 

ENTRYPOINT ["bash"]