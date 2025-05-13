ARG REGISTRY

FROM ${REGISTRY}/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG https_proxy
ARG http_proxy
ARG USER_NAME
ARG USER_ID
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

# Add user to image
RUN groupadd -g "${GROUP_ID}" "${GROUP_NAME}" && \
    useradd -m -u "${USER_ID}" -l "${USER_NAME}" -G "${GROUP_NAME}"

COPY ./requirements.txt /requirements.txt 

RUN pip --no-cache-dir install --upgrade pip\
    && pip install --no-cache-dir --requirement /requirements.txt

# Copy source code
USER "${USER_NAME}":"${GROUP_NAME}"
COPY --chown="${USER_NAME}":"${GROUP_NAME}" ./* /gan/ 

ENTRYPOINT ["bash"]