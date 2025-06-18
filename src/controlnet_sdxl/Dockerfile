FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

ARG GROUP_NAME
ARG GROUP_ID
ARG USER_ID
ARG USER_NAME

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

# Add user to image
RUN groupadd -g "${GROUP_ID}" "${GROUP_NAME}" && \
    useradd -m -u "${USER_ID}" -l "${USER_NAME}" -G "${GROUP_NAME}"

# Copy working dir to clone the git repo inside
COPY ./diffusers /diffusers
COPY ./pyproject_controlnet_sdxl.toml /pyproject_controlnet_sdxl.toml
RUN rm /diffusers/README.md

# Install python packages
# hadolint ignore=DL3013,SC1091,DL3042
RUN pip install --no-cache-dir --user --upgrade pip\
    && git clone https://github.com/huggingface/diffusers /diffusers/\
    && pip install --no-cache-dir -e /diffusers/.\
    && pip install --no-cache-dir -r /diffusers/examples/controlnet/requirements_sdxl.txt\
    && pip install git+https://github.com/facebookresearch/segment-anything.git\
    && pip install opencv-python pycocotools matplotlib onnxruntime onnx 

# hadolint ignore=DL3013,SC1091
RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    pip install --no-cache-dir poetry && \
    poetry install

# Copy source code
USER "${USER_NAME}":"${GROUP_NAME}"
COPY --chown="${USER_NAME}":"${GROUP_NAME}" ./* /sdxl/

ENTRYPOINT ["bash"]