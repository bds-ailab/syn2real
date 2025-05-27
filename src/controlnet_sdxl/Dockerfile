FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

ARG GROUP_NAME
ARG GROUP_ID
ARG USER_ID
ARG USER_NAME

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
    useradd -m -u "${USER_ID}" -l "${USER_NAME}" -G "${GROUP_NAME}" &&

# Copy working dir to clone the git repo inside
COPY ./diffusers /diffusers
COPY ./requirements_add.txt /requirements_add.txt
RUN rm /diffusers/README.md

# Install python packages
RUN pip --no-cache-dir install --upgrade pip\
    && git clone https://github.com/huggingface/diffusers /diffusers/\
    && pip install --no-cache-dir -e /diffusers/.\
    && pip install --no-cache-dir -r /diffusers/examples/controlnet/requirements_sdxl.txt\
    && pip install --no-cache-dir -r /requirements_add.txt\
    && pip install git+https://github.com/facebookresearch/segment-anything.git\
    && pip install opencv-python pycocotools matplotlib onnxruntime onnx 

# Copy source code
USER "${USER_NAME}":"${GROUP_NAME}"
COPY --chown="${USER_NAME}":"${GROUP_NAME}" ./* /sdxl/

ENTRYPOINT ["bash"]