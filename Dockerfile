ARG REGISTRY

FROM ${REGISTRY}/ubuntu:22.04

ARG https_proxy
ARG http_proxy
ARG SERVICE_ACC_NAME
ARG SERVICE_ACC_UID
ARG GROUP_NAME
ARG GROUP_ID

ENV DEBIAN_FRONTEND=noninteractive

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends --fix-missing \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# hadolint ignore=DL3046
RUN groupadd -g "${GROUP_ID}" "${GROUP_NAME}" \
    && useradd -m -u "${SERVICE_ACC_UID}" "${SERVICE_ACC_NAME}" -G "${GROUP_NAME}"

USER "${SERVICE_ACC_NAME}"

COPY entrypoint.sh /entrypoint.sh

# Write your docker commands here