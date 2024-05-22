#!/bin/bash
useradd -l -m -u "${USER_ID}" "${USER_NAME}" -G "${GROUP_NAME}",sudo || echo "Can't create user"
usermod "${USER_NAME}" -g "${GROUP_NAME}" || echo "Can't set default group"
tail -f /dev/null
