#!/bin/bash

mkdir -p "/root/.config/NVIDIA Corporation"
touch "/root/.config/NVIDIA Corporation/nsys-config.ini"
echo "CuptiUseRawGpuTimestamps=false" > "/root/.config/NVIDIA Corporation/nsys-config.ini"
# check:
cat "/root/.config/NVIDIA Corporation/nsys-config.ini"
