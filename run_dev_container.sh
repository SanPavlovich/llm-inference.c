#!/bin/bash
docker run -d -v C:/vscode_projects/complete/llm-inference:/root/llm-inference -it --rm --name pytorch --gpus all --privileged ubuntu-torch:latest bash