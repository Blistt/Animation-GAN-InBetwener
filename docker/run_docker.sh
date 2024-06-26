#!/bin/bash

# Docker image name
IMAGE_NAME="frannarp/gan-inbetween"

# Docker container name
CONTAINER_NAME="my_container"

# Host directory (path to your local repository)
HOST_DIRECTORY="/home/farriaga/gan-interpolator/"

# Container directory (path where you want to mount your repository in the container)
CONTAINER_DIRECTORY="/app"

# Run the Docker container with a bind mount and GPU support
docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        --name $CONTAINER_NAME -v $HOST_DIRECTORY:$CONTAINER_DIRECTORY $IMAGE_NAME /bin/bash