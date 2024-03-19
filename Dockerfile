# Use the official PyTorch CUDA image as the base image
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Update the system and install pip
RUN apt-get update && apt-get install -y python3-pip \
    apt-get update && apt-get install -y libgl1-mesa-glx \
    apt-get update \
    apt-get install -y libtiff5


# Run the script to install packages
RUN bash install_packages.sh && pip list