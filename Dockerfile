# Use a cuda enabled base image
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Set the working directory in the container to /gan-interpolator
WORKDIR /gan-interpolator

# Add the current directory contents into the container at /gan-interpolator
ADD . /gan-interpolator

# Set environment variables for tzdata
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

RUN apt-get update && apt-get install -y python3-opencv

# Run the script to install packages
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    torchvision \
    numpy \
    scipy \
    matplotlib \
    cupy \
    torchmetrics \
    kornia \
    Pillow \
    tqdm

# Make port 80 available to the world outside this container
EXPOSE 80