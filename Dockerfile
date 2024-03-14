# Use an official miniconda3 as parent image
FROM continuumio/miniconda3

# Set the working directory in the container to /gan-interpolator
WORKDIR /gan-interpolator

# Add the current directory contents into the container at /gan-interpolator
ADD . /gan-interpolator

# Run the script to install packages
RUN bash install_packages.sh

# Make port 80 available to the world outside this container
EXPOSE 80
