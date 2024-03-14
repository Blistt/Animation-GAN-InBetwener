FROM continuumio/miniconda3

WORKDIR /gan-interpolator

# Add the current directory contents into the container at /app
ADD . /gan-interpolator

# Install any needed packages specified in conda-requirements.txt
RUN conda install --file conda-requirements.txt

# Install any needed packages specified in pip-requirements.txt
RUN pip install --requirement pip-requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80
