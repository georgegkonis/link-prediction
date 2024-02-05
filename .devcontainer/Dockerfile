# Use the official Miniconda3 image from the Docker Hub
FROM continuumio/miniconda3

# Install LaTeX
RUN apt-get update && apt-get install -y texlive-full

# Set the working directory
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Create the environment from the environment.yml file
COPY environment.yml /workspace/environment.yml
RUN conda env create -f /workspace/environment.yml

# Make sure the environment is activated
RUN echo "source activate link-prediction" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Install any needed packages specified in requirements.txt
RUN /opt/conda/envs/myenv/bin/pip install --upgrade pip
RUN /opt/conda/envs/myenv/bin/pip install -r requirements.txt

# Make port 8888 available to the world outside this container (for Jupyter Notebook)
EXPOSE 8888

# Default command to start a bash shell. Override in docker-compose if needed.
CMD ["bash"]
