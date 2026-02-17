# Use miniconda as base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "bandersnatch", "/bin/bash", "-c"]

# Copy source code
COPY ../src ./src/

# Set the working directory to where run.py is located
WORKDIR /app/src

# Activate environment and set as default for container
ENV CONDA_DEFAULT_ENV=bandersnatch
ENV PATH=/opt/conda/envs/bandersnatch/bin:$PATH
ENV RUNNING_IN_DOCKER=true

# Default command - can be overridden at runtime
ENTRYPOINT ["python", "run.py"]