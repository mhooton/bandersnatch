FROM --platform=linux/amd64 python:3.12

# Set environment variables
ENV PYTHONPATH="/opt/bandersnatch/src:${PYTHONPATH:-}"

WORKDIR /usr/src/app

# Copy and install Python requirements
COPY docker/requirements.txt ./
RUN pip install numpy
RUN pip install --no-cache-dir -r requirements.txt

# Set up pipeline environment
ENV BANDERSNATCH_PATH="/opt/bandersnatch/src"
WORKDIR /opt/bandersnatch

# Copy pipeline code
COPY src/ ./src/

# Create data directory
RUN mkdir -p /data

# Set working directory to pipeline source
WORKDIR /opt/bandersnatch/src

CMD ["/bin/bash"]