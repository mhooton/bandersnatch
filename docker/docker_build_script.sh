#!/bin/bash

# Build the bandersnatch Docker image
echo "Building bandersnatch Docker image..."

# Build from the directory containing the Dockerfile
docker build -t bandersnatch:latest .

echo "Build complete. Image tagged as 'bandersnatch:latest'"

# Show the built image
docker images bandersnatch