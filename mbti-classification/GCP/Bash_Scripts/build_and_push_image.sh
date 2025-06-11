#!/bin/bash
set -e
set -o pipefail

echo "Creating Artifact Registry repository (if it doesn't exist)..."
gcloud artifacts repositories create "$REPO_NAME" \
  --repository-format=docker \
  --location="$REGION" \
  --description="MBTI Docker images" || echo "Repository may already exist, skipping..."

# Authenticating Docker with Artifact Registry is only necessary once
# echo "Authenticating Docker with Artifact Registry"
# gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

# build the image for gcp platform and use the file named Dockerfile in the docker dir automatically
# do it quietly without logging every step for readability
echo "Building Docker image: $IMAGE_URI..."
docker build --quiet --platform=linux/amd64 -t "$IMAGE_URI" "$PARENT_DIR/docker"

for i in {1..3}; do
  echo "Pushing Docker image to Artefact Registry (attempt $i)..."
  docker push "$IMAGE_URI" && break || echo "Push failed on attempt $i"
done

# to inspect the size of the layers run this line in a terminal
#docker history --no-trunc europe-west3-docker.pkg.dev/mbti-444713/mbti-repo/mbti-training:latest

