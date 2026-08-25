# serving/Dockerfile — component C8 (model serving), and the pragmatic
# stand-in for the diagram's "Docker Image -> Kubernetes" step: one image,
# runnable as either the Streamlit UI or the FastAPI service depending on
# the command docker-compose.yml (or you) pass at `docker run` time — a
# real k8s Deployment would just be N replicas of this same image behind a
# Service, which is more infrastructure than a laptop demo needs (see
# docs/diagram_mapping.md for that trade-off).
#
# Turns "works on my machine" into a portable, buildable artifact — the
# concrete fix for the fact that the original app.py assumed model/index
# files sitting in the same folder it was launched from.
#
# Build from the MLOPS/ root so this can COPY both serving/ and the
# artifacts it needs:
#   cd MLOPS
#   docker build -f serving/Dockerfile -t visual-search-app .
#
# Run the Streamlit UI (default):
#   docker run -p 8501:8501 \
#     -v "$(pwd)/artifacts:/app/artifacts" \
#     -v "$(pwd)/data/processed:/app/data/processed" \
#     -v "$(pwd)/mlflow.db:/app/mlflow.db" \
#     visual-search-app
#
# Run the FastAPI service instead (override the entrypoint command):
#   docker run -p 8000:8000 \
#     -v "$(pwd)/artifacts:/app/artifacts" \
#     -v "$(pwd)/data/processed:/app/data/processed" \
#     -v "$(pwd)/mlflow.db:/app/mlflow.db" \
#     visual-search-app api
#
# See docker-compose.yml at the repo root for both running together.
#
# NOTE: artifacts/ (models + indices, ~100s of MB), data/processed/
# (gallery images), and mlflow.db (the registry) are intentionally NOT
# baked into the image — mounted at runtime instead, so a new model
# version (or a freshly promoted registry entry) doesn't require a rebuild.

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY serving/requirements.txt ./serving-requirements.txt
RUN pip install --no-cache-dir -r serving-requirements.txt

COPY src/common.py ./src/common.py
COPY src/__init__.py ./src/__init__.py
COPY serving/app.py ./serving/app.py
COPY serving/api.py ./serving/api.py
COPY serving/search_core.py ./serving/search_core.py
COPY serving/__init__.py ./serving/__init__.py
COPY params.yaml ./params.yaml
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

EXPOSE 8501 8000
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || curl --fail http://localhost:8000/health || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["app"]
