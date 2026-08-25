#!/bin/sh
# docker-entrypoint.sh — picks Streamlit UI vs FastAPI inside the same image
# (see serving/Dockerfile). `docker run visual-search-app` (or CMD's
# default "app") starts Streamlit; `docker run visual-search-app api`
# starts FastAPI instead. docker-compose.yml uses both from one image.
set -e

if [ "$1" = "api" ]; then
    echo "[docker-entrypoint] Starting FastAPI (serving/api.py) on :8000"
    exec uvicorn serving.api:app --host 0.0.0.0 --port 8000
else
    echo "[docker-entrypoint] Starting Streamlit (serving/app.py) on :8501"
    exec streamlit run serving/app.py --server.port=8501 --server.address=0.0.0.0
fi
