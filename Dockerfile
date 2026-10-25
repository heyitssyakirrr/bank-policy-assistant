# One image, three services (see docker-compose.yml): the index bootstrap, the API and the UI.
# PDF parsing (Docling) is offline-only and not installed here: the image indexes the
# committed chunks.jsonl.gz with the committed embedding cache, so it never calls Gemini.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY alembic.ini .
COPY migrations ./migrations
COPY app ./app
COPY ui ./ui
COPY evals ./evals
COPY .streamlit/config.toml ./.streamlit/config.toml

# Not root: the index job writes only the unpacked data under /srv/app/data.
RUN useradd --create-home app && chown -R app /srv
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
