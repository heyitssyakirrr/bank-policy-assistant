"""Offline ingestion pipeline (parse -> structure -> chunk).

Never imported by the API at request time: it pulls in Docling and its
models, which the web app doesn't need. Run it with
`python -m app.ingestion.pipeline`.
"""
