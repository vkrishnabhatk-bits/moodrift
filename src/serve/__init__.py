"""The serving plane: FastAPI app, request/response schemas, model runtime.

This package never imports from ``src.train`` or ``src.ingest``. The boundary between
training and serving is one model artifact plus one config file, and the moment serving
reaches back into the training code that boundary is gone.
"""
