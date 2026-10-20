FROM python:3.12-slim AS base
WORKDIR /app
ENV SEARCH_DB_PATH=/data/index.db \
    PYTHONUNBUFFERED=1
COPY . .
EXPOSE 8000
CMD ["uvicorn", "searchapp.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
