FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn==0.28.0 \
    pydantic==2.6.4 \
    qdrant-client==1.8.2 \
    prometheus-client==0.20.0 \
    httpx==0.27.0

COPY src/ /app/src/

WORKDIR /app/src
EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
