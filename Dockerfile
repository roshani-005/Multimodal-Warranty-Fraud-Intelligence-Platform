FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY claimgraph ./claimgraph
COPY cypher ./cypher
COPY tests ./tests
RUN mkdir -p artifacts data/generated
EXPOSE 8000
CMD ["sh", "-c", "python -m claimgraph.pipeline --rows 5000 && uvicorn claimgraph.api:app --host 0.0.0.0 --port 8000"]
