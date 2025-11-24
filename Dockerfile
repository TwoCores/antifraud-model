FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt
COPY . .

# Create a group and user
RUN addgroup --gid 1001 user && adduser --gid 1001 --uid 1001 user

USER 1001

CMD ["fastapi", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
