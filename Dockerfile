FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

COPY src ./src

RUN uv sync --frozen

COPY app.py ./

EXPOSE 8501

CMD ["uv", "run", "python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]