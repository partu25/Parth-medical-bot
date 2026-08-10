# 1. Lightweight Python image
FROM python:3.12-slim

# 2. Working directory
WORKDIR /app

# 3. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy app files
COPY . .

# 5. Render dynamic port binding
ENV PORT=10000
EXPOSE 10000

# 6. Streamlit command with dynamic port
CMD ["sh", "-c", "streamlit run web_app.py --server.port=${PORT:-10000} --server.address=0.0.0.0"]