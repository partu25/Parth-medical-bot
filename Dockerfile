# 1. Use an official, lightweight Python image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy just the requirements file first (makes building faster)
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your app's code into the container
COPY . .

# 6. Tell Docker which port Streamlit uses
EXPOSE 8501

# 7. Build the database, then run the app
CMD ["sh", "-c", "streamlit run web_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]