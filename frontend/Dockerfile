FROM python:3.12-slim
EXPOSE 8084
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy application code
COPY . ./

# Use the Python executable to run Streamlit
ENTRYPOINT [ "python", "-m", "streamlit", "run", "app.py", "--server.port=8084", "--server.address=0.0.0.0" ]