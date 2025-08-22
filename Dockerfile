# Use the official Python 3.11 image as the base
FROM python:3.11

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from the current directory into the container
COPY . .

# Run the Streamlit app, listening on all interfaces at port 8501
CMD ["streamlit", "run", "main.py","--server.address=0.0.0.0","--server.port=8501"]