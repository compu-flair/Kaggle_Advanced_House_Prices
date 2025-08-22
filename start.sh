# Build the Docker image with the tag 'streamlit_app'
docker build -t streamlit_app .

# Stop any running container named 'streamlit_app' (if exists)
docker stop streamlit_app 

# Run the Docker container, remove it after exit, and map port 8501
docker run --rm -p 8501:8501 streamlit_app
