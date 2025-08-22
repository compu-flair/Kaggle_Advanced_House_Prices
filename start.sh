# Stop any running container named 'streamlit_app'
docker stop streamlit_app 

# Build the Docker image with the tag 'streamlit_app' and run it, mapping port 8501
docker build -t streamlit_app . && docker run --rm -p 8501:8501 streamlit_app
