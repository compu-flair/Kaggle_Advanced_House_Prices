docker stop streamlit_app 
docker build -t streamlit_app . && docker run --rm -p 8501:8501 streamlit_app
