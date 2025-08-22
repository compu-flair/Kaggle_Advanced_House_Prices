# Login to server 

SSH to your user and follow the rest of instructions here. 


# 🐍 Create Virtual Environment

```sh
python3 -m venv venv && source venv/bin/activate
```

# 📥 Install gdown Command Line

`gdown` allows us to download Google Drive files using their Drive ID.

```sh
pip install gdown
```

# 💻 Retrieve Our Code

To retrieve our code, run the following command:

```sh
gdown <Code Drive ID>
```

This will download a zipped file called `Compu-Img-Class.zip`.

# 📦 Extracting Zipped File

To extract the zip file data into the folder `Compu-Img-Class`:

```sh
unzip <Zipfile_name.zip> -d <Directory_name>
```

# 🚀 Starting Our Server

The following command will start a bash script that runs a series of commands to start our Docker service:

```sh
cd <Directory_name> && bash start.sh
```
The explaination for these commands are below.

# 📜 Scripts & Docker Explanations

This section explains how the provided scripts and Docker setup work together to deploy your FastAPI and Streamlit applications.

## 🏁 start.sh

The `start.sh` script automates the Docker workflow:

```sh
# Stop any running container named 'streamlit_app'
docker stop streamlit_app 

# Build the Docker image with the tag 'streamlit_app' and run it, mapping port 8501
docker build -t streamlit_app . && docker run --rm -p 8501:8501 streamlit_app
```

- 🏗️ **Build** — Creates the Docker image from the `Dockerfile`.
- 🛑 **Stop & Remove** — Cleans up any existing container.
- 🚦 **Run** — Starts a new container, maps ports, and ensures auto-restart.

## 🐳 Dockerfile

The Dockerfile packages the application and its dependencies into a portable container:

```dockerfile
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
```

### 🐳 Dockerfile Step-by-Step Explanation

- 🐍 **FROM python:3.11**  
    Uses the official Python 3.11 image as the base for the container, ensuring a consistent Python environment.

- 📁 **WORKDIR /app**  
    Sets `/app` as the working directory inside the container. All subsequent commands will run from this directory.

- 📦 **COPY requirements.txt .**  
    Copies the `requirements.txt` file from your project into the container, so dependencies can be installed.

- ⚡ **RUN pip install --no-cache-dir -r requirements.txt**  
    Installs all Python dependencies listed in `requirements.txt` without caching, reducing image size.

- 📂 **COPY . .**  
    Copies all files from your project directory into the container, making your code and assets available.

- 🚀 **CMD ["streamlit", "run", "main.py","--server.address=0.0.0.0","--server.port=8501"]**  
    Sets the default command to run the Streamlit app (`main.py`), listening on all network interfaces at port 8501, so it’s accessible externally.

---

# 🌐 Accessing the Apps in Your Browser

Once the server is running, open your browser and go to:
```http://<server-ip>:8501```
Replace `<server-ip>` with your server's actual IP address or domain name.
