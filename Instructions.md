# 📚 Instructions to run: Advanced House Prices Prediction Project

Welcome to the Advanced House Prices Prediction project! This comprehensive guide will walk you through every step of building a complete machine learning solution from data preprocessing to deployment to showcasing your work and building your online presence, **needed to get hired in the industry**.

## 🎯 Learning Objectives [?????]

By completing this project, you will learn:

- 🧹 **Data Cleaning & Feature Engineering:** Transform raw data into model-ready features
- 🤖 **Machine Learning:** Build and train regression models for price prediction
- 🚀 **Deployment:** Create an interactive web application using Streamlit
- 📊 **Model Evaluation:** Understand and interpret performance metrics


## 🚀 Quick Start [?????]

Before diving into the full project, we recommend starting with our simplified Party-Time jupyter notebook in Google Colab. This **condensed version** introduces the **main concepts** and workflow without the complexity of the complete implementation. 
Once you're comfortable with the fundamentals, **return here for the comprehensive walkthrough**.

**📓 [Access Party-Time Notebook](https://drive.google.com/file/d/1JDunZdESjo7-qiNZHtjtWKs4oXk7qDG2/view?usp=sharing)** - *A beginner-friendly introduction to get you started*




## 🚀 Getting Started

Follow this comprehensive step-by-step workflow to complete the project. Each step includes both execution instructions and understanding of what you're accomplishing.

### Step 1: Environment Setup

Choose one of the following setup methods and follow the detailed instructions in [Setup Environment Guide](./Docs/3.Setup_Environment.md):[?????]

**Quick Setup:**

**Step 1: Fork the Repository**
1. Sign in to your GitHub account
2. Navigate to https://github.com/compu-flair/Kaggle_Advanced_House_Prices
3. Click the `Fork` button in the top-right corner
4. Click `Create fork` to make a copy of the project in your GitHub account

**Step 2: Clone Your Fork to Your Local Machine**
1. On your forked repository page, click the green `Code` button
2. Select the `SSH` tab (if you see "You don't have any public SSH keys," follow the [SSH Setup Guide](./Docs/2.Add_SSH_to_GitHub.md))
3. Copy the provided SSH URL 


```bash
# Clone the repository
git clone <url-to-your-forked-repo-from-steps-above>
cd Kaggle_Advanced_House_Prices

# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate kaggle-house-prices

# (Optional) Update the environment if you change dependencies
conda env update -f environment.yml --prune

# Add conda kernel to jupyter notebook
conda install ipykernel
python -m ipykernel install --user --name kaggle-house-prices --display-name "kaggle-house-prices"
```

**Additional Setup Steps:**
* **VSCode Python Interpreter Setup:**
   - **Windows/Linux:** Press `Ctrl+Shift+P` 
   - **Mac:** Press `Cmd+Shift+P` 
- Select "Python: Select Interpreter", then choose the "kaggle-house-prices" interpreter.
* Once you open the Jupyter Notebook, it should automatically use the "kaggle-house-prices" kernel. If not, please restart VSCode. And if not successful, then on the top right corner of the notebook, you can manually select the kernel by clicking on it and choosing "kaggle-house-prices". You most likely will find it in the Jupyter kernel list.


### Step 2: Kaggle Setup

Before you can download the dataset, you need to set up Kaggle credentials. Follow the complete guide in [Setup Kaggle](./Docs/4.Setup_Kaggle.md)[?????] which includes:

1. Creating your Kaggle account
2. Joining the competition and accepting rules
3. Setting up API credentials
4. Downloading the dataset


### Step 3: Initial Setup and Data Acquisition

**🎯 Project Status:** Getting Started (Phase 1 of 4)

1. **Download the Dataset**
   - Join the [Kaggle competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
   - Accept the competition rules
   - Download dataset:
     ```bash
     kaggle competitions download -c house-prices-advanced-regression-techniques -p data/
     ```
   
   **💡 What you're doing:** Accessing the Ames Housing dataset, one of the most popular datasets for regression problems. This dataset contains 79 explanatory variables describing residential homes in Ames, Iowa.

### Step 4: Data Exploration, Cleaning, and Model Building

1. **Open the Jupyter Notebook:**
In the left panel of the VSCode, click on the explorer tab, navigate to the notebook file, click and open it. 


2. **Go through the notebook, read, understand, and run cells one by one.**[?????]


### Step 5: Turn Your ML Model Into a Web Application

**💡 What you're doing:** Now that you have a trained machine learning model, you'll learn how to use it as an interactive web application using Streamlit. This step transforms your data science work into a user-friendly interface where anyone can input house characteristics and get price predictions in real-time using your trained model.

1. **Understand the Application Architecture**
   
   **📁 Key Files Explained:**
   
   - **`main.py`**: Streamlit application entry point
     - Orchestrates the entire web application
     - Handles navigation between different views
   
   - **`views/house_price.py`**: Main prediction interface
     - Loads the trained model
     - Creates input forms for house features
     - Makes predictions and displays results
   - **`views/custom_linear_app.py`**:
     - Accepts custom data.
     - Performs dynamic data analysis to each dataset.
     - Allows to train a linear regression model on the custom dataset.
   - **`views/custom_xgboost.py`**:
     - Accepts custom data.
     - Performs dynamic data analysis to each dataset.
     - Allows to train a custom XGBoost model on the custom dataset.
   - **`models/schemas.py`**: Data validation schemas (Pydantic)
     - Ensures input data has correct format
     - Validates feature types and ranges
     - Provides clear error messages for invalid inputs
   
   - **`configs/config.py`**: Application configuration
     - Model file paths and settings
     - Default values for features
     - App parameters and constants

2. **Launch the Streamlit Application**
   ```bash
   streamlit run main.py
   ```
   - Open your browser to `http://localhost:8501`

   **🎮 Explore These Features:**
   Use the **left panel** to select the following features
   - **House Price Prediction**: 
      - This page will use the model trained during `data_cleaning_and_feature_engineering.ipynb` to predict prices
   - **Custom Linear Regression**:
      - Upload a csv file of any dataset, for example `data/train.csv`
      - Data will be processed and a linear regression will be trained
      - Go down the page, select your target column (the y) from drop-down labeled `Select label (Y) column`
      - select as many columns as you wish to serve as your X in the linear regression in `Select feature columns (X)` dropdown
      - Click the `Start Training` button. This will train the linear regression.
      - Under `Make a Prediction`, choose some values for each of the features, and then click `Predict` to get a prediction.
   - **Custom XGBoost**:
      - Same as in `Custom Linear Regression` above with the difference that an XGBoost model will be used instead of linear regression. 
   - **About**: 
      - Here you need to explain what the application is doing and how to use it. (After you make your own changes.)

   **✅ Checkpoint:** After this step, you should have:
   - A working web application
   - Ability to make predictions through the UI
   - Understanding of one simple way of how an ML model can be turned into a web application. (There are more advanced methods that go beyond the scope of this project.)

### 🚀 One-Click Deployment & Showcase

**💡 Ready to share your work with the world?** Your Streamlit application includes a built-in deployment feature that makes it easy to showcase your project online:

1. **Deploy to Streamlit Cloud:**
   - In your running application, look for the `Deploy` button in the top-right corner and click it.
   - Choose the `Streamlit Community Cloud` and click the `Deploy now`
   - Sign in with your GitHub account when prompted
   - Select your repository and branch
      - Use the green `Code` button and copy the HTML url
   - Click the `Deploy` button
   - Your app will be automatically deployed and accessible via a public URL
   - Save the public URL to be used when showcasing your work
      - Here's an example pointing to Ardavan's: [Live Demo](https://kaggleadvancedhousepricesgit-cnmxptag4fayriochntgry.streamlit.app) (If it's inactive please wake it up).

2. **Share Your Live Demo:**
   - Copy the deployment URL and add it to your GitHub repository README
   - Share the link on LinkedIn, Twitter, or your portfolio
   - Include it in job applications as a live demonstration of your skills

3. **Benefits of Live Deployment:**
   - **Professional Portfolio**: Demonstrate real, working applications to potential employers
   - **Easy Sharing**: Anyone can test your model without installing anything
   - **Automatic Updates**: Your deployment updates automatically when you push changes to your repo in GitHub


## 🐳 Alternative: Docker Deployment (Production-Ready)

If you want a production-ready, portable, and reproducible deployment (the preferred method for most professionals):
  
**Why Docker?**
> Most professionals deploy with Docker because it guarantees that your application will run the same way everywhere—on your laptop, a server, or the cloud. Docker containers package your code, dependencies, and environment together, eliminating "it works on my machine" problems and making scaling, testing, and collaboration much easier.

1. **Using Docker Commands**
   ```bash
   docker build -t house-prices-app .
   docker run --name house_price_container -d --rm -p 8501:8501 house-prices-app
   ```

2. **Use the following url to access the app:**
    - http://0.0.0.0:8501
   
    **To terminate the app:**
    - Go to the terminal window where the Docker container is running and press `Ctrl+C`.
    - Alternatively, in a new terminal, you can run:
       ```bash
       docker stop house_price_container
       ```



### 🚀 (Optional) Deploying Your Docker Image to a Server

Once you have tested your Docker image locally, you can deploy it to any server (cloud or on-premises) where Docker is installed. Here are the next steps:

1. **Push the Image to a Container Registry:**
   - Tag your image for your registry (e.g., Docker Hub, AWS ECR, Google GCR):
     ```bash
     docker tag house-prices-app <your-username>/house-prices-app:latest
     docker push <your-username>/house-prices-app:latest
     ```
   - Replace `<your-username>` with your Docker Hub or registry username or ip/address.

2. **Pull and Run on the Server:**
   - On your server, install Docker if not already installed.
   - Pull your image:
     ```bash
     docker pull <your-username>/house-prices-app:latest
     ```
   - Run the container:
     ```bash
     docker run --rm -p 8501:8501 <your-username>/house-prices-app:latest
     ```

3. **Access the App:**
   - Open a browser and go to `http://<server-ip>:8501` (replace `<server-ip>` with your server's public IP address).


## 🔄 Potential Changes

Consider adding these extensions:

1. **Advanced Feature Engineering**: Try creating polynomial features, interaction terms, or domain-specific features
2. **Ensemble Methods**: Train and test multiple models and choose the best for better predictions



### 🏆 Change the Project and Showcase Your Skills on GitHub

Follow these steps to make your own improvements to the project and demonstrate your learning:

1. **Create a New Branch for Your Work**
   - It's best practice to make changes on a new branch:
     ```bash
     git checkout -b my-feature-branch
     ```

2. **Make Your Changes**
   - Edit code, add features, improve documentation, or experiment with new models.
   - Commit your changes regularly:
     ```bash
     git add .
     git commit -m "Describe your change"
     ```
   Follow commit [conventions](https://www.conventionalcommits.org/en/v1.0.0/) in your change message.

3. **Push Your Changes to GitHub**
   - Push your branch to your fork:
     ```bash
     git push origin my-feature-branch
     ```

4. **Showcase Your Work**
   - **Update the [ReadMe file](README.md)** to describe the new features you added.
   - Add your deployed Streamlit app url to the ReadMe file.
   - Screen record the Streamlit app
      - walk the viewer through overall app
      - demonstrate your own updates
   - Turn the recorded video to gif image and add to your ReadMe file.
   - For a detailed guide on how to write a job-winning ReadMe, read [this file](./Docs/6.How_to_Prepare_ReadMe.md).[?????]


5. **(Optional) Create a Pull Request**
   - If you think your changes could help others, open a pull request to the original repo to contribute back. For a step-by-step guide, see [How to Make a Pull Request](./Docs/5.How_to_Make_a_Pull_Request.md) [?????].
   - **This will improve your GitHub presence, which is publicly trackable by future employers**

### Present Your Project to Potential Hiring Managers
   - Make a LinkedIn post about your project, and the value you added.
   - See a detailed guide on how to write a catchy LinkedIn post [here](./Docs/7.Present_project_on_LinkedIn.md). [????]

