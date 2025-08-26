### Smoothen "Setup and Installation Instructions" in ReadMe

I also added conda

**AM:** [Resolved]

---

### Move Kaggle setup to docs 

Cut "🏆 Kaggle Setup" and instead point to Kaggle setup file in Docs/ (bring it in from the Project_Template)

**AM:** [Resolved]


---

### Faced this error 

(kaggle-house-prices) Kaggle_Advanced_House_Prices % kaggle competitions download -c house-prices-advanced-regression-techniques -p data/
zsh: command not found: kaggle

**AM:** [Resolved]

---


### Accept competition

Add to instructions or readme

**AM:** [Resolved]

---

### file name does not match

Data_Cleaning_and_Feature_Engineering_Final_Version.ipynb
referred in the ReadMe is not available in the folder. Lower the letters to match?

**AM:** [Resolved]

---

### move kaggle install out of notebook

1. 
!pip install kaggle in Data_Cleaning_and_Feature_Engineering_Final_Version.ipynb needs to be removed and placed in the requirements

**AM:** It's needed for google collab. 


2. 
!kaggle competitions download -c house-prices-advanced-regression-techniques -p data
this is redundant? since we already downloaded the data

**AM:** It's needed for google collab. 

---

### make a version of the project that runs entirely in google colab (also add this to Project_template, i.e. we always need this)

**AM:** Added a section in the current notebook to uncomment if running within collab.
---

### modify the following

In 
"""
2️⃣ Train Model (if needed)
- Use scripts or notebook to train and save models in `models/` (default model provided).
"""

remove "if needed" and then instruct how to.

**AM:** [Resolved]

---

### package not found

% streamlit run main.py
zsh: command not found: streamlit

**AM:** I can't replicate this behavior, perhabs you activated the wrong environment between steps.

---

### streamlit issue

I see the following right on the first app page:
ModuleNotFoundError: No module named 'xgboost'

**AM:** I can't replicate this behavior, perhabs you activated the wrong environment between steps.

---

### add a suggestion section for their future work to improve 

add this to project template and then bring here

**AM:** [Resolved]
