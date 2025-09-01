## 🚀 Setup Kaggle
Follow this once (10–15 min). After this you can start any competition fast.

### 📝 0.1 Create & Prep Your Account
1. Go to https://www.kaggle.com and Sign Up / Log In.
2. Complete profile (photo, short bio, location) – it increases trust when others view your notebooks.
3. (Optional but good) Enable 2FA under Settings > Account.

### 📜 0.2 Accept Competition Rules
Before you can download data you MUST open the competition page and click: Join Competition / I Understand & Accept. Do this for every new competition (even “Getting Started” ones) or the API will return 403 errors. **Make sure you understand what you're agreeing to as well.**

### 🔑 0.3 Generate an API Token

To interact with kaggle using API, we need our kaggle configuration file.
1. Click your profile avatar (top-right) → Settings.
2. Scroll to “API” section → Create New Token.
3. This downloads `kaggle.json` (contains username + key). Keep it private.

### ⚙️ 0.4 Install & Configure Kaggle API
You can work in Google Colab (recommended for beginners) or locally (VS Code). Both steps below.

#### 🟢 A. Google Colab
Run these cells at the **top of every new notebook** (only the upload once per session):

```python
# 1. Install (usually fast)
!pip install -q kaggle

# 2. Upload kaggle.json (downloaded earlier)
from google.colab import files
files.upload()  # choose kaggle.json

# 3. Move it into the hidden .kaggle folder with correct permissions
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 4. Quick test
!kaggle --version
!pip install <'replace with your packages versions'>
```

💡 Tip: To avoid uploading every time, you can store the token in a private Google Drive folder and copy it programmatically, but manual upload is fine for now.

#### 💻 B. Local (macOS / Linux)
```bash
pip install kaggle
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/ #Sometimes you should place it in ~/.config/kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle --version
```

**//TODO not sure, I never encountered that//** If you use VS Code Python virtual environments, install inside that env.

### 📂 0.5 Recommended Project Folder Structure (Create Early)
```
project_root/
   data/
      raw/        # untouched downloaded files
      processed/  # cleaned / engineered
   notebooks/
      01_eda.ipynb
      02_model_baseline.ipynb
   models/
   submissions/
   reports/ **//TODO???//**
   README.md
```

### 📥 0.6 Download Competition Data
Replace `<competition-name>` with the Kaggle slug, e.g. `titanic`, `house-prices-advanced-regression-techniques`.

> **Note:** The command line for downloading each dataset from Kaggle is usually available on the competition page, typically at the bottom. Always check there for the exact command.

Colab / local (same commands):
```bash
!kaggle competitions download -c <competition-name> -p data/raw --force
```
Then unzip (example):
```bash
!unzip -o data/raw/<competition-name>.zip -d data/raw/
```
OR for datasets (not competitions):
```bash
!kaggle datasets download -d <owner>/<dataset-slug> -p data/raw
```

### 🔎 0.7 Verify Files in Notebook
**//TODO why use python not ls and cat//????** we have been using command line for each step so far, so it makes sense to execute `cat data/raw/train.csv | head -n 1`
But I don't know the equivelent on Windows **//TOOD//???**

```python
import os, pandas as pd
os.listdir('data/raw')
train = pd.read_csv('data/raw/train.csv')
train.head()
```

### ⚡ 0.8 Enable GPU (If Needed for DL / CV)
In Colab: Runtime → Change runtime type → Hardware accelerator: GPU (or T4 / A100 if available). Re-run the setup cell after switch.

### 🔒 0.9 Safe Handling of Secrets
Never commit `kaggle.json` to GitHub. Add to `.gitignore` if working locally:
```
echo 'kaggle.json' >> .gitignore
```

### 🛠️ 0.10 Common Errors & Fixes
| Issue | Symptom | Fix |
|-------|---------|-----|
| 403 - Forbidden | Download blocked | You didn’t accept competition rules. Visit page & click Join. |
| 401 - Unauthorized | API key rejected | Re-create token in Settings, replace old `kaggle.json`. |
| File not found | `No such file` after unzip | Check actual zip name in `data/raw`, maybe extra version suffix. |
| Permission denied | `kaggle.json` ignored | Ensure `chmod 600 ~/.kaggle/kaggle.json`. |
| Slow / timeout | Large dataset | Download locally then upload subset; or use Kaggle Notebook with data attached. |

### 🚅 0.11 (Optional) Speed Booster Workflow
1. Start in Colab for experimentation.
2. When stable, export notebook (`File → Download .ipynb`) and store in `notebooks/` in GitHub repo.
3. Convert to script if needed: `jupyter nbconvert --to script 01_eda.ipynb` (local).
4. Automate submissions with a small shell or Python helper (later stage).

Once this section is done you are ready to proceed to competition selection.

---

### 📓 Kaggle Notebooks

Kaggle Notebooks are an in-browser Jupyter environment provided by Kaggle for running code, sharing analyses, and making submissions directly on the platform.

#### 🛠️ How to Use Kaggle Notebooks

1. **Create a Notebook:**
   - Go to any competition or dataset page.
   - Click "Code" → "New Notebook".

2. **Attach Data:**
   - Use the "Add Data" button to attach competition datasets or public datasets.

3. **Write & Run Code:**
   - Use Python or R.
   - Install additional packages with `!pip install <package>` if needed.

4. **Save & Share:**
   - Save your work frequently.
   - Share notebooks publicly or keep them private.

5. **Submit to Competition:**
   - For competitions, generate your submission file (e.g., `submission.csv`).
   - Click "Submit to Competition" from the notebook interface.

#### 💡 Tips

- GPU/TPU resources are available for deep learning tasks.
- Notebooks are versioned; you can revert to previous versions.
- Use "Copy & Edit" to fork public notebooks for your own experiments.

#### 🏆 Example: Submitting to a Competition

```python
# Example: Save predictions for submission
submission.to_csv('submissions/submission.csv', index=False) # saved to our submission folder.
```
Then use the notebook's "Submit" button to upload your file.
Or use kaggle API for submission, for example `!kaggle competitions submit -c <competition-name> -f submissions/submission.csv -m "<Your message for submission>"`

---

### 🏁 Joining Competitions

1. Browse competitions at [Kaggle Competitions](https://www.kaggle.com/competitions).
2. Click on a competition, read the description and rules.
3. Click "Join Competition" and accept the rules.
4. Download data or use Kaggle Notebooks as described above.
5. Make submissions and track your leaderboard position.

