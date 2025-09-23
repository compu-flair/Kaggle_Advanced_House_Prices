# Practice Creating Project Step-by-Step Instructions

This document provides step-by-step instructions for creating a project. Follow each step carefully to ensure successful completion.

You're required to follow and think the steps below to create the project. Please try to implement each step on your own before looking at the provided code snippets.

Click reveal to see the code snippets for each step.
<details>
  <summary>Reveal</summary>
  Correct, you shall press on this button when you are stuck or want to see the solution.
</details>

## Step 1: Kaggle Notebook, EDA, Feauture Engineering and Model Training

We will be doing this step by creating a notebook locally and explore everything in the notebook.

Create a notebook in your name or as `data_cleaning_and_feature_engineering.ipynb`.

Before anything, we need to join the compitition at Kaggle from this link <https://www.kaggle.com/c/house-prices-advanced-regression-techniques>  

1. We need to download the dataset from the competition page. Can you use Kaggle command line to do it? If you don't have Kaggle command line installed, please follow the instruction from this link <https://www.kaggle.com/docs/api> to install it.

Note: download the dataset to a folder named `data` in your current working directory.

<details>
    <summary>Reveal</summary>
    You can use the following command to download the dataset:

```python
!kaggle competitions download -c house-prices-advanced-regression-techniques -p data
```

</details>

2. After downloading the dataset, we need to unzip the dataset. Can you use Python to do it?

<details>
    <summary>Reveal</summary>
    You can use the following code to unzip the dataset:

```python
import zipfile
import os
with zipfile.ZipFile(os.path.join('data','house-prices-advanced-regression-techniques.zip'), 'r') as zip_ref:
    zip_ref.extractall('data')
```

</details>

3. Now, we need to make necessary imports for our notebook. Can you import the following libraries: pandas, numpy, matplotlib, seaborn, sklearn?

<details>
    <summary>Reveal</summary>

```python
# Import pandas for data manipulation and analysis
import pandas as pd

# Import numpy for numerical operations and array handling
import numpy as np
# Import matplotlib for plotting and visualizations
import matplotlib.pyplot as plt
# Import seaborn for statistical data visualization (built on top of matplotlib)
import seaborn as sns

# Set seaborn plot style to 'whitegrid' for better readability of plots
sns.set_style('whitegrid')
# Enable inline plotting in Jupyter Notebook so plots appear in the output cells
%matplotlib inline

# Import display from IPython to display DataFrames and other objects in a readable format
from IPython.display import display

# Import warnings module to control warning messages
import warnings
# Ignore all warnings to keep notebook output clean and focused
warnings.filterwarnings('ignore')
```

</details>

4. Now, we need to load the dataset. Can you load the dataset using pandas?

Note: display the first 5 rows of both the training and test datasets.

<details>
    <summary>Reveal</summary>

```python
# Load training and test datasets from CSV files
train_data = pd.read_csv(os.path.join('data', 'train.csv'))
test_data = pd.read_csv(os.path.join('data', 'test.csv'))


print(train_data.shape)
display(train_data.head(1))
print(test_data.shape)
display(test_data.head(1))
```

</details>

5. Now, we need to combine the training and test datasets for feature engineering. Can you combine the training and test datasets using pandas?

<details>
    <summary>Reveal</summary>

```python

# Combine train and test datasets for feature engineering and preprocessing.
# - The first line concatenates the training data (up to and including 'SalePrice') and the test data.
# - This is useful for applying transformations (like encoding or imputing missing values) consistently across both datasets.
combined_data = pd.concat([train_data.loc[:, : 'SalePrice'], test_data])

# Align columns to match the test_data structure.
# - This ensures that combined_data has the same columns as test_data, which is important for making predictions later.
combined_data = combined_data[test_data.columns]

# Display the first row of the combined dataset for a quick preview.
display(combined_data.head(1))

# Print the shape (number of rows and columns) of the combined dataset.
print(combined_data.shape)
```

</details>

8. We would like to see how many missing values are there in each column of in the combined dataset, can you print how many missing values are there in each column of the combined dataset?

<details>
    <summary>Reveal</summary>

```python

has_null_columns = combined_data.columns[combined_data.isnull().any()].tolist()


combined_null = combined_data[has_null_columns].isnull().sum()

combined_null
```

</details>

9. We would like to convert the `MSSubClass` column to a categorical column. Can you do it?

Hint: Check the data description for more information about the `MSSubClass` column.

<details>
    <summary>Reveal</summary>

```python


mssubclass_dict = {
    20: 'SC20',
    30: 'SC30',
    40: 'SC40',
    45: 'SC45',
    50: 'SC50',
    60: 'SC60',
    70: 'SC70',
    75: 'SC75',
    80: 'SC80',
    85: 'SC85',
    90: 'SC90',
    120: 'SC120',
    150: 'SC150',
    160: 'SC160',
    180: 'SC180',
    190: 'SC190',
}


combined_data['MSSubClass'] = combined_data['MSSubClass'].replace(mssubclass_dict)
```

</details>

10. We would like to now how many features we need to hundle in terms of missing values, can you check it please?

<details>
    <summary>Reveal</summary>

```python
print(f"{len(combined_null[combined_null>0])} features to be handled")
```

</details>

12. Before we take any decision about how to fill the missing values, we would like to know the mean, median and cv of each values as well as the number of missing rows. Can you do it?

<details>
    <summary>Reveal</summary>

```python
for feature, missing_count in combined_null[combined_null > 0].items():
    print(f"Feature: {feature} Missing: {missing_count}")
    col_data = combined_data[feature]
    if pd.api.types.is_numeric_dtype(col_data):
        mean = col_data.mean()
        median = col_data.median()
        std = col_data.std()
        cv = std / mean if mean != 0 else float('nan')
        print(f"  Mean: {mean}")
        print(f"  Median: {median}")
        print(f"  CV: {cv}")
    else:
        categories = col_data.dropna().unique()
        print(f"  Categories: {categories}")
    print()
```

</details>

13. Interesting insights, right?, the number of missing values in each column is different, and the type of the column is different as well. But if the number of missing values is small, we can handle it differently than if the number of missing values is large. Please get us these values!

<details>
    <summary>Reveal</summary>

```python
# Find numeric features with missing value count less than 100
numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
subset_cols = []
for col in numeric_cols:
    missing_count = combined_data[col].isnull().sum()
    if 0 < missing_count < 100 and col not in subset_cols:
        subset_cols.append(col)
    if missing_count > 0: ## show all (even above 100)
        print(f"{col}: {missing_count}")
print(f"cols we use to drop na rows: {subset_cols}")
```

</details>

15. Notice there're so many missing values in `LotFrontage` column, we would like to fill it using KNN imputer. Can you do it?

Hint: Use the columns `LotFrontage`, `LotArea`, `LotShape`, `LotConfig` since they are related to each other.

Note: You need to encode the categorical columns before using KNN imputer.

<details>
    <summary>Reveal</summary>

```python

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# Select relevant columns for KNN imputation
knn_features = ['LotFrontage', 'LotArea', 'LotShape', 'LotConfig']

# Prepare a copy for imputation
knn_data = combined_data[knn_features].copy()

# Encode categorical features
for col in ['LotShape', 'LotConfig']:
    le = LabelEncoder()
    knn_data[col] = le.fit_transform(knn_data[col].astype(str))

# Apply KNN imputer
imputer = KNNImputer(n_neighbors=5)
knn_imputed = imputer.fit_transform(knn_data)

# Impute only missing values in combined_data['LotFrontage']
lotfrontage_missing = combined_data['LotFrontage'].isnull()
combined_data.loc[lotfrontage_missing, 'LotFrontage'] = knn_imputed[lotfrontage_missing, 0] ## 0 for LotFrontage
```

</details>

16. Now, can you visualize the imputed LotFrontage values?

<details>
    <summary>Reveal</summary>

```python
plt.figure(figsize=(6, 4))
plt.hist(combined_data['LotFrontage'])
plt.hist(knn_imputed[lotfrontage_missing, 0], bins=30, edgecolor='black')
mean_lotfrontage = combined_data['LotFrontage'].mean()
median_lotfrontage = combined_data['LotFrontage'].median()
plt.axvline(mean_lotfrontage, color='red', linestyle='--', label=f'Mean: {mean_lotfrontage:.2f}')
plt.axvline(median_lotfrontage, color='green', linestyle='-.', label=f'Median: {median_lotfrontage:.2f}')
plt.yscale("log")
plt.legend()
plt.xlabel('Imputed LotFrontage')
plt.ylabel('Frequency')
plt.title('Histogram of Imputed LotFrontage')
plt.show()
```

</details>

17. Interesting, right? can you do the same steps for `GarageYrBlt` column?

Hint: use columns  `GarageYrBlt` column using KNN imputer. Use the columns `GarageYrBlt`, `GarageType`, `GarageFinish`, `GarageCars`, `GarageArea`, `GarageQual`, and `GarageCond` as they are related.

<details>
    <summary>Reveal</summary>

```python

# Select relevant columns for KNN imputation
knn_features = ["GarageYrBlt", "GarageType", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond"]

# Prepare a copy for imputation
knn_data = combined_data[knn_features].copy()

# Encode categorical features
for col in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
    le = LabelEncoder()
    knn_data[col] = le.fit_transform(knn_data[col].astype(str))

# Apply KNN imputer
imputer = KNNImputer(n_neighbors=5)
knn_imputed = imputer.fit_transform(knn_data)

# Impute only missing values in combined_data['GarageYrBlt']
GarageYrBlt_missing = combined_data['GarageYrBlt'].isnull()
combined_data.loc[GarageYrBlt_missing, 'GarageYrBlt'] = knn_imputed[GarageYrBlt_missing, 0]
```

</details>

18. Don't forget to visualize the imputed `GarageYrBlt` values!.

<details>
    <summary>Reveal</summary>

```python
# Plot histogram of imputed GarageYrBlt values
plt.figure(figsize=(6, 4))
plt.hist(combined_data['GarageYrBlt'], bins=30, edgecolor='black', alpha=0.7, label='All GarageYrBlt')
plt.hist(knn_imputed[lotfrontage_missing, 0], bins=30, edgecolor='black', label='Imputed GarageYrBlt')
mean_garageyrblt = combined_data['GarageYrBlt'].mean()
median_garageyrblt = combined_data['GarageYrBlt'].median()
plt.axvline(mean_garageyrblt, color='red', linestyle='--', label=f'Mean: {mean_garageyrblt:.2f}')
plt.axvline(median_garageyrblt, color='green', linestyle='-.', label=f'Median: {median_garageyrblt:.2f}')
plt.yscale("log")
plt.legend(loc='upper center')
plt.xlabel('Imputed GarageYrBlt')
plt.ylabel('Frequency')
plt.title('Histogram of Imputed GarageYrBlt')
plt.show()
```

</details>

19. Remember we had columns with a very small number of missing values. What do you think we should do about them? The columns are: `MasVnrArea`, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath`, `BsmtHalfBath`, `GarageCars`, and `GarageArea`.

 <details>
    <summary>Reveal</summary>
Correct!, we can simply drop the rows with missing values in these columns.

```python

# Remove 'GarageYrBlt' and 'logtfrontage' from subset_cols since they no longer have missing values
subset_cols = [col for col in subset_cols if col not in ['GarageYrBlt', 'logtfrontage']]

## method 1

# Since number of missing values is few comparing to ~3000 rows in the combined data, we simply drop them
# Drop rows with missing values in these columns
combined_data.dropna(subset=subset_cols, inplace=True)
```

</details>

20. Now, we would like to know which features in the categorical columns are ordinal and which ones are nominal. We will be using ChatGPT to help us with this task. First, can you extract the categorical columns from the combined dataset?

<details>
    <summary>Reveal</summary>

```python
## Define categorical columns 
categorical_cols = [col for col in combined_data.columns if col not in numeric_cols]
print(f"# of categorical: {len(categorical_cols)}")
print(categorical_cols)
```

</details>

21. Now, remember there's a file named `data_description.txt` in the data folder that contains the description of each column in the dataset. Can you use this file and ChatGPT to help us classify the categorical columns into ordinal and nominal?

<details>
    <summary>Reveal</summary>

```python

## read the description of data
with open('data/data_description.txt', 'r') as f:
    data_description_txt = f.read()

## make a dictionary with keys being categorical columns and values being their descriptions

feature_descriptions = {}
lines = data_description_txt.split('\n')
current_feature = None
current_description = []

for line in lines:
    line = line.strip()
    if not line:
        continue
    # Check if line starts with a feature name followed by ':'
    if ':' in line:
        feature_name = line.split(':')[0].strip()
        if feature_name in categorical_cols:
            # Save previous feature description
            if current_feature and current_description:
                feature_descriptions[current_feature] = '\n'.join(current_description)
            # Start new feature
            current_feature = feature_name
            current_description = [line]
        # elif current_feature:
        #     current_description.append(line)
    # elif current_feature:
    else:
        current_description.append(line)

# Save last feature description
if current_feature and current_description:
    feature_descriptions[current_feature] = '\n'.join(current_description)

# !pip install langchain-openai langchain-ollama python-dotenv

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv(".env")

llm = ChatOpenAI(
    temperature=1, model="gpt-4o-mini", 
    api_key=os.getenv("openai_api_key")
)

## for a given categorical feature and its description, determine if it is ordered or unordered by nature
schema = {
    "title": "Categorical_Feature_Ordering",
    "description": "Determine if the given categorical feature is ordered or unordered based on its description.",
    "type": "object",
    "properties": {
        "is_ordered": {"type": "boolean", "description": "True if the feature is ordered (ordinal), False if unordered (nominal)."},
    },
    "required": ["is_ordered"]
}

## Structured output
structure_llm = llm.with_structured_output(schema)


## make a dictionary that gives the categorical type for each feature
import json
categorical_type_file = 'extra_data/categorical_type.json'
if os.path.exists(categorical_type_file):
    print('reading from stored file')
    with open(categorical_type_file, 'r') as f:
        categorical_type = json.load(f)
else: 
    categorical_type = {}
    for feature in categorical_cols:
        result = structure_llm.invoke(f"feature:\n{feature}\n\ndescription:\n{feature_descriptions[feature]}")

        categorical_type[feature] = result['is_ordered']
    with open(categorical_type_file, 'w') as f:
        json.dump(categorical_type, f, indent=2)
```

</details>

22. Good job! Now we need to one hot encode the nominal categorical columns and keep our ordinal categorical columns in a sepearte variable.

<details>
    <summary>Reveal</summary>

```python

## for unordered features use one hot encoder, i.e. we will have one distribution of dataset for each category
ordinal_categories = []
for ifeature, feature in enumerate(categorical_cols):
    print("===="*15)
    print("feature: ", feature)
    print("----\ndescription:\n",feature_descriptions[feature],"\n----\n")
    print("is_ordered", categorical_type[feature])
    print("===="*15)

    if not categorical_type[feature]:
        # One-hot encode the MSSubClass feature and concatenate to combined_data
        ms_subclass_dummies = pd.get_dummies(combined_data[feature], prefix=feature)
        combined_data = pd.concat([combined_data.drop(columns=[feature]), ms_subclass_dummies], axis=1)
        print(f"Add this number of columns: {len(ms_subclass_dummies.columns) - 1 }")
    else:
        ordinal_categories.append(feature)
print(f"ordinal categorical variables to be addressed now:\n{ordinal_categories}")
```

</details>

23. We would like to fill the missing values in the dataset. Can you create a function named `fill_missing_values_combined_data` to fill the missing values in the combined dataset taking two arguments, `column` and `value`?

<details>
    <summary>Reveal</summary>

```python
def fill_missing_combined_data(column, value):
    combined_data.loc[combined_data[column].isnull(),column] = value
```

</details>

24. Now, We would like to label encode all the nominal categorical columns. Can you do it?
Note: We would like to keep `Na` values as it is.

<details>
    <summary>Reveal</summary>

```python

# Ordinal/nominal maps derived from your descriptions
enc_maps = {

    'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},

    # Higher is "better"/more regular
    'LotShape':    {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3},

    # Flatter is "better"
    'LandContour': {'Low': 0, 'HLS': 1, 'Bnk': 2, 'Lvl': 3},

    # Gentler is "better"
    'LandSlope':   {'Sev': 0, 'Mod': 1, 'Gtl': 2},

    # Quality scales: Ex > Gd > TA > Fa > Po
    'ExterQual':   {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'ExterCond':   {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'HeatingQC':   {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

    # Basement qualities (+ NA=0 = no basement)
    'BsmtQual':    {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtCond':    {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

    # Exposure: Gd > Av > Mn > No > NA
    'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},

    # Finished types: GLQ > ALQ > BLQ > Rec > LwQ > Unf > NA
    'BsmtFinType1': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'BsmtFinType2': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},

    # Functional: Typ best → Salvage worst
    'Functional':  {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8},

    # Fireplace quality (+ NA=0 = no fireplace)
    'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

    # Garage finish (+ NA=0 = no garage)
    'GarageFinish': {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},

    # Garage quality/condition (+ NA=0 = no garage)
    'GarageQual':  {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond':  {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

    # Pool quality (+ NA=0 = no pool)
    'PoolQC':      {'NA': 0, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
}


## "NA" in features above is set to null. return them back to "NA"
columns_with_NA = [col for col in enc_maps if 'NA' in enc_maps[col]]
for feature in columns_with_NA:
    fill_missing_combined_data(feature, "NA")

# Apply replacements in one shot
combined_data = combined_data.replace(enc_maps)

```

</details>

25. We still have a few missing values in the categorical columns, can you check them and take a decision on your own?

<details>
    <summary>Reveal</summary>

```python
for feature in combined_data.columns:
    missing_count = combined_data[feature].isnull().sum()
    if 0 < missing_count:
        if missing_count < 10:
            ## since a few missing points, we drop them
            combined_data.dropna(subset=[feature], inplace=True)
            print(f"{missing_count} missing values in {feature}: dropped the rows")
        else:
            print(f"{missing_count} missing values in {feature}: to be handled later")
```

</details>

26. Now,  we would like to compute scores and give us an overall idea about the quality of the house, create the following features `OverallGrade`, `GarageGrade`, `ExterGrade`, `KitchenScore`, `FireplaceScore`, `GarageScore`, `PoolScore`, `TotalBath`, `AllSF`, `AllFlrsSF` and `AllPorchSF`. Can you do it?

Hint: Inspect columns with similar names to understand how to create these new features. it includes multiplication between two columns.

<details>
    <summary>Reveal</summary>

```python
## Features that engineered in [0]
combined_data["OverallGrade"] = combined_data["OverallQual"] * combined_data["OverallCond"]
combined_data["GarageGrade"] = combined_data["GarageQual"] * combined_data["GarageCond"]
combined_data["ExterGrade"] = combined_data["ExterQual"] * combined_data["ExterCond"]
combined_data["KitchenScore"] = combined_data["KitchenAbvGr"] * combined_data["KitchenQual"]
combined_data["FireplaceScore"] = combined_data["Fireplaces"] * combined_data["FireplaceQu"]
combined_data["GarageScore"] = combined_data["GarageArea"] * combined_data["GarageQual"]
combined_data["PoolScore"] = combined_data["PoolArea"] * combined_data["PoolQC"]
combined_data["TotalBath"] = combined_data["BsmtFullBath"] + (0.5 * combined_data["BsmtHalfBath"]) + combined_data["FullBath"] + (0.5 * combined_data["HalfBath"])
combined_data["AllSF"] = combined_data["GrLivArea"] + combined_data["TotalBsmtSF"]
combined_data["AllFlrsSF"] = combined_data["1stFlrSF"] + combined_data["2ndFlrSF"]
combined_data["AllPorchSF"] = combined_data["OpenPorchSF"] + combined_data["EnclosedPorch"] + combined_data["3SsnPorch"] + combined_data["ScreenPorch"]
```

</details>

27. Now, can you create a feature representing the local bedroom count and size adjustment? Store it in a column named `BedCntSize_Local`.

<details>
    <summary>Reveal</summary>

```python
combined_data["BedCntSize_Local"] = (
    combined_data["GrLivArea"] * combined_data["BedroomAbvGr"] / combined_data["TotRmsAbvGrd"].clip(lower=1)
)
```

</details>

28. Now, Create a feature representing finished basement livable area adjustment. Store it in a column named `BsmtLivableAdj`.

<details>
    <summary>Reveal</summary>

```python
combined_data["BsmtLivableAdj"] = combined_data["BsmtFinSF1"] * combined_data["BsmtExposure"] * combined_data["BsmtQual"]
```

</details>

29. We would like to create a `ParkingUtilityAppeal` feature that represents the overall parking utility and appeal of the property. Can you create this new feature?

This feature is constructed using several columns, each with specific weights:

- `PavedDrive_Y`: Adds 0.3 if the driveway is paved (Yes).
- `PavedDrive_P`: Adds 0.15 if the driveway is partially paved.
- `PavedDrive_N`: Subtracts 0.1 if the driveway is not paved.
- `GarageType_CarPort`: Adds 0.4 if the garage type is a carport.
- `GarageCars`: Adds up to 0.25, scaled by the number of garage cars (maximum 3).
- `LotFrontage`: Adds 0.2 times the ratio of the lot frontage to its median value.
- `LotConfig_Corner`: Adds 0.15 if the lot is on a corner.
- `LotConfig_FR2`: Adds 0.18 (0.15 × 1.2) if the lot is a frontage type FR2.
- `LotConfig_FR3`: Adds 0.21 (0.15 × 1.4) if the lot is a frontage type FR3.
- `LotConfig_CulDSac`: Adds 0.075 (0.15 × 0.5) if the lot is a cul-de-sac.

Each component reflects the contribution of driveway paving, garage presence and quality, and lot configuration to the overall parking utility and appeal.

<details>
    <summary>Reveal</summary>

```python
combined_data["ParkingUtilityAppeal"] = (
    0.3 * combined_data.get("PavedDrive_Y", 0) +
    0.15 * combined_data.get("PavedDrive_P", 0) -
    0.1 * combined_data.get("PavedDrive_N", 0) +
    0.4 * combined_data.get("GarageType_CarPort", 0) +
    0.25 * combined_data["GarageCars"].clip(0, 3) / 3 +
    0.2 * (combined_data["LotFrontage"] / combined_data["LotFrontage"].median()) +
    0.15 * (
        combined_data.get("LotConfig_Corner", 0) +
        1.2 * combined_data.get("LotConfig_FR2", 0) +
        1.4 * combined_data.get("LotConfig_FR3", 0) +
        0.5 * combined_data.get("LotConfig_CulDSac", 0)
    )
)
```

</details>

30. Create a new feature named `FloorSurfaceQuality` that represents the overall quality of the flooring surfaces in the house. This feature is calculated by combining several factors: it adds the `OverallQual` (overall quality of the house) and `KitchenQual` (quality of the kitchen), subtracts the ratio of `LowQualFinSF` (low quality finished square feet) to `GrLivArea` (above ground living area), and further subtracts a small adjustment based on the difference between `YrSold` (year sold) and `YearRemodAdd` (year remodeled). This composite measure provides a nuanced assessment of the flooring surface quality, considering both quality ratings and the impact of lower quality areas and recent renovations.

<details>
    <summary>Reveal</summary>

```python

combined_data["FloorSurfaceQuality"] = (
    combined_data["OverallQual"] +
    combined_data["KitchenQual"] -
    (combined_data["LowQualFinSF"] / combined_data["GrLivArea"].clip(lower=1)) -
    0.02 * (combined_data["YrSold"] - combined_data["YearRemodAdd"])
)
```

</details>


31. We would like to create a `RoofBuyerRisk` column that represents the risk associated with the roof of the house from a buyer's perspective. This feature is calculated by estimating the expected lifespan of the roof based on its material (using one-hot encoded roof material columns), and then comparing the roof's age (years since the most recent of remodeling or construction) to this expected lifespan. The formula subtracts the fraction of expected life remaining from 1, so higher values indicate higher risk (an older roof relative to its expected life). If no roof material is specified, a default expected life of 25 years is used.


<details>
    <summary>Reveal</summary>


```python
# Calculate denominator: sum of all roof material one-hot columns
den = (
    combined_data.get("RoofMatl_ClyTile", 0) +
    combined_data.get("RoofMatl_Metal", 0) +
    combined_data.get("RoofMatl_WdShake", 0) +
    combined_data.get("RoofMatl_WdShngl", 0) +
    combined_data.get("RoofMatl_Membran", 0) +
    combined_data.get("RoofMatl_CompShg", 0) +
    combined_data.get("RoofMatl_Tar&Grv", 0) +
    combined_data.get("RoofMatl_Roll", 0)
)

# Estimate expected roof life (years) based on material type
exp_life = (
    50 * combined_data.get("RoofMatl_ClyTile", 0) +
    40 * combined_data.get("RoofMatl_Metal", 0) +
    30 * combined_data.get("RoofMatl_WdShake", 0) +
    25 * combined_data.get("RoofMatl_WdShngl", 0) +
    25 * combined_data.get("RoofMatl_Membran", 0) +
    20 * combined_data.get("RoofMatl_CompShg", 0) +
    18 * combined_data.get("RoofMatl_Tar&Grv", 0) +
    12 * combined_data.get("RoofMatl_Roll", 0) +
    25 * (den == 0)
) / den.clip(lower=1)

# Calculate roof age: years since most recent of YearRemodAdd or YearBuilt
roof_age = (combined_data["YrSold"] - combined_data[["YearRemodAdd", "YearBuilt"]].max(axis=1)).clip(lower=0)

# RoofBuyerRisk: 1 minus fraction of expected life remaining
combined_data["RoofBuyerRisk"] = 1 - ((exp_life - roof_age).clip(lower=0) / exp_life.clip(lower=1))
```
</details>

32. We would like to create a `HVAC_AgeEff` feature that represents the efficiency and risk associated with the house's HVAC (heating, ventilation, and air conditioning) system from a buyer's perspective. This feature is calculated by:

- Estimating the age of the HVAC system as the years since the most recent remodel or construction.
- Assigning weighted scores to the heating type, with higher weights for more efficient or modern systems.
- Assigning weighted scores to the heating quality, with higher weights for better quality ratings.
- Boosting the score if central air conditioning is present.
- Penalizing the score for older systems by dividing by a factor based on the system's age.

The final score combines these factors to reflect both the quality and age of the HVAC system, helping buyers assess potential risk and efficiency.

<details>
    <summary>Reveal</summary>

```python
# Calculate the age of the HVAC system as the years since the most recent remodel or build
age = (combined_data["YrSold"] - combined_data[["YearRemodAdd", "YearBuilt"]].max(axis=1)).clip(lower=0)

# Compute a weighted score for heating type:
# - Assign higher weights to more efficient/modern heating types
# - If none of the listed types are present, add a fallback weight
type_w = (
    1.0 * combined_data.get("Heating_GasA", 0) +
    0.9 * combined_data.get("Heating_GasW", 0) +
    0.85 * combined_data.get("Heating_OthW", 0) +
    0.8 * combined_data.get("Heating_Floor", 0) +
    0.7 * combined_data.get("Heating_Wall", 0) +
    0.6 * combined_data.get("Heating_Grav", 0)
) + 0.85 * (
    (combined_data.get("Heating_GasA", 0) +
     combined_data.get("Heating_GasW", 0) +
     combined_data.get("Heating_OthW", 0) +
     combined_data.get("Heating_Floor", 0) +
     combined_data.get("Heating_Wall", 0) +
     combined_data.get("Heating_Grav", 0)) == 0
)

# Compute a weighted score for heating quality:
# - Higher weights for better quality ratings
# - If no quality rating is present, add a fallback weight
qc_w = (
    1.2 * combined_data.get("HeatingQC_Ex", 0) +
    1.1 * combined_data.get("HeatingQC_Gd", 0) +
    1.0 * combined_data.get("HeatingQC_TA", 0) +
    0.9 * combined_data.get("HeatingQC_Fa", 0) +
    0.8 * combined_data.get("HeatingQC_Po", 0)
) + 1.0 * (
    (combined_data.get("HeatingQC_Ex", 0) +
     combined_data.get("HeatingQC_Gd", 0) +
     combined_data.get("HeatingQC_TA", 0) +
     combined_data.get("HeatingQC_Fa", 0) +
     combined_data.get("HeatingQC_Po", 0)) == 0
)

# Calculate the HVAC age efficiency feature:
# - Multiply heating type and quality scores
# - Boost if central air is present
# - Penalize for older systems (age effect)
combined_data["HVAC_AgeEff"] = (
    type_w * qc_w *
    (1 + 0.2 * combined_data.get("CentralAir_Y", combined_data.get("CentralAir", 0)).clip(0, 1))
    / (1 + age / 20.0)
)
```
</details>

33. We would like to create a column named `EnvelopeEfficiency` that represents the overall efficiency of the building envelope, which includes the exterior walls, roof, and basement. This feature is calculated by averaging the exterior quality (`ExterQual`), exterior condition (`ExterCond`), and heating quality (`HeatingQC`), boosting the score if central air is present, and penalizing for older envelope age (years since the latest remodel or build).



<details>
    <summary>Reveal</summary>

```python
age = (combined_data["YrSold"] - combined_data[["YearRemodAdd", "YearBuilt"]].max(axis=1)).clip(lower=0)
ca = combined_data.get("CentralAir_Y", combined_data.get("CentralAir", 0)).clip(0, 1)
combined_data["EnvelopeEfficiency"] = (
    (combined_data["ExterQual"] + combined_data["ExterCond"] + combined_data["HeatingQC"]) / 3
) * (1 + 0.2 * ca) / (1 + age / 20.0)
```
</details>

34. We would like to create an `EnergyCertProxy` feature that acts as a proxy for the energy certification or efficiency of the house. This feature combines several factors:

- The age of the building envelope (years since the latest remodel or construction).
- The presence of central air conditioning.
- Whether the electrical system is modern (`SBrkr`).
- The heating quality, exterior quality and condition, and overall quality.

The formula boosts the score for homes with central air and modern electrical systems, and penalizes for older envelope age. The calculation is:

- Add together the heating quality, the average of exterior quality and condition, and half of the overall quality.
- Multiply by a factor that increases for central air and modern electrical.
- Divide by a factor that increases with the age of the envelope.

This provides a single score reflecting the home's energy efficiency and modernity.


<details>
    <summary>Reveal</summary>

```python

# Calculate the age of the building envelope (years since latest remodel or build)
age = (combined_data["YrSold"] - combined_data[["YearRemodAdd", "YearBuilt"]].max(axis=1)).clip(lower=0)

# Get central air indicator (1 if present, 0 if not)
ca = combined_data.get("CentralAir_Y", combined_data.get("CentralAir", 0)).clip(0, 1)

# Get electrical system indicator (1 if 'SBrkr', 0 otherwise)
em = combined_data.get("Electrical_SBrkr", 0).clip(0, 1)

# Create a proxy for energy certification/efficiency:
# - Combines heating quality, exterior quality/condition, and overall quality
# - Boosts for central air and modern electrical system
# - Penalizes for older envelope age
combined_data["EnergyCertProxy"] = (
    (combined_data["HeatingQC"] + (combined_data["ExterQual"] + combined_data["ExterCond"]) / 2 + 0.5 * combined_data["OverallQual"])
    * (1 + 0.2 * ca + 0.05 * em)
    / (1 + age / 20.0)
)
```
</details>


35. We would like to create a column named `OwnedVsLeasedSystems` that represents the balance between owned and leased systems in the house, adjusted for improvements and age. This feature is calculated as follows:

- Count the number of owned systems using the one-hot encoded columns for `SaleType_WD`, `SaleType_CWD`, `SaleType_VWD`, and `SaleType_New`.
- Count the number of leased systems using the one-hot encoded columns for `SaleType_Con`, `SaleType_ConLw`, `SaleType_ConLI`, and `SaleType_ConLD`.
- Calculate an improvement factor as the average of the presence of central air (`CentralAir_Y` or `CentralAir`) and a modern electrical system (`Electrical_SBrkr`).
- Calculate an age factor that penalizes older remodels (more recent remodels have a higher value).
- The final feature is computed by subtracting 0.8 times the leased systems from the owned systems, multiplying by the improvement factor, and then multiplying by the age factor.

This composite feature helps quantify the relative value of owned versus leased systems, while accounting for improvements and the age of the property.


<details>
    <summary>Reveal</summary>

```python
# Calculate the number of owned systems based on SaleType one-hot columns
owned = (
    combined_data.get("SaleType_WD", 0) +
    combined_data.get("SaleType_CWD", 0) +
    combined_data.get("SaleType_VWD", 0) +
    combined_data.get("SaleType_New", 0)
)

# Calculate the number of leased systems based on SaleType one-hot columns
leased = (
    combined_data.get("SaleType_Con", 0) +
    combined_data.get("SaleType_ConLw", 0) +
    combined_data.get("SaleType_ConLI", 0) +
    combined_data.get("SaleType_ConLD", 0)
)

# Improvement factor: average of central air and modern electrical system indicators
impr = (
    combined_data.get("CentralAir_Y", combined_data.get("CentralAir", 0)).clip(0, 1) +
    combined_data.get("Electrical_SBrkr", 0).clip(0, 1)
) / 2

# Age factor: penalizes older remodels (more recent = higher value)
agefac = 1 / (1 + (combined_data["YrSold"] - combined_data["YearRemodAdd"]).clip(lower=0) / 10.0)

# Final feature: combines owned/leased systems, improvement, and age factors
combined_data["OwnedVsLeasedSystems"] = (owned - 0.8 * leased) * (1 + 0.3 * impr) * agefac
```
</details>


36. Now we would like to create another feature named `SmartConvenienceEff` that represents the overall convenience and efficiency of the home's smart and hardware systems. This feature combines several factors:

- Modern electrical system (`Electrical_SBrkr` one-hot column): weighted at 0.5.
- Central air conditioning (`CentralAir_Y` one-hot column, or `CentralAir` if not one-hot): weighted at 0.4.
- Finished garage (`GarageFinish_Fin` one-hot column plus half of `GarageFinish_RFn`): weighted at 0.1.
- The average of heating quality (`HeatingQC`), overall quality (`OverallQual`), and functionality (`Functional`).
- The score is penalized for older homes by dividing by (1 + age of the envelope / 15), where age is years since the latest remodel or build.

This composite score reflects both the presence of modern hardware and smart features, as well as the overall efficiency and convenience, adjusted for the age of the property.

<details>
    <summary>Reveal</summary>

```python
# Create a zero-filled Series to use as a default for missing one-hot columns
z = pd.Series(0, index=combined_data.index)

# Calculate the age of the home envelope (years since latest remodel or build)
age = (combined_data["YrSold"] - combined_data[["YearRemodAdd", "YearBuilt"]].max(axis=1)).clip(lower=0)

# Get one-hot encoded indicator for modern electrical system (SBrkr)
elec = combined_data.get("Electrical_SBrkr", z)

# Get one-hot encoded indicator for central air (Y), fallback to CentralAir if not one-hot
ca = combined_data.get("CentralAir_Y", combined_data.get("CentralAir", z))

# Get one-hot encoded indicators for garage finish types
gfin = combined_data.get("GarageFinish_Fin", z)
grfn = combined_data.get("GarageFinish_RFn", z)

# Compute a hardware convenience score:
# - 0.5 weight for modern electrical
# - 0.4 weight for central air
# - 0.1 weight for finished garage (Fin + 0.5*RFn)
# Clip each to [0,1] to ensure binary indicators
hw = 0.5 * elec.clip(0, 1) + 0.4 * ca.clip(0, 1) + 0.1 * (gfin + 0.5 * grfn).clip(0, 1)

# Combine hardware score with average of HeatingQC, OverallQual, and Functional (for smart/efficiency context)
# Divide by (1 + age/15) to penalize older systems
combined_data["SmartConvenienceEff"] = hw * (
    0.5 + 0.5 * ((combined_data["HeatingQC"] + combined_data["OverallQual"] + combined_data["Functional"]) / 3.0)
) / (1 + age / 15.0)
```
</details>


37. Now, let's create a feature named `StagingDeclutterScore` to represent the overall staging and declutter quality of the house. This score combines multiple factors:

- 35%: Average of overall quality, condition, and kitchen quality.
- 15%: Average of exterior quality and condition.
- 15%: Normalized average room size.
- 10%: Storage score (garage cars and unfinished basement ratio).
- 10%: Curb appeal score (porch and deck area normalized by living area).
- 10%: Paved driveway indicator (Y minus half N).
- 5%: Normal sale condition indicator.
- Subtract 20%: Ratio of low-quality finished area to living area (penalty).


<details>
    <summary>Reveal</summary>


```python
# Create a zero-filled Series for missing one-hot columns
z = pd.Series(0, index=combined_data.index)

# Average room size
avg_room = combined_data["GrLivArea"] / combined_data["TotRmsAbvGrd"].clip(lower=1)

# Storage score
storage = combined_data["GarageCars"].clip(lower=0) / 2 + (
    combined_data["BsmtUnfSF"] / combined_data["TotalBsmtSF"].replace(0, 1)
).clip(0, 1)

# Curb appeal score
curb = (combined_data["OpenPorchSF"] + combined_data["WoodDeckSF"]) / combined_data["GrLivArea"].clip(lower=1)

# StagingDeclutterScore calculation
combined_data["StagingDeclutterScore"] = (
    0.35 * ((combined_data["OverallQual"] + combined_data["OverallCond"] + combined_data["KitchenQual"]) / 3)
    + 0.15 * ((combined_data["ExterQual"] + combined_data["ExterCond"]) / 2)
    + 0.15 * (avg_room / avg_room.median())
    + 0.1 * storage
    + 0.1 * curb
    + 0.1 * (combined_data.get("PavedDrive_Y", z) - 0.5 * combined_data.get("PavedDrive_N", z))
    + 0.05 * combined_data.get("SaleCondition_Normal", z)
    - 0.2 * (combined_data["LowQualFinSF"] / combined_data["GrLivArea"].clip(lower=1))
)
``` 
</details>

38. Create another column `OutdoorUsableAmenityScore`

This feature represents the overall usability and appeal of outdoor amenities for each house. It combines several factors:

- **Amenity score:** Deck, porch, pool, and screen areas, normalized by living area.
- **Terrain score:** Gentle/moderate slope, level contour, regular/irregular lot shape.
- **Privacy score:** Weighted by fence types for privacy/quality.
- **Context score:** Positive weights for good location, negative for busy/rail/arterial surroundings.
- **Lot area ratio:** Lot area normalized by living area.

You can use the following code to create this feature:


<details>
    <summary>Reveal</summary>

```python
# Create a zero-filled Series to use as a default for missing one-hot columns
z = pd.Series(0, index=combined_data.index)

# Amenity score: combines deck, porch, pool, and screen areas, normalized by living area
amen = (
    combined_data["WoodDeckSF"] +
    combined_data["OpenPorchSF"] +
    0.6 * combined_data["EnclosedPorch"] +
    0.8 * combined_data["ScreenPorch"] +
    0.9 * combined_data["3SsnPorch"] +
    0.2 * combined_data["PoolArea"]
) / combined_data["GrLivArea"].clip(lower=1)

# Terrain score: combines gentle slope, moderate slope, level contour, regular lot shape, and irregular lot shape
terrain = (
    combined_data.get("LandSlope_Gtl", z) +
    0.5 * combined_data.get("LandSlope_Mod", z) +
    combined_data.get("LandContour_Lvl", z) +
    combined_data.get("LotShape_Reg", z) +
    0.5 * combined_data.get("LotShape_IR1", z)
) / 3.0

# Privacy score: combines fence types with different weights for privacy/quality
privacy = (
    0.6 * combined_data.get("Fence_GdPrv", z) +
    0.4 * combined_data.get("Fence_MnPrv", z) +
    0.3 * combined_data.get("Fence_GdWo", z) +
    0.1 * combined_data.get("Fence_MnWw", z)
)

# Context score: positive weights for good location, negative for busy/rail/arterial surroundings
context = (
    0.5 * (
        combined_data.get("Condition1_PosA", z) +
        combined_data.get("Condition1_PosN", z) +
        combined_data.get("Condition2_PosA", z) +
        combined_data.get("Condition2_PosN", z)
    )
    - 0.3 * (
        combined_data.get("Condition1_Artery", z) +
        combined_data.get("Condition1_Feedr", z) +
        combined_data.get("Condition1_RRAn", z) +
        combined_data.get("Condition1_RRAe", z) +
        combined_data.get("Condition1_RRNn", z) +
        combined_data.get("Condition1_RRNe", z) +
        combined_data.get("Condition2_Artery", z) +
        combined_data.get("Condition2_Feedr", z) +
        combined_data.get("Condition2_RRAn", z) +
        combined_data.get("Condition2_RRAe", z) +
        combined_data.get("Condition2_RRNn", z) +
        combined_data.get("Condition2_RRNe", z)
    )
)

# Final feature: weighted sum of amenity, terrain, privacy, lot area ratio, and context
combined_data["OutdoorUsableAmenityScore"] = (
    0.45 * amen +
    0.25 * terrain +
    0.15 * privacy +
    0.1 * (combined_data["LotArea"] / combined_data["GrLivArea"].clip(lower=1)) +
    0.05 * context
).clip(lower=0)
```
</details>

39. We can also create another column named `InteriorDaylightScore` that represents the overall daylight quality and interior lighting potential of the house. This feature combines several factors:

- **Window daylight contribution:** Weighted sum of porch areas with windows, normalized by living area.
- **Basement exposure:** Weighted sum of good, average, and minimum basement exposure indicators.
- **Multi-story score:** Weighted sum of house styles indicating more stories.
- **Corner lot score:** Weighted sum for corner and frontage lot configurations.
- **Aspect/environment score:** Positive weights for good location, negative for busy/arterial surroundings.
- **Topography score:** Weighted sum for level contour and regular lot shape, penalty for low contour.
- **Interior quality:** Average of kitchen and overall quality, normalized.
- **Age factor:** Penalizes older homes based on years since remodel or build.


<details>
    <summary>Reveal</summary>


```python
# Create a zero-filled Series to use as a default for missing one-hot columns
z = pd.Series(0, index=combined_data.index)

# Clip living area to avoid division by zero
area = combined_data["GrLivArea"].clip(lower=1)

# Estimate window daylight contribution:
windowy = (
    0.5 * combined_data["3SsnPorch"] +
    0.4 * combined_data["ScreenPorch"] +
    0.2 * combined_data["OpenPorchSF"]
) / area

# Basement exposure score:
exposure = (
    1.0 * combined_data.get("BsmtExposure_Gd", z) +
    0.6 * combined_data.get("BsmtExposure_Av", z) +
    0.3 * combined_data.get("BsmtExposure_Mn", z)
)

# Multi-story score:
stories = (
    combined_data.get("HouseStyle_2Story", z) +
    combined_data.get("HouseStyle_2.5Fin", z) +
    0.6 * combined_data.get("HouseStyle_1.5Fin", z) +
    0.5 * combined_data.get("HouseStyle_SLvl", z) +
    0.4 * combined_data.get("HouseStyle_SFoyer", z)
)

# Corner lot score:
corner = (
    0.6 * combined_data.get("LotConfig_Corner", z) +
    0.8 * combined_data.get("LotConfig_FR2", z) +
    1.0 * combined_data.get("LotConfig_FR3", z)
)

# Aspect/environment score:
aspect_env = (
    0.3 * (
        combined_data.get("Condition1_PosA", z) +
        combined_data.get("Condition1_PosN", z) +
        combined_data.get("Condition2_PosA", z) +
        combined_data.get("Condition2_PosN", z)
    )
    - 0.2 * (
        combined_data.get("Condition1_Artery", z) +
        combined_data.get("Condition1_Feedr", z) +
        combined_data.get("Condition2_Artery", z) +
        combined_data.get("Condition2_Feedr", z)
    )
)

# Topography score:
topo = (
    0.3 * combined_data.get("LandContour_Lvl", z) +
    0.1 * combined_data.get("LotShape_Reg", z) -
    0.1 * combined_data.get("LandContour_Low", z)
)

# Interior quality score:
quality = ((combined_data["KitchenQual"] + combined_data["OverallQual"]) / 20.0).clip(upper=1)

# Age factor:
age = (combined_data["YrSold"] - combined_data[["YearRemodAdd", "YearBuilt"]].max(axis=1)).clip(lower=0)
age_factor = 1 / (1 + age / 25.0)

# Final InteriorDaylightScore:
combined_data["InteriorDaylightScore"] = (
    0.35 * windowy +
    0.20 * exposure +
    0.15 * stories +
    0.10 * corner +
    0.10 * aspect_env +
    0.05 * topo +
    0.05 * quality
).clip(lower=0) * age_factor
```

</details>


40. Now, we would like to build a feature named `MonthlyCostRestrictions` that estimates the monthly cost of living in each house, adjusted for restrictions such as HOA and zoning. This feature combines several factors:

- **Size adjustment:** Scaled values of above-ground living area, basement area, and garage area.
- **Utilities and heating:** Adds weights for certain utility types and heating systems, penalizes for gas heating and central air.
- **HOA and zoning restrictions:** Sums up indicators for HOA-related subclasses and restrictive zoning/building types.
- **Final score:** Multiplies the monthly cost estimate by a factor reflecting the level of restrictions.

<details>
    <summary>Reveal</summary>


```python
size = (
    combined_data['GrLivArea'] / combined_data['GrLivArea'].median()
    + 0.5 * combined_data['TotalBsmtSF'] / combined_data['TotalBsmtSF'].median()
    + 0.25 * combined_data['GarageArea'] / combined_data['GarageArea'].median()
)
monthly = (
    size
    + 0.5 * (
        combined_data.get('Utilities_ELO', 0)
        + combined_data.get('Utilities_NoSeWa', 0)
        + combined_data.get('Utilities_NoSewr', 0)
    )
    + combined_data.filter(regex=r'^Heating_(Floor|Grav|Wall|OthW)$').sum(axis=1)
    - (
        combined_data.get('Heating_GasA', 0)
        + 0.5 * combined_data.get('Heating_GasW', 0)
    )
    - 0.25 * combined_data.get('CentralAir_Y', 0)
).clip(lower=0)
hoa = (
    combined_data.get('MSSubClass_120', 0)
    + combined_data.get('MSSubClass_150', 0)
    + combined_data.get('MSSubClass_160', 0)
    + combined_data.get('MSSubClass_180', 0)
    + combined_data.get('BldgType_TwnhsE', 0)
    + combined_data.get('BldgType_TwnhsI', 0)
)
restrict = (
    hoa
    + combined_data.get('MSZoning_RL', 0)
    + combined_data.get('MSZoning_RP', 0)
    + combined_data.get('BldgType_1Fam', 0)
    - (
        combined_data.get('MSZoning_RM', 0)
        + combined_data.get('MSZoning_RH', 0)
        + combined_data.get('BldgType_Duplx', 0)
        + combined_data.get('BldgType_2FmCon', 0)
    )
).clip(lower=0)
combined_data['MonthlyCostRestrictions'] = monthly * (1 + 0.5 * restrict)
```

</details>


41. We would like to create a column named `HighRiskMarket` that represents the risk level of the housing market for each property. This feature combines several factors:

- **Rail risk:** Sums up indicators for proximity to railroad conditions and adds a neighborhood railroad indicator.
- **Wind risk:** Combines severe land slope, high land contour, banked land contour, and roof styles associated with higher wind risk, subtracting the hip roof style (which is less risky).
- **Hail risk:** Sums up roof materials prone to hail damage and subtracts those that are more resistant.
- **Wildfire risk:** Sums up exterior materials prone to wildfire and adds a timber neighborhood indicator.
- **Flood risk:** Sums up indicators for low land contour and neighborhoods prone to flooding.

The final `HighRiskMarket` score is the sum of these risk components, clipped at zero to avoid negative values.


<details>
    <summary>Reveal</summary>

```python
rail = combined_data.filter(regex=r'^Condition[12]_RR').sum(axis=1) + combined_data.get('Neighborhood_IDOTRR',0)
wind = (combined_data.get('LandSlope_Sev',0)+combined_data.get('LandContour_HLS',0)+combined_data.get('LandContour_Bnk',0)
    + combined_data.filter(regex=r'^RoofStyle_(Gable|Gambrel|Shed|Flat)$').sum(axis=1) - combined_data.get('RoofStyle_Hip',0))
hail = combined_data.filter(regex=r'^RoofMatl_(WdShake|WdShngl|CompShg)$').sum(axis=1) - combined_data.filter(regex=r'^RoofMatl_(Metal|ClyTile|Membran|Tar&Grv)$').sum(axis=1)
wildfire = combined_data.filter(regex=r'^Exterior(1st|2nd)_(Wd.?Sdng|WdShing)$').sum(axis=1) + combined_data.get('Neighborhood_Timber',0)
flood = (combined_data.get('LandContour_Low',0)+combined_data.get('Neighborhood_BrkSide',0)+combined_data.get('Neighborhood_ClearCr',0)
     + combined_data.get('Neighborhood_Sawyer',0)+combined_data.get('Neighborhood_SawyerW',0))
combined_data['HighRiskMarket'] = (rail + wind + hail + wildfire + flood).clip(lower=0)
```
</details>


42. Create a column named `FEMAZoneElevationClaims` to represent the flood risk and elevation-related claims for each property. This feature combines several risk factors:

- **Water proximity risk:** Sums indicators for neighborhoods near water (`BrkSide`, `ClearCr`, `Sawyer`, `SawyerW`).
- **Elevation risk:** Adds low elevation and severe slope, half of moderate slope, and subtracts high land and bank contours (higher elevation is safer).
- **Basement risk:** Double penalty for poor basement condition, penalty for fair and typical condition, penalty for no basement exposure, and subtracts half the good basement exposure (safer).
- **Foundation adjustment:** Subtracts slab foundation and half penalty for poured concrete foundation (these are safer).
- The final score is the sum of these components, with double weight for water proximity, clipped at zero to avoid negative risk.

This composite score provides a proxy for FEMA flood zone and elevation-related claims.


<details>
    <summary>Reveal</summary>

```python
# Calculate water proximity risk: sum of one-hot indicators for neighborhoods near water
water_nb = (
    combined_data.get('Neighborhood_BrkSide', 0) +
    combined_data.get('Neighborhood_ClearCr', 0) +
    combined_data.get('Neighborhood_Sawyer', 0) +
    combined_data.get('Neighborhood_SawyerW', 0)
)

# Calculate elevation risk:
elev = (
    combined_data.get('LandContour_Low', 0) +
    combined_data.get('LandSlope_Sev', 0) +
    0.5 * combined_data.get('LandSlope_Mod', 0) -
    (combined_data.get('LandContour_HLS', 0) + combined_data.get('LandContour_Bnk', 0))
)

# Calculate basement risk:
bsmt = (
    2 * combined_data.get('BsmtCond_Po', 0) +
    combined_data.get('BsmtCond_Fa', 0) +
    0.5 * combined_data.get('BsmtCond_TA', 0) +
    combined_data.get('BsmtExposure_No', 0) -
    0.5 * combined_data.get('BsmtExposure_Gd', 0)
)

# Combine all risk factors for FEMA flood zone/elevation/claims proxy:
combined_data['FEMAZoneElevationClaims'] = (
    2 * water_nb + elev + bsmt -
    combined_data.get('Foundation_Slab', 0) -
    0.5 * combined_data.get('Foundation_PConc', 0)
).clip(lower=0)
```
</details>


43. Create a new column called `TaxRate_Reassessment_Afford` that represents a proxy for the tax rate, reassessment likelihood, and affordability of each property. This feature is calculated as follows:

- **Assessed value proxy:** Combines normalized above-ground living area, basement area, and garage area, then multiplies by normalized overall quality to reflect higher-value homes.
- **Zoning factor:** Adds weights for high-density residential and commercial zoning types.
- **Reassessment indicator:** Adds 1 if the house was sold within 3 years of a remodel, and 1 if sold within 5 years of construction.
- **Final score:** Multiplies the assessed value proxy by (1 + zoning factor) and by (1 + 0.5 × reassessment indicator) to estimate the overall tax rate, reassessment risk, and affordability.

This composite feature helps capture the impact of property size, quality, zoning, and recent changes on tax and affordability.

<details>
    <summary>Reveal</summary>

```python
# Estimate assessed value proxy:
assessed = (
    (combined_data['GrLivArea'] / combined_data['GrLivArea'].median()
     + 0.5 * combined_data['TotalBsmtSF'] / combined_data['TotalBsmtSF'].median()
     + 0.25 * combined_data['GarageArea'] / combined_data['GarageArea'].median())
    * (combined_data['OverallQual'] / combined_data['OverallQual'].median())
)

# Zoning factor:
zfac = (
    0.75 * combined_data.get('MSZoning_RH', 0)
    + 0.5 * combined_data.get('MSZoning_RM', 0)
    + 0.5 * combined_data.get('MSZoning_C', 0)
)

# Reassessment indicator:
reassess = (
    ((combined_data['YrSold'] - combined_data['YearRemodAdd']) <= 3).astype(int)
    + ((combined_data['YrSold'] - combined_data['YearBuilt']) <= 5).astype(int)
)

# Final feature: Tax rate/reassessment/affordability proxy
combined_data['TaxRate_Reassessment_Afford'] = assessed * (1 + zfac) * (1 + 0.5 * reassess)
```
</details>



44. Create another feature called `OptionValue_ADU_Attic_Expand` that estimates the option value for a property based on its potential for accessory dwelling units (ADU), finishable attic/basement space, and footprint expansion. This feature is a weighted sum of several components:

- **Lot value:** Considers normalized lot area, favorable lot configurations (corner, frontage), regular shape, level contour, and penalizes for moderate/severe slope and low/high/bank contours.
- **Attic value:** Rewards unfinished attic/upper stories and penalizes finished attic/upper stories.
- **Basement value:** Uses normalized unfinished basement area.
- **Zoning value:** Rewards multi-family/commercial zoning and penalizes single-family/restricted zoning.
- **PUD penalty:** Penalizes condo/townhome subclasses, which have less expansion potential.
- **Garage value:** Rewards detached/2-type/carport garages and penalizes built-in/basement garages.
- **Utility value:** Rewards all public utilities and penalizes limited utilities.

The final feature is the weighted sum of these components, clipped to be non-negative.

<details>
    <summary>Reveal</summary>

```python
# Estimate the option value for a property based on ADU (Accessory Dwelling Unit) potential, finishable attic/basement, and footprint expansion.

# Lot value: combines normalized lot area, favorable lot configurations (corner, frontage), regular shape, level contour,
# and penalizes for moderate/severe slope and low/high/bank contours.
lot = (
       combined_data['LotArea'] / combined_data['LotArea'].median()
       + 0.5 * (
              combined_data.get('LotConfig_Corner', 0)
              + combined_data.get('LotConfig_FR2', 0)
              + combined_data.get('LotConfig_FR3', 0)
       )
       + 0.25 * combined_data.get('LotShape_Reg', 0)
       + 0.25 * combined_data.get('LandContour_Lvl', 0)
       - 0.5 * (
              combined_data.get('LandSlope_Mod', 0)
              + combined_data.get('LandSlope_Sev', 0)
              + combined_data.get('LandContour_Low', 0)
              + combined_data.get('LandContour_HLS', 0)
              + combined_data.get('LandContour_Bnk', 0)
       )
)

# Attic value: positive for unfinished attic/upper stories, negative for finished attic/upper stories.
attic = (
       combined_data.get('HouseStyle_1.5Unf', 0)
       + combined_data.get('HouseStyle_2.5Unf', 0)
       + combined_data.get('MSSubClass_45', 0)
       - (
              combined_data.get('HouseStyle_1.5Fin', 0)
              + combined_data.get('HouseStyle_2.5Fin', 0)
              + combined_data.get('MSSubClass_40', 0)
       )
)

# Basement value: normalized unfinished basement area.
bsmt = combined_data['BsmtUnfSF'] / combined_data['BsmtUnfSF'].median()

# Zoning value: positive for multi-family/commercial zoning, negative for single-family/restricted zoning.
z = (
       combined_data.get('MSZoning_RM', 0)
       + combined_data.get('MSZoning_RH', 0)
       + 0.5 * combined_data.get('MSZoning_C', 0)
       - (
              combined_data.get('MSZoning_RL', 0)
              + combined_data.get('MSZoning_RP', 0)
              + combined_data.get('MSZoning_A', 0)
              + combined_data.get('MSZoning_I', 0)
       )
)

# PUD penalty: penalizes for condo/townhome subclasses (less expansion potential).
pud_pen = (
       combined_data.get('MSSubClass_120', 0)
       + combined_data.get('MSSubClass_150', 0)
       + combined_data.get('MSSubClass_160', 0)
       + combined_data.get('MSSubClass_180', 0)
)

# Garage value: positive for detached/2-type/carport garages, negative for built-in/basement garages.
gar = (
       combined_data.get('GarageType_Detchd', 0)
       + combined_data.get('GarageType_2Types', 0)
       + 0.5 * combined_data.get('GarageType_CarPort', 0)
       - 0.5 * (
              combined_data.get('GarageType_BuiltIn', 0)
              + combined_data.get('GarageType_Basment', 0)
       )
)

# Utility value: positive for all public utilities, negative for limited utilities.
util = (
       combined_data.get('Utilities_AllPub', 0)
       - (
              combined_data.get('Utilities_ELO', 0)
              + combined_data.get('Utilities_NoSeWa', 0)
              + combined_data.get('Utilities_NoSewr', 0)
       )
)

# Final option value feature: weighted sum of all components, clipped to be non-negative.
combined_data['OptionValue_ADU_Attic_Expand'] = (
       1.0 * lot
       + 0.8 * attic
       + 0.7 * bsmt
       + 1.2 * z
       + 0.8 * gar
       + 0.5 * util
       - 0.8 * pud_pen
).clip(lower=0)
```
</details>


45. Create a column named `RemoteWorkConnectivity` that represents the overall connectivity and suitability of a house for remote work. This feature is a weighted sum of several factors:

- **Utilities score:** Adds 1 for all public utilities, subtracts 1 for each limited utility type.
- **Access score:** Adds 1 for paved street, 0.5 for paved alley, subtracts 0.25 for gravel street/alley, and adds 0.25 for a paved driveway.
- **Wiring score:** Adds 1 for modern breaker electrical systems, subtracts 0.5 for each fuse or mixed system.
- **Recency score:** Uses the average percentile rank of the build and remodel year to reward newer homes.
- **Density score:** Adds 1 for commercial/multifamily zoning, subtracts 0.5 for single-family/restricted zoning.
- **Arterial access score:** Adds 0.25 for each arterial or feeder road condition.
- **Campus proximity score:** Adds 1 for SWISU neighborhood, 0.5 for CollgCr neighborhood.
- **Lot frontage score:** Adds 0.4 times the normalized lot frontage (relative to the median).

The final `RemoteWorkConnectivity` score is the weighted sum of these components, clipped to be non-negative.

<details>
    <summary>Reveal</summary>

```python
# Calculate normalized lot frontage (relative to median)
front = combined_data['LotFrontage'] / combined_data['LotFrontage'].median()

# Utilities score: +1 for all public utilities, -1 for each limited utility type
utils = (
    combined_data.get('Utilities_AllPub', 0)
    - (
        combined_data.get('Utilities_ELO', 0)
        + combined_data.get('Utilities_NoSeWa', 0)
        + combined_data.get('Utilities_NoSewr', 0)
    )
)

# Access score: +1 for paved street, +0.5 for paved alley, -0.25 for gravel street/alley, +0.25 for paved driveway
access = (
    combined_data.get('Street_Pave', 0)
    + 0.5 * combined_data.get('Alley_Pave', 0)
    - 0.25 * combined_data.get('Street_Grvl', 0)
    - 0.25 * combined_data.get('Alley_Grvl', 0)
    + 0.25 * combined_data.get('PavedDrive_Y', 0)
)

# Wiring score: +1 for modern breaker, -0.5 for each fuse/mixed system
wiring = (
    combined_data.get('Electrical_SBrkr', 0)
    - 0.5 * (
        combined_data.get('Electrical_FuseA', 0)
        + combined_data.get('Electrical_FuseF', 0)
        + combined_data.get('Electrical_FuseP', 0)
        + combined_data.get('Electrical_Mix', 0)
    )
)

# Recency score: average percentile rank of build and remodel year
recency = (
    combined_data['YearBuilt'].rank(pct=True)
    + combined_data['YearRemodAdd'].rank(pct=True)
) / 2

# Density score: +1 for commercial/multifamily zoning, -0.5 for single-family/restricted zoning
density = (
    combined_data.get('MSZoning_C', 0)
    + combined_data.get('MSZoning_RM', 0)
    + combined_data.get('MSZoning_RH', 0)
    - 0.5 * (
        combined_data.get('MSZoning_RL', 0)
        + combined_data.get('MSZoning_RP', 0)
    )
)

# Arterial access score: +0.25 for each arterial/feedr condition (sum across both Condition1 and Condition2)
arterial = 0.25 * combined_data.filter(regex=r'^Condition[12]_(Artery|Feedr)$').sum(axis=1)

# Campus proximity score: +1 for SWISU neighborhood, +0.5 for CollgCr
campus = (
    combined_data.get('Neighborhood_SWISU', 0)
    + 0.5 * combined_data.get('Neighborhood_CollgCr', 0)
)

# Final feature: weighted sum of all components, clipped to non-negative
combined_data['RemoteWorkConnectivity'] = (
    1.2 * utils
    + 0.8 * access
    + 0.8 * wiring
    + recency
    + 0.6 * density
    + arterial
    + 0.4 * front
    + 0.5 * campus
).clip(lower=0)
```
</details>

46. Finally, create a column named `PricingContext_Comps_DOM_Inventory` that represents the overall pricing context for each property. This feature combines several factors:

- **Recent comparable sales:** Uses the year and month of sale to rank each property by recency, so more recent sales have higher values.
- **Days on market (DOM) proxy:** Estimates inventory turnover by comparing the maximum monthly sales to the current month's sales (higher values indicate slower months and less inventory turnover).
- **Typicality:** Measures how close a home is to the median living area and overall quality, with homes closer to the median considered more typical.
- **Inventory/density proxy:** Sums indicators for high-density zoning, campus neighborhoods, subtracts cul-de-sac lots, and adds arterial/feedr road conditions.

The final feature is a weighted sum of typicality, recent sales rank, inventory proxy, minus the DOM proxy, clipped at zero to avoid negative values.


<details>
    <summary>Reveal</summary>

```python
# Combine year and month of sale into a fractional time variable
t = combined_data['YrSold'] + combined_data['MoSold'] / 12.0

# Rank each sale by time (percentile), so more recent sales have higher values
recent = t.rank(pct=True)

# Estimate number of sales per month (inventory proxy)
month_sales = combined_data.groupby('MoSold')['MoSold'].transform('size')

# DOM proxy: ratio of max monthly sales to current month sales (higher = slower month, less inventory turnover)
dom = month_sales.max() / month_sales

# Typicality: how close a home is to median living area and median overall quality
gl_med = combined_data['GrLivArea'].median()
oq_med = combined_data['OverallQual'].median()
typicality = 1.0 / (1.0 + (combined_data['GrLivArea'] / gl_med - 1.0).abs() + 0.5 * (combined_data['OverallQual'] / oq_med - 1.0).abs())

# Inventory/density proxy
inv = (
    combined_data.get('MSZoning_RH', 0) +
    combined_data.get('MSZoning_RM', 0) +
    0.5 * combined_data.get('MSZoning_C', 0) +
    0.3 * (combined_data.get('Neighborhood_SWISU', 0) + combined_data.get('Neighborhood_CollgCr', 0)) -
    0.4 * combined_data.get('LotConfig_CulDSac', 0) +
    0.2 * combined_data.filter(regex=r'^Condition[12]_(Artery|Feedr)$').sum(axis=1)
)

# Final feature: weighted sum of typicality, recent sales rank, inventory proxy, minus DOM proxy (clip at zero)
combined_data['PricingContext_Comps_DOM_Inventory'] = (
    0.6 * typicality + 0.5 * recent + 0.6 * inv - 0.7 * dom
).clip(lower=0)
```

</details>


47. Since we are almost done, we need to normalize our dataset, apply standard scaling to float columns, has high cardinality or not strictly binary. And apply min-max scaling to integer columns that has maximum values greater than 5.


<details>
    <summary>Reveal</summary>

```python
## normalize float-type columns
for col in combined_data.columns:
    if col == "Id": 
        continue  # Skip the Id column, as it's just an identifier
    s = combined_data[col]
    # If the column is float type, has high cardinality, or is not strictly binary (0/1), standardize it
    if s.dtype == 'float' or (s.nunique() > 10) or (not s.isin([0,1]).all()):
        mean = s.mean()
        std = s.std()
        combined_data[col] = (s - mean) / std  # Standardize: zero mean, unit variance
    # If the column's max value is greater than 5 (likely ordinal/categorical), scale to [0,1]
    elif s.max() > 5:
        combined_data[col] = s / s.max()
    else:
        # For low-cardinality columns (likely binary or small categorical), print unique values for inspection
        print(f"{col}: {s.unique()}")
```
</details>

48. Now it's time to handle outliers, please make a histogram to visualize the distribution of all columns

<details>
    <summary>Reveal</summary>

```python

# Plot a histogram for every column in X in log-scale
for i, col in enumerate(combined_data.columns.tolist()):
    try:
        print(f"feature: {col}")
        plt.figure(figsize=(6, 4))
        plt.hist(np.array(combined_data)[:, i], bins=30, edgecolor='black')
        plt.yscale("log")
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {col}')
        plt.show()
    except:
        pass
```
</details>


49. After visualizing the histograms, we noticed that some columns have outliers that need to be capped. can you apply mask to these columns and make the decision on which values to cap?


<details>
    <summary>Reveal</summary>

```python
# Create a mask to identify outlier rows in combined_data based on several feature thresholds.
# Each condition checks if the value in a column exceeds a specified threshold (indicating a potential outlier).
mask = (
    (combined_data['LotFrontage'] > 6) |        # Outlier if LotFrontage is unusually high
    (combined_data['LotArea'] > 10) |           # Outlier if LotArea is unusually high
    (combined_data['BsmtFinSF1'] > 5) |         # Outlier if BsmtFinSF1 is unusually high
    (combined_data['TotalBsmtSF'] > 6) |        # Outlier if TotalBsmtSF is unusually high
    (combined_data['1stFlrSF'] > 6) |           # Outlier if 1stFlrSF is unusually high
    (combined_data['WoodDeckSF'] > 8) |         # Outlier if WoodDeckSF is unusually high
    (combined_data['EnclosedPorch'] > 10) |     # Outlier if EnclosedPorch is unusually high
    (combined_data['MiscVal'] > 15) |           # Outlier if MiscVal is unusually high
    (combined_data['BsmtLivableAdj'] > 10)      # Outlier if BsmtLivableAdj is unusually high
)

# Remove rows identified as outliers from combined_data
combined_data = combined_data[~mask]

# Print the number of rows dropped due to outlier filtering
print(f"Number of dropped rows: {mask.sum()}")
```
</details>

50. We also need to normalize out Target Value in the training dataset, can you normalize it?

<details>
    <summary>Reveal</summary>

```python
train_data["SalePrice"] = (train_data["SalePrice"] - train_data["SalePrice"].mean()) / train_data["SalePrice"].std()
```
</details>

51. Can you visualize the target value distribution after normalization?


<details>
    <summary>Reveal</summary>

```python
plt.figure(figsize=(8, 4))
plt.hist(train_data["SalePrice"], bins=30, edgecolor='black')
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.title("Histogram of SalePrice in train_data")
plt.yscale("log")
plt.show()
```
</details>

52. Now we need to add SalePrice back to the combined_data and assign it to new dataframe called kaggle_train_data. At the same time we need to create another dataframe called kaggle_test_data that contains only test data.


<details>
    <summary>Reveal</summary>

```python
## Add the SalePrice back to train part of combined_data 
kaggle_train_data = train_data[["Id", "SalePrice"]].merge(combined_data, on="Id")
kaggle_test_data = combined_data[~combined_data["Id"].isin(kaggle_train_data["Id"])]
kaggle_train_data.head()
```
</details>

53. We don't need the Id column for training, drop it but keep a copy of it for the test dataset as we will need it for submission.

<details>
    <summary>Reveal</summary>

```python
## keep id of test for submission
test_Id = kaggle_test_data['Id'].tolist()

## Id has no information. Drop the column
kaggle_train_data.drop('Id', axis=1, inplace=True)
kaggle_test_data.drop('Id', axis=1, inplace=True)
```
</details>


54. We need to plot a histogram of the distribution of the maximum values column, and we need to plot it from the combined data, can you do it?

<details>
    <summary>Reveal</summary>

```python
data_matrix = kaggle_train_data.to_numpy(dtype=float) 
# Find the column in combined_data with the maximum value across all rows in data_matrix.
# - np.max(data_matrix, axis=0) computes the max value for each column (feature).
# - np.argmax(...) returns the index of the column with the highest max value.
max_col = combined_data.columns[np.argmax(np.max(data_matrix, axis=0))]

# Print the column name and its maximum value for inspection.
print(f"Column with max value ({combined_data[max_col].max()}): {max_col}")

# Plot a histogram of the 'MiscVal' column to visualize its distribution.
plt.hist(combined_data['MiscVal'])
```
</details>

55. For our feature selection, we will calculate the sample covariance matrix of the features in data_matrix. Then calculate the precision matrix and print these values, can you do it?

<details>
    <summary>Reveal</summary>

```python
# Calculate the sample covariance matrix of the features in data_matrix.
# - np.cov(data_matrix.T) computes the covariance matrix for columns (features) of data_matrix.
sigma = np.cov(data_matrix.T)

# Compute the precision matrix (inverse covariance).
# - np.linalg.pinv(sigma) calculates the Moore-Penrose pseudo-inverse of the covariance matrix.
# - The precision matrix is useful for feature selection and graphical models.
precision = np.linalg.pinv(sigma)
print("Max absolute value in precision matrix:", abs(precision).max())
print("Max absolute value in covariance matrix (sigma):", abs(sigma).max())
print("Max absolute value in sigma * precision (should be close to 1 for identity):", abs(np.dot(sigma, precision)).max())
```
</details>

56. For visualization, can plot the historgram of Precision Matrix for SalePrice column? Can you also do the same in scatter plot?


<details>
    <summary>Reveal</summary>

```python

# Find the index of "SalePrice" in kaggle_train_data columns
saleprice_idx = kaggle_train_data.columns.get_loc("SalePrice")

# Extract the corresponding row from the precision matrix
precision_row = precision[saleprice_idx, :]
## selection criterion for the features
precision_min = -1
precision_max = 1

plt.figure(figsize=(8, 4))
plt.hist(precision_row, bins=30, edgecolor='black')
plt.axvline(precision_min, color='red', linestyle='--', label='precision_min')
plt.axvline(precision_max, color='red', linestyle='--', label='precision_max')
plt.yscale("log")
plt.xlabel('Precision Value')
plt.ylabel('Frequency')
plt.title('Histogram of Precision Matrix Row for SalePrice')
plt.legend()
plt.show()
```
</details>


<details>
    <summary>Reveal</summary>

```python
# Plot the row
plt.figure(figsize=(12, 4))
plt.scatter(np.arange(len(precision_row)),precision_row, marker='.')
plt.axhline(precision_min, color='red', linestyle='--', label='precision_min')
plt.axhline(precision_max, color='red', linestyle='--', label='precision_max')
plt.title('Row of Precision Matrix Corresponding to SalePrice')
plt.xlabel('Feature Index')
plt.ylabel('Precision Value')
plt.legend()
plt.grid(True)
plt.show()
```
</details>

57. Now we need to select features based on the precision matrix, can you do it? Assign the new values into `X_train_kaggle`, `y_train_kaggle`, and `X_test_kaggle` respectively as numpy arrays.

Hint: For each feature, if its precision value is outside the interval [precision_min, precision_max],it is considered strongly associated (either positively or negatively) with SalePrice and is selected.

<details>
    <summary>Reveal</summary>

```python
# Select features based on the precision matrix row for SalePrice.
# For each feature, if its precision value is outside the interval [precision_min, precision_max],
# it is considered strongly associated (either positively or negatively) with SalePrice and is selected.
selected_features = [
    kaggle_train_data.columns[i]
    for i, val in enumerate(precision_row)
    if val > precision_max or val < precision_min
]
print(selected_features)



## drop SalePrice from selected_features
selected_features = [col for col in selected_features if col != 'SalePrice']

X_train_kaggle = np.array(kaggle_train_data[selected_features])
X_test_kaggle  = np.array(kaggle_test_data[selected_features])
y_train_kaggle = np.array(kaggle_train_data['SalePrice'])
```
</details>


58. Can you plot PCA of the selected features?

<details>
    <summary>Reveal</summary>

```python
from sklearn.decomposition import PCA

# Fit PCA on y_train_kaggle and transform
pca = PCA()
pca.fit(X_train_kaggle)
print("Explained variance ratio (first 10 components):", pca.explained_variance_ratio_[:10])
print("Total variance explained:", np.sum(pca.explained_variance_ratio_))

plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Variance Explained by PCA Components')
plt.grid(True)
plt.show()
```

</details>

59. Looks great, can you apply the PCA on our datasets?

<details>
    <summary>Reveal</summary>

```python
# Transform X_train_kaggle using PCA and keep only the first 10 components
# Find the minimum number of components needed to explain at least 95% of the variance
cum_var = np.cumsum(pca.explained_variance_ratio_)
n_components = np.searchsorted(cum_var, 0.999999999) + 1

X_train_kaggle_pca = pca.transform(X_train_kaggle)[:, :n_components]
X_test_kaggle_pca  = pca.transform(X_test_kaggle)[:, :n_components]
print("Shape of X_train_kaggle_pca10:", X_train_kaggle_pca.shape)
print("Shape of X_test_kaggle_pca10:", X_test_kaggle_pca.shape)
```
</details>

60. We are almost there, but before we train our model, we need to split our training dataset into training and validation datasets. Can you do it?

<details>
    <summary>Reveal</summary>

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_train_kaggle_pca, y_train_kaggle, test_size=0.2, random_state=42
)
```
</details>

61. Sounds great, now we can train our model. Can you start by Linear Regression and plot the regression line?

<details>
    <summary>Reveal</summary>

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fit linear regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Predict on test set
y_pred_linreg = linreg.predict(X_test)

# Evaluate performance
rmse_linreg = mean_squared_error(y_test, y_pred_linreg, squared=False)
print("Linear Regression Test RMSE:", rmse_linreg)

# Plot results
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_linreg, alpha=0.6)
plt.xlabel('True SalePrice (y_test)')
plt.ylabel('Predicted SalePrice (y_pred)')
plt.title('Linear Regression: Predicted vs True SalePrice')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
```
</details>

62. RMSE value of 0.299, can you try Ridge Regression and see if it improves the performance?



<details>
    <summary>Reveal</summary>

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Define parameter grid for ElasticNet
param_grid = {
    'alpha': np.arange(0.01, 2.0, 0.1)
}

# Initialize ElasticNet model
ridge = Ridge()

# Set up GridSearchCV
grid_search = GridSearchCV(
    ridge,
    param_grid,
    cv=3,
    n_jobs=-1
)

# Fit GridSearchCV on training data
grid_search.fit(X_train, y_train)

# Best estimator
best_model = grid_search.best_estimator_

# Predict on test set
y_pred = best_model.predict(X_test)

# Evaluate performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Best ElasticNet params:", grid_search.best_params_)
print("Test RMSE:", rmse)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('True SalePrice (y_test)')
plt.ylabel('Predicted SalePrice (y_pred)')
plt.title('ElasticNet: Predicted vs True SalePrice')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
```
</details>


63. Great, better results. You can take it from here and try to improve the performance further by trying different models and tuning hyperparameters. Our concern at the model is make prediction and submit to Kaggle. Can you make prediction on the test dataset and prepare submission file?


<details>
    <summary>Reveal</summary>

```python
submission_pred = best_model.predict(X_test_kaggle_pca)
submission = pd.DataFrame({'Id': test_Id, 'SalePrice': submission_pred})
submission.to_csv(os.path.join('data', 'submission_Ardavan.csv'), index=False)
```
</details>

64. Seems like we are done, can you use kaggle command line to submit our results?

<details>
    <summary>Reveal</summary>

```python
!kaggle competitions submit -c house-prices-advanced-regression-techniques -f data/submission_Ardavan.csv -m "Elastic net + pca"
```
</details>

