from pydantic import BaseModel, conint, confloat

# List of features used for prediction, with their types and constraints.
# Each tuple contains the feature name and its Pydantic type constraint.
FEATURES = [
    ("AllSF", confloat(ge=0)),                # Total square footage, must be >= 0
    ("OverallQual", conint(ge=1, le=10)),     # Overall material and finish quality, 1-10
    ("NeighborPrice", confloat(ge=0)),        # Average price in neighborhood, must be >= 0
    ("GrLivArea", conint(ge=0)),              # Above ground living area, must be >= 0
    ("ExterQual_", conint(ge=0)),             # Exterior quality (encoded), must be >= 0
    ("NeighborBin", conint(ge=0)),            # Neighborhood bin (encoded), must be >= 0
    ("KitchenQual_", conint(ge=0)),           # Kitchen quality (encoded), must be >= 0
    ("SimplOverallQual", conint(ge=0)),       # Simplified overall quality (encoded), must be >= 0
    ("TotalBsmtSF", confloat(ge=0)),          # Total basement area, must be >= 0
    ("GarageCars", confloat(ge=0)),           # Number of garage cars, must be >= 0
    ("TotalBath", confloat(ge=0)),            # Total number of bathrooms, must be >= 0
    ("GarageScore", confloat(ge=0)),          # Garage score (custom feature), must be >= 0
    ("GarageArea", confloat(ge=0)),           # Garage area, must be >= 0
    ("1stFlrSF", conint(ge=0)),               # First floor square footage, must be >= 0
    ("BsmtQual_", conint(ge=0)),              # Basement quality (encoded), must be >= 0
    ("OverallGrade", conint(ge=0)),           # Overall grade (custom feature), must be >= 0
    ("FullBath", conint(ge=0)),               # Number of full bathrooms, must be >= 0
    ("TotRmsAbvGrd", conint(ge=0)),           # Total rooms above ground, must be >= 0
    ("Now_YearBuilt", conint(ge=0)),          # Years since built (custom feature), must be >= 0
    ("YearBuilt", conint(ge=1800, le=2024)),  # Year built, between 1800 and 2024
]

# Pydantic model for validating prediction requests.
# Each field corresponds to a feature, with type and constraints.
class GaragePredictRequest(BaseModel):
    AllSF: confloat(ge=0)                     # Total square footage
    OverallQual: conint(ge=1, le=10)          # Overall quality (1-10)
    NeighborPrice: confloat(ge=0)             # Neighborhood average price
    GrLivArea: conint(ge=0)                   # Above ground living area
    ExterQual_: conint(ge=0)                  # Exterior quality (encoded)
    NeighborBin: conint(ge=0)                 # Neighborhood bin (encoded)
    KitchenQual_: conint(ge=0)                # Kitchen quality (encoded)
    SimplOverallQual: conint(ge=0)            # Simplified overall quality (encoded)
    TotalBsmtSF: confloat(ge=0)               # Total basement area
    GarageCars: confloat(ge=0)                # Number of garage cars
    TotalBath: confloat(ge=0)                 # Total number of bathrooms
    GarageScore: confloat(ge=0)               # Garage score (custom feature)
    GarageArea: confloat(ge=0)                # Garage area
    stFlrSF: conint(ge=0)                     # First floor square footage (typo: should be '1stFlrSF')
    BsmtQual_: conint(ge=0)                   # Basement quality (encoded)
    OverallGrade: conint(ge=0)                # Overall grade (custom feature)
    FullBath: conint(ge=0)                    # Number of full bathrooms
    TotRmsAbvGrd: conint(ge=0)                # Total rooms above ground
    Now_YearBuilt: conint(ge=0)               # Years since built (custom feature)
    YearBuilt: conint(ge=1800, le=2024)       # Year built (between 1800 and 2024)