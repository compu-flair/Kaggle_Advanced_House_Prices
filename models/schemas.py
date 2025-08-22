from pydantic import BaseModel, conint, confloat

# Only use the specified features (excluding SalePrice)
FEATURES = [
    ("AllSF", confloat(ge=0)),
    ("OverallQual", conint(ge=1, le=10)),
    ("NeighborPrice", confloat(ge=0)),
    ("GrLivArea", conint(ge=0)),
    ("ExterQual_", conint(ge=0)),
    ("NeighborBin", conint(ge=0)),
    ("KitchenQual_", conint(ge=0)),
    ("SimplOverallQual", conint(ge=0)),
    ("TotalBsmtSF", confloat(ge=0)),
    ("GarageCars", confloat(ge=0)),
    ("TotalBath", confloat(ge=0)),
    ("GarageScore", confloat(ge=0)),
    ("GarageArea", confloat(ge=0)),
    ("1stFlrSF", conint(ge=0)),
    ("BsmtQual_", conint(ge=0)),
    ("OverallGrade", conint(ge=0)),
    ("FullBath", conint(ge=0)),
    ("TotRmsAbvGrd", conint(ge=0)),
    ("Now_YearBuilt", conint(ge=0)),
    ("YearBuilt", conint(ge=1800, le=2024)),
]

# Dynamically create the Pydantic model
class GaragePredictRequest(BaseModel):
    AllSF: confloat(ge=0)
    OverallQual: conint(ge=1, le=10)
    NeighborPrice: confloat(ge=0)
    GrLivArea: conint(ge=0)
    ExterQual_: conint(ge=0)
    NeighborBin: conint(ge=0)
    KitchenQual_: conint(ge=0)
    SimplOverallQual: conint(ge=0)
    TotalBsmtSF: confloat(ge=0)
    GarageCars: confloat(ge=0)
    TotalBath: confloat(ge=0)
    GarageScore: confloat(ge=0)
    GarageArea: confloat(ge=0)
    stFlrSF: conint(ge=0)
    BsmtQual_: conint(ge=0)
    OverallGrade: conint(ge=0)
    FullBath: conint(ge=0)
    TotRmsAbvGrd: conint(ge=0)
    Now_YearBuilt: conint(ge=0)
    YearBuilt: conint(ge=1800, le=2024)