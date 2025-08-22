import streamlit as st
from pydantic import conint, confloat
import pickle
import pandas as pd
from models.schemas import GaragePredictRequest, FEATURES
from configs.config import HOUSE_PRICE_TAB as TABE_NAME

def run_house_price_app():
    # Load the model
    try:
        model = pickle.load(open('models/linear_regression_model.pkl', 'rb'))
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        st.error("Trained model file not found. Please train the model first by running the `data_cleaning_and_feature_engineering.ipynb` notebook.")
        return

    for fname, ftype in FEATURES:
        setattr(GaragePredictRequest, fname, ftype)

    st.title("House Price Prediction")
    st.subheader("Input Features")
    inputs = {}
    for fname, ftype in FEATURES:
        if ftype == confloat(ge=0):
            inputs[fname] = st.number_input(fname, min_value=0.0, value=0.0)
        elif ftype == conint(ge=1, le=10):
            inputs[fname] = st.number_input(fname, min_value=1, max_value=10, value=5)
        elif ftype == conint(ge=1800, le=2024):
            inputs[fname] = st.number_input(fname, min_value=1800, max_value=2024, value=2000)
        elif ftype == conint(ge=0):
            inputs[fname] = st.number_input(fname, min_value=0, value=0)
        else:
            inputs[fname] = st.number_input(fname, value=0)

    # Initialize session state for predictions
    if "predictions" not in st.session_state:
        st.session_state.predictions = []

    if st.button("Predict"):
        inputs["stFlrSF"] = inputs.pop("1stFlrSF")  
        request = GaragePredictRequest(**inputs)

        feature_dict = request.model_dump()
        feature_dict["1stFlrSF"] = feature_dict.pop("stFlrSF")
        features = pd.DataFrame([feature_dict])
        cols = features.columns.tolist()
        cols.remove("1stFlrSF")
        insert_idx = -6 if len(cols) >= 5 else len(cols)
        cols.insert(insert_idx, "1stFlrSF")
        features = features[cols]
        
        prediction = model.predict(features)
        st.session_state.predictions.append(float(prediction[0]))
        st.success(f"Predicted Value: {float(prediction[0]):.2f}")

    # Only show sidebar if this tab is selected
    if st.session_state.get("selected_tab") == TABE_NAME:
        with st.sidebar:
            st.markdown(
                """
                <style>
                [data-testid="stSidebar"] {
                    min-width: 400px;
                    max-width: 400px;
                    width: 400px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.line_chart(st.session_state.predictions or [0,1,2,3,4])
