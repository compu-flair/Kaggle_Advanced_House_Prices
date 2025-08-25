import streamlit as st
from views.house_price import run_house_price_app
from views.custom_linear_app import custom_model_linear_regression_app
from views.custom_xgboost import custom_model_xgboost_app
from configs.config import HOUSE_PRICE_TAB, CUSTOM_APP_TAB, XGBOOST_APP_TAB

# Configure Streamlit page settings
st.set_page_config(page_title="House Prices App", layout="wide")

# Main page title
st.title("Linear Regression & XGBoost Module")

# Tab names for sidebar navigation
tab_names = [HOUSE_PRICE_TAB, CUSTOM_APP_TAB, XGBOOST_APP_TAB, "About"]

# Sidebar for navigation between tabs
with st.sidebar:
    st.header("Navigation")
    selected_tab = st.radio("Select a tab:", tab_names, index=0)

# Display content based on selected tab
if selected_tab == tab_names[0]:
    st.session_state["selected_tab"] = HOUSE_PRICE_TAB
    run_house_price_app()  # House price prediction module
elif selected_tab == tab_names[1]:
    st.session_state["selected_tab"] = CUSTOM_APP_TAB
    custom_model_linear_regression_app()  # Custom linear regression module
elif selected_tab == tab_names[2]:
    st.session_state["selected_tab"] = XGBOOST_APP_TAB
    custom_model_xgboost_app()  # Custom XGBoost module
elif selected_tab == tab_names[3]:
    st.session_state["selected_tab"] = "About"
    st.header("About")
    st.write("This app demonstrates house price prediction using Streamlit.")  # About section
