import streamlit as st
from views.house_price import run_house_price_app
from views.custom_app import custom_model_linear_regression_app
from configs.config import HOUSE_PRICE_TAB, CUSTOM_APP_TAB

st.set_page_config(page_title="House Prices App", layout="wide")

st.title("Linear Regression Module")

tab_names = [HOUSE_PRICE_TAB, CUSTOM_APP_TAB, "About"]
# house_price_tab,custom_tab,about_tab = st.tabs(tab_names)

with st.sidebar:
    st.header("Navigation")
    selected_tab = st.radio("Select a tab:", tab_names, index=0)

if selected_tab == tab_names[0]:
    st.session_state["selected_tab"] = HOUSE_PRICE_TAB
    run_house_price_app()
elif selected_tab == tab_names[1]:
    st.session_state["selected_tab"] = CUSTOM_APP_TAB
    custom_model_linear_regression_app()
elif selected_tab == tab_names[2]:
    st.session_state["selected_tab"] = "About"
    st.header("About")
    st.write("This app demonstrates house price prediction using Streamlit.")
