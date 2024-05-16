import shap
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from pyarrow import parquet as pq
from catboost import CatBoostRegressor, Pool
import joblib

# Path of the trained model and data
MODEL_PATH = "model\catboost_model.cbm" 
DATA_PATH = "fulldata.parquet"

st.set_page_config(page_title="House Price Prediction Project")

@st.cache_resource
def load_data():
    data = pd.read_parquet(DATA_PATH)
    return data

def load_x_y(file_path):
    data = joblib.load(file_path)
    data.reset_index(drop=True, inplace=True)
    return data

def load_model():
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model

def calculate_shap(model, X_train, X_test):
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_cat_train = explainer.shap_values(X_train)
    shap_values_cat_test = explainer.shap_values(X_test)
    return explainer, shap_values_cat_train, shap_values_cat_test

def plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, home_id, X_test, X_train):
    # Visualize SHAP values for a specific customer
    home_index = X_test[X_test['Id'] == home_id].index[0]
    fig, ax_2 = plt.subplots(figsize=(6,6), dpi=200)
    shap.decision_plot(explainer.expected_value, shap_values_cat_test[home_index], X_test[X_test['Id'] == home_id])
    st.pyplot(fig)
    plt.close()

def display_shap_summary(shap_values_cat_train, X_train):
    # Create the plot summarizing the SHAP values
    shap.summary_plot(shap_values_cat_train, X_train, plot_type="bar", plot_size=(12,12))
    summary_fig, _ = plt.gcf(), plt.gca()
    st.pyplot(summary_fig)
    plt.close()

def display_shap_waterfall_plot(explainer, expected_value, shap_values, feature_names, max_display=20):
    # Create SHAP waterfall drawing
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, feature_names=feature_names, max_display=max_display, show=False)
    st.pyplot(fig)
    plt.close()

def summary(model, data, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)

    # Summarize and visualize SHAP values
    display_shap_summary(shap_values_cat_train, X_train)

def plot_shap(model, data, home_id, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)
    
    # Visualize SHAP values
    plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, home_id, X_test, X_train)

    # Waterfall
    home_index = X_test[X_test['Id'] == home_id].index[0]
    display_shap_waterfall_plot(explainer, explainer.expected_value, shap_values_cat_test[home_index], feature_names=X_test.columns, max_display=20)

st.title("House Price Prediction Project")

def main():
    model = load_model()
    data = load_data()

    X_train = load_x_y("X_train.pkl")
    X_test = load_x_y("X_test.pkl")
    y_train = load_x_y("y_train.pkl")
    y_test = load_x_y("y_test.pkl")

    # Radio buttons for options
    election = st.radio("Make Your Choice:", ("Feature Importance", "Home Specific SHAP"))
    available_customer_ids = X_test['Id'].tolist()
    
    # If User-based SHAP option is selected
    if election == "Home Specific SHAP":
        # Customer ID text input
        home_id = st.selectbox("Choose the Home", available_customer_ids)
        home_index = X_test[X_test['Id'] == home_id].index[0]
        st.write(f'Home {home_id}: Actual value for the Home Price : {np.exp(y_test.iloc[home_index]).round(2)}')
        y_pred = model.predict(X_test)
        st.write(f"Home {home_id}: CatBoost Model's prediction for the Home Price : {np.round(np.exp(y_pred[home_index]))}")
        plot_shap(model, data, home_id, X_train=X_train, X_test=X_test)
    
    
    
    # If Feature Importance is selected
    elif election == "Feature Importance":
        summary(model, data, X_train=X_train, X_test=X_test)

if __name__ == "__main__":
    main()