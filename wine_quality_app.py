import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration setup
st.set_page_config(
    page_title="Vinho Verde Quality Predictor",
    layout="wide"
)

# Constants for model paths
RF_MODEL_PATH = 'random_forest_wine_model.pkl'
XGB_MODEL_PATH = 'xgboost_wine_model.pkl'
SCALER_PATH = 'wine_scaler.pkl'

@st.cache_resource
def load_artifacts():
    """
    Load machine learning artifacts with error handling.
    Using cache_resource to prevent reloading on every interaction.
    """
    try:
        if not os.path.exists(RF_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {RF_MODEL_PATH}")
            
        rf = joblib.load(RF_MODEL_PATH)
        xgb = joblib.load(XGB_MODEL_PATH)
        scl = joblib.load(SCALER_PATH)
        return rf, xgb, scl
    except Exception as e:
        st.error(f"Critical Error: Failed to load model artifacts. {str(e)}")
        return None, None, None

def get_user_input():
    """
    Collect and format user input features from the sidebar.
    """
    st.sidebar.header("Input Parameters")
    
    # Quantitative features
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 3.0, 16.0, 7.0)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.08, 1.6, 0.34)
    citric_acid = st.sidebar.slider('Citric Acid', 0.0, 1.7, 0.32)
    residual_sugar = st.sidebar.slider('Residual Sugar', 0.6, 66.0, 5.4)
    chlorides = st.sidebar.slider('Chlorides', 0.009, 0.61, 0.056)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 1, 290, 30)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', 6, 440, 116)
    density = st.sidebar.slider('Density', 0.987, 1.039, 0.995)
    pH = st.sidebar.slider('pH', 2.72, 4.01, 3.22)
    sulphates = st.sidebar.slider('Sulphates', 0.22, 2.0, 0.53)
    alcohol = st.sidebar.slider('Alcohol', 8.0, 14.9, 10.5)
    
    # Categorical features
    wine_type = st.sidebar.selectbox('Wine Type', ['Red', 'White'])
    type_white_encoded = 1 if wine_type == 'White' else 0
    
    features = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol],
        'type_white': [type_white_encoded]
    })
    
    return features

def main():
    st.title("Vinho Verde Quality Predictor")
    st.markdown("This system predicts the quality score of Portuguese Vinho Verde wine based on physicochemical properties.")
    
    # Load models
    rf_model, xgb_model, scaler = load_artifacts()
    
    # Stop execution if models failed to load
    if rf_model is None:
        st.warning("Application halted due to missing dependencies.")
        return

    # Get input
    input_df = get_user_input()
    
    # Display current input for verification
    st.subheader("Current Input Configuration")
    st.dataframe(input_df)

    # Transform input
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Scaling Error: Input data format mismatch. {str(e)}")
        return

    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Random Forest Prediction"):
            prediction = rf_model.predict(input_scaled)[0]
            st.success(f"Random Forest Quality Score: {prediction:.2f}")

    with col2:
        if st.button("Generate XGBoost Prediction"):
            prediction = xgb_model.predict(input_scaled)[0]
            st.info(f"XGBoost Quality Score: {prediction:.2f}")

if __name__ == "__main__":
    main()
