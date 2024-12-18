import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the saved model, scaler, and feature names
linear_model = joblib.load('linear_regression_hp.pkl')
scaler = joblib.load('scaler.pkl')

# Load feature names to maintain the correct order
with open('feature_names.pkl', 'rb') as f:
    feature_names = joblib.load(f)

# Define the second row (input data) from the dataset
input_data = {
    'MSSubClass': 20,
    'MSZoning': 'RH',
    'LotFrontage': 80,
    'LotArea': 11622,
    'Street': 'Pave',
    'Alley': 'NA',
    'LotShape': 'Reg',
    'LandContour': 'Lvl',
    'Utilities': 'AllPub',
    'LotConfig': 'Inside',
    'LandSlope': 'Gtl',
    'Neighborhood': 'NAmes',
    'Condition1': 'Feedr',
    'Condition2': 'Norm',
    'BldgType': '1Fam',
    'HouseStyle': '1Story',
    'OverallQual': 5,
    'OverallCond': 6,
    'YearBuilt': 1961,
    'YearRemodAdd': 1961,
    'RoofStyle': 'Gable',
    'RoofMatl': 'CompShg',
    'Exterior1st': 'VinylSd',
    'Exterior2nd': 'VinylSd',
    'MasVnrType': 'None',
    'MasVnrArea': 0,
    'ExterQual': 'TA',
    'ExterCond': 'TA',
    'Foundation': 'CBlock',
    'BsmtQual': 'TA',
    'BsmtCond': 'TA',
    'BsmtExposure': 'No',
    'BsmtFinType1': 'Rec',
    'BsmtFinSF1': 468,
    'BsmtFinType2': 'LwQ',
    'BsmtFinSF2': 144,
    'BsmtUnfSF': 270,
    'TotalBsmtSF': 882,
    'Heating': 'GasA',
    'HeatingQC': 'TA',
    'CentralAir': 'Y',
    'Electrical': 'SBrkr',
    '1stFlrSF': 896,
    '2ndFlrSF': 0,
    'LowQualFinSF': 0,
    'GrLivArea': 896,
    'BsmtFullBath': 0,
    'BsmtHalfBath': 0,
    'FullBath': 1,
    'HalfBath': 0,
    'BedroomAbvGr': 2,
    'KitchenAbvGr': 1,
    'KitchenQual': 'TA',
    'TotRmsAbvGrd': 5,
    'Functional': 'Typ',
    'Fireplaces': 0,
    'FireplaceQu': 'NA',
    'GarageType': 'Attchd',
    'GarageYrBlt': 1961,
    'GarageFinish': 'Unf',
    'GarageCars': 1,
    'GarageArea': 730,
    'GarageQual': 'TA',
    'GarageCond': 'TA',
    'PavedDrive': 'Y',
    'WoodDeckSF': 140,
    'OpenPorchSF': 0,
    'EnclosedPorch': 0,
    '3SsnPorch': 0,
    'ScreenPorch': 120,
    'PoolArea': 0,
    'PoolQC': 'NA',
    'Fence': 'MnPrv',
    'MiscFeature': 'NA',
    'MiscVal': 0,
    'MoSold': 6,
    'YrSold': 2010,
    'SaleType': 'WD',
    'SaleCondition': 'Normal'
}

# Convert the dictionary to a pandas DataFrame
user_input_df = pd.DataFrame([input_data])

# Streamlit app title
st.title("House Price Prediction App")

st.subheader("Input Feature Values")
st.write("Please provide values for the following features:")

# Create an empty dictionary to store user inputs
user_inputs = {}

# Dynamically generate input fields for each feature
test_data = pd.read_csv('test.csv').drop(columns=['Id'])  # Used to determine types (numeric or categorical)
num_cols = test_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = test_data.select_dtypes(include=['object']).columns.tolist()

# Initialize input fields with the given dataset values
for feature in feature_names:
    if feature in num_cols:  # Numeric feature
        # Use value from input data or default to 0.0
        user_input = st.number_input(f"{feature} (numeric)", value=input_data.get(feature, 0.0))
        user_inputs[feature] = user_input
    elif feature in cat_cols:  # Categorical feature
        # Get unique values from the test data for the categorical feature
        unique_values = test_data[feature].dropna().unique().tolist()
        
        # Handle 'NA' or missing data by using a placeholder for missing categories
        input_value = input_data.get(feature, 'Missing')  # Use 'Missing' as a placeholder for missing values
        
        # Ensure the placeholder ('Missing') is in the list of unique values if necessary
        if input_value not in unique_values:
            unique_values.append(input_value)
        
        # Add a default value at the top (e.g., "Select a value" for the user to choose)
        unique_values.insert(0, "Select a value")  # Adds a "Select a value" option as the first choice
        
        # Ensure the user input is selected properly, index the placeholder if the input_value is not found
        user_input = st.selectbox(f"{feature} (categorical)", unique_values, index=unique_values.index(input_value))
        user_inputs[feature] = user_input


# Convert the dictionary of inputs to a DataFrame
user_input_df = pd.DataFrame([user_inputs])

# Impute missing values (consistent with training phase)
st.write("Imputing missing values...")
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Ensure imputers work on single columns, not the entire DataFrame
for col in num_cols:
    if col in user_input_df.columns:
        user_input_df[[col]] = num_imputer.fit_transform(user_input_df[[col]])

for col in cat_cols:
    if col in user_input_df.columns:
        user_input_df[[col]] = cat_imputer.fit_transform(user_input_df[[col]])

# Encode categorical variables (consistent with training phase)
st.write("Encoding categorical features...")
for col in cat_cols:
    if col in user_input_df.columns:
        le = LabelEncoder()
        user_input_df[col] = le.fit_transform(user_input_df[col].astype(str))

# Ensure user_input_df columns match the order in feature_names
user_input_df = user_input_df[feature_names]

# Scale the features using the pre-trained scaler
st.write("Scaling features...")
user_input_scaled = scaler.transform(user_input_df)

# Predict the house price
if st.button("Predict"):
    st.write("Predicting house price...")
    predicted_price = linear_model.predict(user_input_scaled)
    st.success(f"Predicted Price: ${predicted_price[0]:,.2f}")
