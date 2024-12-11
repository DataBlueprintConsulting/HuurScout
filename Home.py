import streamlit as st
import pandas as pd
from modules.ml.regression import regression_analysis

st.set_page_config(layout="wide")
st.title('Huur Data Analysis')

# Load the data
file = 'data/rental_data.csv'
df = pd.read_csv(file)

# Drop unnecessary columns
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Feature preparation function
@st.cache_data
def prepare_features(df):
    X = df.select_dtypes(include=['number'])
    one_hot_features = pd.get_dummies(df[['energielabel', 'plaatsnaam']])
    X = pd.concat([X, one_hot_features], axis=1).drop(columns=['huurmaand_woning', 'huurmaand'], errors='ignore')
    trained_features = X.columns.tolist()  # Save these during training
    categorical_features = [col for col in X.columns if col.startswith("energielabel_") or col.startswith("plaatsnaam_")]
    numeric_features = [col for col in X.columns if col not in categorical_features]
    default_numeric = X[numeric_features].median().to_dict()  # Median for numeric features
    default_categorical = {feature_group: df[feature_group].mode()[0] for feature_group in ['energielabel', 'plaatsnaam']}
    return numeric_features, categorical_features, default_numeric, default_categorical, trained_features

# Prepare features
numeric_features, categorical_features, default_numeric, default_categorical, trained_features = prepare_features(df)

# Separate categorical features into energielabel and plaatsnaam
energielabel_features = [col for col in categorical_features if col.startswith("energielabel_")]
plaatsnaam_features = [col for col in categorical_features if col.startswith("plaatsnaam_")]

# Dynamic attribute selection
st.markdown("## Select Attributes to Input")
selected_features = st.multiselect(
    "Choose the attributes you want to provide inputs for:",
    options=numeric_features + ["energielabel", "plaatsnaam"],
    default=numeric_features[:] + ["energielabel", "plaatsnaam"],  # Default selection
)

# Collect user inputs
st.markdown("### Input Feature Values")
input_data = {feature: 0 for feature in numeric_features + categorical_features}  # Initialize with zeros

# Handle numeric inputs
for feature in selected_features:
    if feature in numeric_features:
        input_data[feature] = int(
            st.number_input(
                f"Enter value for {feature}:",
                value=int(default_numeric[feature]),  # Convert default to integer
                step=1,  # Increment by 1
            )
        )
# Handle categorical inputs
if "energielabel" in selected_features:
    st.subheader("Energielabel Selection")
    energielabel_display_options = [option[len("energielabel_"):] for option in energielabel_features]
    energielabel_selected = st.selectbox(
        "Select Energielabel:",
        energielabel_display_options,
        index=energielabel_display_options.index(default_categorical["energielabel"]) if default_categorical["energielabel"] in energielabel_display_options else 0,
    )
    for option in energielabel_features:
        input_data[option] = 1 if option == f"energielabel_{energielabel_selected}" else 0

if "plaatsnaam" in selected_features:
    st.subheader("Plaatsnaam Selection")
    plaatsnaam_display_options = [option[len("plaatsnaam_"):] for option in plaatsnaam_features]
    plaatsnaam_selected = st.selectbox(
        "Select Plaatsnaam:",
        plaatsnaam_display_options,
        index=plaatsnaam_display_options.index(default_categorical["plaatsnaam"]) if default_categorical["plaatsnaam"] in plaatsnaam_display_options else 0,
    )
    for option in plaatsnaam_features:
        input_data[option] = 1 if option == f"plaatsnaam_{plaatsnaam_selected}" else 0

actual_rent = int(
    st.number_input(
        "Actual Rent (Huurmaand Woning):",
        value=1200,  # Default to an integer
        step=1,  # Increment by 1
    )
)

if st.button("Run Prediction"):
    new_data = pd.DataFrame([input_data])
    # Align new_data with trained_features
    new_data = new_data.reindex(columns=trained_features, fill_value=0)
    # Ensure alignment
    assert set(new_data.columns) == set(trained_features), "Feature mismatch"
    # Prediction
    results = regression_analysis(df, new_data, actual_rent)
    # Display results
    st.write("Prediction Results:")
    st.write(f"Predicted Rent: €{results['predicted_rent']:,.2f}")
    st.write(f"Mean Absolute Error (MAE): {results['mae']:.2f}")
    st.write(f"R² Score: {results['r2']:.2f}")