# import streamlit as st
# import pandas as pd
# from modules.ml.regression import regression_analysis

# st.set_page_config(layout="wide")
# st.title('Huur Data Analysis')

# # Load the data
# file = 'data/rental_data.csv'
# df = pd.read_csv(file)

# # Drop unnecessary columns
# if 'Unnamed: 0' in df.columns:
#     df = df.drop(columns=['Unnamed: 0'])

# # Feature preparation function
# @st.cache_data
# def prepare_features(df):
#     X = df.select_dtypes(include=['number'])
#     one_hot_features = pd.get_dummies(df[['energielabel', 'plaatsnaam']])
#     X = pd.concat([X, one_hot_features], axis=1).drop(columns=['huurmaand_woning', 'huurmaand'], errors='ignore')

#     categorical_features = [col for col in X.columns if col.startswith("energielabel_") or col.startswith("plaatsnaam_")]
#     numeric_features = [col for col in X.columns if col not in categorical_features]

#     default_numeric = X[numeric_features].median().to_dict()  # Median for numeric features
#     default_categorical = {feature_group: df[feature_group].mode()[0] for feature_group in ['energielabel', 'plaatsnaam']}

#     return numeric_features, categorical_features, default_numeric, default_categorical

# # Prepare features
# numeric_features, categorical_features, default_numeric, default_categorical = prepare_features(df)

# # Separate categorical features into energielabel and plaatsnaam
# energielabel_features = [col for col in categorical_features if col.startswith("energielabel_")]
# plaatsnaam_features = [col for col in categorical_features if col.startswith("plaatsnaam_")]

# # Dynamic attribute selection
# st.markdown("## Select Attributes to Input")
# selected_features = st.multiselect(
#     "Choose the attributes you want to provide inputs for:",
#     options=numeric_features + ["energielabel", "plaatsnaam"],
#     default=numeric_features[:] + ["energielabel", "plaatsnaam"],  # Default selection
# )

# # Collect user inputs
# st.markdown("### Input Feature Values")
# input_data = {feature: 0 for feature in numeric_features + categorical_features}  # Initialize with zeros

# # Handle numeric inputs
# for feature in selected_features:
#     if feature in numeric_features:
#         input_data[feature] = st.number_input(f"Enter value for {feature}:", value=default_numeric[feature])

# # Handle categorical inputs
# if "energielabel" in selected_features:
#     st.subheader("Energielabel Selection")
#     energielabel_display_options = [option[len("energielabel_"):] for option in energielabel_features]
#     energielabel_selected = st.selectbox(
#         "Select Energielabel:",
#         energielabel_display_options,
#         index=energielabel_display_options.index(default_categorical["energielabel"]) if default_categorical["energielabel"] in energielabel_display_options else 0,
#     )
#     for option in energielabel_features:
#         input_data[option] = 1 if option == f"energielabel_{energielabel_selected}" else 0

# if "plaatsnaam" in selected_features:
#     st.subheader("Plaatsnaam Selection")
#     plaatsnaam_display_options = [option[len("plaatsnaam_"):] for option in plaatsnaam_features]
#     plaatsnaam_selected = st.selectbox(
#         "Select Plaatsnaam:",
#         plaatsnaam_display_options,
#         index=plaatsnaam_display_options.index(default_categorical["plaatsnaam"]) if default_categorical["plaatsnaam"] in plaatsnaam_display_options else 0,
#     )
#     for option in plaatsnaam_features:
#         input_data[option] = 1 if option == f"plaatsnaam_{plaatsnaam_selected}" else 0

# # Prepare the input data as a DataFrame
# if st.button("Run Prediction"):
#     new_data = pd.DataFrame([input_data])
    
#     # Debugging: Drop unnecessary columns
#     if 'Unnamed: 0' in new_data.columns:
#         new_data = new_data.drop(columns=['Unnamed: 0'], errors='ignore')
#     # During training
#     X = df.drop(columns=["target_column"], errors="ignore")  # Replace with your actual target column
#     trained_features = X.columns.tolist()

#     # Debugging: Check for feature alignment
#     trained_features = [...]  # Replace with the actual list of trained features
#     st.write("Trained features:", trained_features)
#     st.write("new_data Columns:", new_data.columns.tolist())

#     # Assert alignment with trained features
#     try:
#         assert set(new_data.columns) == set(trained_features), "Feature mismatch between new_data and trained features"
#     except AssertionError as e:
#         st.error(str(e))
#         st.stop()  # Stop execution if features don't match

#     # Debugging: One-Hot Encoded Plaatsnaam Columns
#     plaatsnaam_columns = [col for col in new_data.columns if col.startswith("plaatsnaam_")]
#     st.write("One-Hot Encoded Plaatsnaam Columns:", plaatsnaam_columns)

#     # Check selected plaatsnaam
#     selected_plaatsnaam = new_data.loc[:, plaatsnaam_columns].sum(axis=1)
#     st.write("Selected Plaatsnaam (sum of one-hot columns):", selected_plaatsnaam)

#     # Show the prepared new_data
#     st.write("Prepared new_data for prediction:")
#     # st.write(new_data)

#     # Prediction Results
#     st.markdown("### Prediction Results")
#     try:
#         # Call regression_analysis to process the data and make predictions
#         results = regression_analysis(df, new_data)

#         # Display evaluation metrics
#         st.subheader("Model Evaluation Metrics")
#         st.text(f"Mean Absolute Error (MAE): {results['mae']:.2f}")
#         st.text(f"R² Score: {results['r2']:.2f}")

#     except KeyError as e:
#         st.error(f"Key Error: {e}")
#     except Exception as e:
#         st.error(f"An error occurred: {e}")

# # # Prepare the input data as a DataFrame
# # if st.button("Run Prediction"):
# #     new_data = pd.DataFrame([input_data])
# #     new_data = new_data.drop(columns=['Unnamed: 0'], errors='ignore')
# #     trained_features = [...]  # List of features used during training
# #     assert set(new_data.columns) == set(trained_features), "Feature mismatch"
# #     st.text("new_data Columns:", new_data.columns.tolist())
# #     st.text("One-Hot Encoded Plaatsnaam Columns:", [col for col in new_data.columns if col.startswith("plaatsnaam_")])
# #     st.text("Selected Plaatsnaam:", new_data.loc[:, new_data.columns.str.startswith("plaatsnaam_")].sum(axis=1))

# #     st.write(new_data)

# #     st.markdown("### Prediction Results")

# #     try:
# #         # Call regression_analysis to process the data and make predictions
# #         results = regression_analysis(df, new_data)

# #         # Display evaluation metrics
# #         st.subheader("Model Evaluation Metrics")
# #         st.text(f"Mean Absolute Error (MAE): {results['mae']:.2f}")
# #         st.text(f"R² Score: {results['r2']:.2f}")

# #     except KeyError as e:
# #         st.error(f"Key Error: {e}")
# #     except Exception as e:
# #         st.error(f"An error occurred: {e}")

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

    return numeric_features, categorical_features, default_numeric, default_categorical, trained_features, X.columns.tolist()

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
        input_data[feature] = st.number_input(f"Enter value for {feature}:", value=default_numeric[feature])

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

if st.button("Run Prediction"):
    new_data = pd.DataFrame([input_data])
    st.write("Prepared `new_data` for prediction:")
    st.write(new_data)

    plaatsnaam_columns = [col for col in new_data.columns if col.startswith("plaatsnaam_")]
    st.write("Plaatsnaam Encoding:")
    st.write(new_data[plaatsnaam_columns])
    st.write("Trained features:", trained_features)
    st.write("new_data Columns:", new_data.columns.tolist())

    # Ensure alignment with trained features
    trained_features = X.columns.tolist()  # Save this during training
    st.write("Trained Features:", trained_features)
    assert set(new_data.columns) == set(trained_features), "Feature mismatch"

    # Prediction
    results = regression_analysis(df, new_data)
    st.write("Predicted Rent:", results)




# # Prepare the input data as a DataFrame
# if st.button("Run Prediction"):
#     new_data = pd.DataFrame([input_data])

#     # Debugging: Check for feature alignment
#     st.write("Trained features:", trained_features)
#     st.write("new_data Columns:", new_data.columns.tolist())

#     # Assert alignment with trained features
#     try:
#         # assert set(new_data.columns) == set(trained_features), "Feature mismatch between new_data and trained features"
#         assert set(new_data.columns) == set(trained_features), "Feature mismatch"

#     except AssertionError as e:
#         st.error(str(e))
#         st.stop()  # Stop execution if features don't match

#     # Debugging: One-Hot Encoded Plaatsnaam Columns
#     plaatsnaam_columns = [col for col in new_data.columns if col.startswith("plaatsnaam_")]
#     st.write("One-Hot Encoded Plaatsnaam Columns:", plaatsnaam_columns)

#     # Check selected plaatsnaam
#     selected_plaatsnaam = new_data.loc[:, plaatsnaam_columns].sum(axis=1)
#     st.write("Selected Plaatsnaam (sum of one-hot columns):", selected_plaatsnaam)

#     # Show the prepared new_data
#     st.write("Prepared new_data for prediction:")
#     st.write(new_data)

#     # Prediction Results
#     st.markdown("### Prediction Results")
#     try:
#         # Call regression_analysis to process the data and make predictions
#         results = regression_analysis(df, new_data)

#         # Display evaluation metrics
#         st.subheader("Model Evaluation Metrics")
#         st.text(f"Mean Absolute Error (MAE): {results['mae']:.2f}")
#         st.text(f"R² Score: {results['r2']:.2f}")

#     except KeyError as e:
#         st.error(f"Key Error: {e}")
#     except Exception as e:
#         st.error(f"An error occurred: {e}")