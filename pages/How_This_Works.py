import streamlit as st
import pandas as pd
from modules.data_analysis.data_analysis import (
    kmeans_clustering,
    plaatsnaam_statistics,
    pairplot_analysis,
    correlation_analysis,
    plot_distribution, 
    plot_histogram_and_normal_probability,
    plot_scatter
)
from modules.ml.regression import regression_analysis

st.set_page_config(layout="wide")

st.title('Huur Data Analysis')

@st.cache_data
def load_data():
    file = 'data/rental_data.csv'
    df = pd.read_csv(file)
    columns_to_drop = [col for col in ['Unnamed: 0', 'huurmaand'] if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    return df

df = load_data()

# Drop the column 'Unnamed: 0' if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Create three columns for visualizations
col1, col2, col3 = st.columns(3)

columns_to_analyze = ["bouwjaar", "inhoud", "oppervlakte_wonen"]


# Assign one dataframe column per visualization column
columns_mapping = zip(columns_to_analyze, [col1, col2, col3])

# Generate visualizations
for col_name, col in columns_mapping:
    with col:
        st.subheader(f"Analysis for {col_name}")
        try:
            # Distribution and KDE plot
            dist_fig = plot_distribution(df, column=col_name)
            st.pyplot(dist_fig)
        except KeyError as e:
            st.text(str(e))

        try:
            # Histogram and normal probability plot
            hist_fig, prob_fig = plot_histogram_and_normal_probability(df, column=col_name)
            st.pyplot(hist_fig)
            st.pyplot(prob_fig)
        except KeyError as e:
            st.text(str(e))

        try:
            # Scatter plot
            scatter_fig = plot_scatter(df, x_var=col_name)
            st.pyplot(scatter_fig)
        except KeyError as e:
            st.text(str(e))

# K-Means Clustering
st.subheader('K-Means Clustering: Plaatsnaam vs Huurprijs')
try:
    kmeans_clustering(df)
except KeyError as e:
    st.text(str(e))

# Plaatsnaam Statistics
st.subheader('Plaatsnaam Statistics')
try:
    plaatsnaam_statistics(df)
except KeyError as e:
    st.text(str(e))

# Pairplot Analysis
st.subheader('Pairplot Analysis')
columns_to_pairplot = ['oppervlakte_wonen', 'slaapkamers', 'huurmaand_woning', 'bouwjaar', 'aantal_kamers']
try:
    pairplot_analysis(df, columns_to_pairplot)
except KeyError as e:
    st.text(str(e))

# Correlation Analysis
st.subheader('Correlation Analysis')
try:
    correlation_analysis(df)
except KeyError as e:
    st.text(str(e))

@st.cache_data
def prepare_features(df):
    X = df.select_dtypes(include=['number'])
    one_hot_features = pd.get_dummies(df[['energielabel', 'plaatsnaam']])
    X = pd.concat([X, one_hot_features], axis=1).drop(columns=['huurmaand_woning', 'huurmaand'], errors='ignore')
    categorical_features = [col for col in X.columns if col.startswith("energielabel_") or col.startswith("plaatsnaam_")]
    numeric_features = [col for col in X.columns if col not in categorical_features]
    default_numeric = X[numeric_features].median().to_dict()  # Median for numeric features
    default_categorical = {feature_group: df[feature_group].mode()[0] for feature_group in ['energielabel', 'plaatsnaam']}
    return numeric_features, categorical_features, default_numeric, default_categorical

numeric_features, categorical_features, default_numeric, default_categorical = prepare_features(df)

# Collect user inputs
st.markdown("### Input Feature Values")
input_data = {}

# Numeric inputs
for col in numeric_features:
    input_data[col] = st.number_input(
        f"Enter value for {col}:",
        value=default_numeric[col],
        key=col
    )

# Dropdown for categorical features
for feature_group in ['energielabel', 'plaatsnaam']:
    options = [col for col in categorical_features if col.startswith(feature_group + "_")]
    display_options = [option[len(feature_group) + 1:] for option in options]
    default_value = default_categorical[feature_group]
    selected_display_option = st.selectbox(
        f"Select {feature_group.capitalize()}:",
        display_options,
        index=display_options.index(default_value) if default_value in display_options else 0,
        key=feature_group
    )
    selected_option = options[display_options.index(selected_display_option)]
    for option in options:
        input_data[option] = 1 if option == selected_option else 0

new_data = pd.DataFrame([input_data])

# Add a button to trigger the regression analysis
if st.button("Run Prediction"):
    st.markdown("### Prediction Results")
    try:
        results = regression_analysis(df, new_data)
        st.subheader("Model Evaluation Metrics")
        st.text(f"Mean Absolute Error (MAE): {results['mae']:.2f}")
        st.text(f"R² Score: {results['r2']:.2f}")
    except KeyError as e:
        st.error(f"Key Error: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Extract features
@st.cache_data
def prepare_features(df):
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    return numeric_features, categorical_features

numeric_features, categorical_features = prepare_features(df)

# UI for feature selection
st.markdown("## Select Attributes to Input")
selected_features = st.multiselect(
    "Choose the attributes you want to provide inputs for:",
    options=numeric_features + categorical_features
)

# Create input fields for selected features
st.markdown("### Provide Values for Selected Attributes")
input_data = {feature: None for feature in numeric_features + categorical_features}

for feature in selected_features:
    if feature in numeric_features:
        input_data[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)
    elif feature in categorical_features:
        unique_values = df[feature].dropna().unique().tolist()
        input_data[feature] = st.selectbox(f"Select value for {feature}:", options=unique_values)

# Prepare the input data as a DataFrame
if st.button("Run Prediction"):
    new_data = pd.DataFrame([input_data])
    st.markdown("### Prediction Results")

    try:
        # Call regression_analysis to process the data and make predictions
        results = regression_analysis(df, new_data)

        # Display evaluation metrics
        st.subheader("Model Evaluation Metrics")
        st.text(f"Mean Absolute Error (MAE): {results['mae']:.2f}")
        st.text(f"R² Score: {results['r2']:.2f}")

    except KeyError as e:
        st.error(f"Key Error: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")