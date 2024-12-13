import streamlit as st
import pandas as pd
from modules.ml.regression import regression_analysis

st.set_page_config(
    page_title="Home 🏠", 
    layout="wide", 
    page_icon='favicon.ico')

# Google Analytics Script
GA_SCRIPT = """
<script async src="https://www.googletagmanager.com/gtag/js?id=G-HJB59JJYS5"></script>
<script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());

    gtag('config', 'G-HJB59JJYS5');
</script>
"""

# Embed Google Analytics Script
st.markdown(GA_SCRIPT, unsafe_allow_html=True)

st.title('Huur Data Analysis 🏠')
st.logo('logo.png')
st.html("""
    <style>
    [alt=Logo] {
        height: 2,5rem;
    }
    </style>
""")

# Add an introduction
st.markdown("""
Welcome to **HuurScout**! This application is designed to help you analyze rental property data and predict fair rental prices based on key property attributes. 

### What You Can Do Here:
- Input details about a property, such as its size, location, and energy label.
- Predict the expected rental price using advanced machine learning models.
- Evaluate rental deals with **HoeHardWordIkGenaaid-meter**.
- Explore property data trends with interactive visualizations.
""")

# Load the data
file = 'data/rental_data.csv'
df = pd.read_csv(file)

# Drop multiple columns if they exist in the DataFrame
columns_to_drop = [
    'link', 'Unnamed: 0', 'web-scraper-order', 'web-scraper-start-url', 
    'link-href', 'specifiek', 'aangeboden_sinds', 'huurmaand', 
    'opp_gebouwgebonden_buitenruimte', 'opp_externe_bergruimte'
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

@st.cache_data
def prepare_features(df):
    X = df.select_dtypes(include=['number'])
    one_hot_features = pd.get_dummies(df[['energielabel', 'plaatsnaam']])
    X = pd.concat([X, one_hot_features], axis=1).drop(columns=['huurmaand_woning', 'huurmaand'], errors='ignore')
    trained_features = X.columns.tolist()  # Save these during training
    categorical_features = [col for col in X.columns if col.startswith("energielabel_") or col.startswith("plaatsnaam_")]
    numeric_features = [col for col in X.columns if col not in categorical_features]
    default_numeric = X[numeric_features].median().to_dict()  # Median for numeric features
    default_categorical = {
        feature_group: df[feature_group].mode()[0] for feature_group in ['energielabel', 'plaatsnaam']
    }
    return numeric_features, categorical_features, default_numeric, default_categorical, trained_features

# Prepare features
numeric_features, categorical_features, default_numeric, default_categorical, trained_features = prepare_features(df)

# Separate categorical features into energielabel and plaatsnaam
energielabel_features = [col for col in categorical_features if col.startswith("energielabel_")]
plaatsnaam_features = [col for col in categorical_features if col.startswith("plaatsnaam_")]

# Add a selectbox for the type of woning
woning_type = st.selectbox(
    "Select the Type of Woning:",
    ["Grondgebonden woning 🏠", "Stapelwoning 🏢"]
)

# Define which attributes are relevant for each woning type
if woning_type == "Grondgebonden woning 🏠":
    relevant_numeric_features = [f for f in numeric_features if f in [
        'oppervlakte_wonen', 'slaapkamers', 'aantal_kamers', 'bouwjaar', 
        'inhoud', 'opp_perceel', 'energielabel', 
        'aantal_badkamers', 'aantal_woonlagen', 
        'oppervlakte_kadaster', 'aantal_toiletten']]
    
else:  # "Stapelwoning"
    relevant_numeric_features = [f for f in numeric_features if f in [
        'oppervlakte_wonen', 'slaapkamers', 'aantal_kamers', 'bouwjaar',
        'inhoud', 'aantal_woonlagen', 'energielabel', 'aantal_badkamers', 'aantal_woonlagen']]

relevant_features = relevant_numeric_features + ["energielabel", "plaatsnaam"]

# Dynamic attribute selection
st.markdown("## Select Attributes to Input")
selected_features = st.multiselect(
    "Choose the attributes you want to provide inputs for:",
    options=relevant_features,
    default=relevant_features,
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
                value=int(default_numeric.get(feature, 0)),  # get default or 0 if not found
                step=1,
            )
        )

# Handle categorical inputs
if "energielabel" in selected_features:
    st.subheader("Energielabel Selection")
    energielabel_display_options = [option[len("energielabel_"):] for option in energielabel_features]
    default_label = default_categorical["energielabel"]
    if default_label not in energielabel_display_options:
        default_label = energielabel_display_options[0]
    energielabel_selected = st.selectbox("Select Energielabel:", energielabel_display_options, index=energielabel_display_options.index(default_label))
    for option in energielabel_features:
        input_data[option] = 1 if option == f"energielabel_{energielabel_selected}" else 0

if "plaatsnaam" in selected_features:
    st.subheader("Plaatsnaam Selection")
    plaatsnaam_display_options = [option[len("plaatsnaam_"):] for option in plaatsnaam_features]
    default_place = default_categorical["plaatsnaam"]
    if default_place not in plaatsnaam_display_options:
        default_place = plaatsnaam_display_options[0]
    plaatsnaam_selected = st.selectbox("Select Plaatsnaam:", plaatsnaam_display_options, index=plaatsnaam_display_options.index(default_place))
    for option in plaatsnaam_features:
        input_data[option] = 1 if option == f"plaatsnaam_{plaatsnaam_selected}" else 0

actual_rent = int(
    st.number_input(
        "Actual Rent (Huurmaand Woning):",
        value=1200,
        step=1,
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

# Footer
st.markdown(
    """
    ---
    <div style="text-align: center;">
        <small>
            📌 Made by Adam Asbai Halifa | 
            <a href="https://datablueprintconsulting.nl" target="_blank">Data Blueprint Consulting</a> | 
            <a href="https://www.linkedin.com/in/adam-asbai-halifa" target="_blank">LinkedIn</a> | 
            <a href="mailto:a.asbaihalifa@dbp-c.nl">Email</a>
        </small>
    </div>
    """,
    unsafe_allow_html=True,
)
