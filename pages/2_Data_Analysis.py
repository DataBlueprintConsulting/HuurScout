import streamlit as st
import pandas as pd
from modules.data_analysis.data_analysis import (
    kmeans_clustering,
    plaatsnaam_statistics,
    pairplot_analysis,
    correlation_analysis
)

st.set_page_config(page_title="Data Analysis 📊", layout="wide", page_icon='favicon.ico')

st.title('Huur Data Analysis 📊')
st.logo('logo.png')
st.html("""
    <style>
    [alt=Logo] {
        height: 2,5rem;
    }
    </style>
""")    

# Load the data
file = 'data/rental_data.csv'
df = pd.read_csv(file)

# Drop multiple columns if they exist in the DataFrame
columns_to_drop = ['link', 'Unnamed: 0', 'web-scraper-order', 'web-scraper-start-url', 'link-href', 'specifiek', 'aangeboden_sinds', 'huurmaand', 'opp_gebouwgebonden_buitenruimte', 'opp_externe_bergruimte']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Initialize `selected_places` with default value
if "selected_places" not in st.session_state:
    st.session_state["selected_places"] = ["All Locations"]

# Reset button
if st.button("Reset Filters"):
    st.session_state["selected_places"] = ["All Locations"]

# Multi-select dropdown for filtering
selected_places = st.multiselect(
    "Select one or more locations:",
    options=["All Locations"] + list(df['plaatsnaam'].unique()),
    default=st.session_state["selected_places"],  # Use session state to preserve/reset
    help="Search or select locations to filter the dataset. Choose 'All Locations' to include everything."
)

# Update session state with the user's selection
st.session_state["selected_places"] = selected_places

# Apply filter logic
if "All Locations" in selected_places:
    filtered_df = df
else:
    filtered_df = df[df['plaatsnaam'].isin(selected_places)]

# Display filtered data
st.write(f"Filtered Dataset: {filtered_df.shape[0]} rows")
st.dataframe(filtered_df, hide_index=True)

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    # Dynamic Key Metrics
    st.subheader('🔑 Key Metrics')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### Huurmaand Metrics")
        st.write(f"**Min:** {int(filtered_df['huurmaand_woning'].min())} €")
        st.write(f"**Mean:** {int(filtered_df['huurmaand_woning'].mean().round(2))} €")
        st.write(f"**Median:** {int(filtered_df['huurmaand_woning'].median())} €")
        st.write(f"**Mode:** {int(filtered_df['huurmaand_woning'].mode()[0].round(2))} €")
        st.write(f"**Max:** {int(filtered_df['huurmaand_woning'].max())} €")

    with col2:
        st.markdown("### Bouwjaar Metrics")
        st.write(f"**Min:** {int(filtered_df['bouwjaar'].min())}")
        st.write(f"**Mean:** {int(filtered_df['bouwjaar'].mean())}")
        st.write(f"**Median:** {int(filtered_df['bouwjaar'].median())}")
        st.write(f"**Mode:** {int(filtered_df['bouwjaar'].mode()[0])}")
        st.write(f"**Max:** {int(filtered_df['bouwjaar'].max())}")

    with col3:
        st.markdown("### Inhoud Metrics")
        st.write(f"**Min:** {int(filtered_df['inhoud'].min())} m³")
        st.write(f"**Mean:** {int(filtered_df['inhoud'].mean().round(2))} m³")
        st.write(f"**Median:** {int(filtered_df['inhoud'].median())} m³")
        st.write(f"**Mode:** {int(filtered_df['inhoud'].mode()[0])} m³")
        st.write(f"**Max:** {int(filtered_df['inhoud'].max())} m³")

    with col4:
        st.markdown("### Opp Wonen Metrics")
        cleaned_oppervlakte_wonen = filtered_df['oppervlakte_wonen'].dropna()
        st.write(f"**Min:** {int(cleaned_oppervlakte_wonen.min())} m²")
        st.write(f"**Mean:** {int(cleaned_oppervlakte_wonen.mean().round(2))} m²")
        st.write(f"**Median:** {int(cleaned_oppervlakte_wonen.median())} m²")
        st.write(f"**Mode:** {int(cleaned_oppervlakte_wonen.mode()[0])} m²")
        st.write(f"**Max:** {int(cleaned_oppervlakte_wonen.max())} m²")

    st.markdown("""
    ### About the Metrics
    The above metrics provide an overview of the key attributes of rental properties in the dataset:

    - **Huurmaand Woning Metrics**: A summary of rental prices (in Euros) per month, giving an idea of the cost range, average, and typical pricing patterns.
    - **Bouwjaar Metrics**: Insights into the construction years of properties, highlighting the age distribution and trends over time.
    - **Inhoud Metrics**: Details about the volume of properties (in cubic meters), useful for understanding the spaciousness of homes.
    - **Oppervlakte Wonen Metrics**: Information on the living area (in square meters), helping to assess property sizes and compare different offerings.
    """)

    st.markdown("## Advanced Analytics")

    # K-Means Clustering
    st.subheader('🔗 K-Means Clustering: Plaatsnaam vs Huurprijs')
    try:
        kmeans_clustering(df)
    except KeyError as e:
        st.text(str(e))

    st.markdown("""
    #### About K-Means Clustering
    K-Means clustering is an unsupervised machine learning algorithm used to identify patterns or groupings within the data. 
    - **Purpose**: In this analysis, we cluster rental prices (`Huurprijs`) based on locations (`Plaatsnaam`) to uncover trends and groupings within the dataset. 
    - **Benefits**: This can help identify areas with similar rental price ranges, detect anomalies, and provide insights for strategic decision-making.
    - **Insights**: The resulting clusters highlight patterns in the relationship between location and rental pricing, enabling a better understanding of the market segmentation.
    
    #### Insights from Cluster Analysis
    - `Cluster0` == **Expensive Clusters**: High average rental prices, likely representing premium areas.
    - `Cluster2` == **Medium Clusters**: Moderate average rental prices, representing balanced affordability.
    - `Cluster1` == **Cheap Clusters**: Low average rental prices, suitable for budget-friendly considerations.
    """)
    
    # Plaatsnaam Statistics
    st.subheader('📍 Plaatsnaam Statistics')
    try:
        plaatsnaam_statistics(df)
    except KeyError as e:
        st.text(str(e))

    st.markdown("""
    ### About Plaatsnaam Statistics
    This section provides a detailed statistical breakdown of rental data based on different locations (`Plaatsnaam`). 

    - **Purpose**: To explore how rental prices vary across different cities or towns. This analysis helps in identifying patterns and trends specific to each location.
    - **Benefits**: Provides a comparative view of locations, enabling users to make data-driven decisions, whether it's for choosing an affordable area or understanding premium locations.
    - **Insights**: Users can uncover average rental costs, common trends, and outliers within specific areas, offering valuable information for renters, property investors, and analysts.
    """)

    # Correlation Analysis
    st.subheader('📈 Correlation Analysis')
    try:
        correlation_analysis(df)
    except KeyError as e:
        st.text(str(e))
    st.markdown(
        """
        ### About the correlation matrix
        The **correlation analysis** examines how numerical features in the dataset relate to each other.
        This helps understand which variables have a strong positive or negative relationship, which can 
        be useful in predictive modeling and feature engineering.
        """
    )

    # Pairplot Analysis
    st.subheader('📊 Pairplot Analysis')
    columns_to_pairplot = ['oppervlakte_wonen', 'slaapkamers', 'huurmaand_woning', 'bouwjaar', 'aantal_kamers']
    try:
        pairplot_analysis(df, columns_to_pairplot)
    except KeyError as e:
        st.text(str(e))
    st.markdown(
        """
        ### About the pair plot analysis
        The **pairplot analysis** visualizes the relationships between multiple numerical features in the dataset. 
        It helps identify trends, correlations, or potential outliers that may exist between these variables. 
        The selected features include:
        - `oppervlakte_wonen`: Living area in square meters
        - `slaapkamers`: Number of bedrooms
        - `huurmaand_woning`: Monthly rent of the property
        - `bouwjaar`: Year of construction
        - `aantal_kamers`: Total number of rooms
        """
    )
    col1, col2, col3 = st.columns(3)

    columns_to_analyze = ["bouwjaar", "inhoud", "oppervlakte_wonen"]

    # Assign one dataframe column per visualization column
    columns_mapping = zip(columns_to_analyze, [col1, col2, col3])

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
