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

# Drop the column 'Unnamed: 0' if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

st.subheader('🔑 Key metrics')
col1, col2, col3, col4 = st.columns(4)

with col1:
    # Display metrics for `huurmaand_woning`
    st.markdown("### Huurmaand Metrics")
    st.write(f"**Min:** {min(df['huurmaand_woning'])} €")
    st.write(f"**Mean:** {df['huurmaand_woning'].mean().round(2)} €")
    st.write(f"**Median:** {df['huurmaand_woning'].median()} €")
    st.write(f"**Mode:** {df['huurmaand_woning'].mode()[0].round(2)} €")
    st.write(f"**Max:** {max(df['huurmaand_woning'])} €")

with col2:
    # Display metrics for `bouwjaar`
    st.markdown("### Bouwjaar Metrics")
    st.write(f"**Min:** {min(df['bouwjaar'])}")
    st.write(f"**Mean:** {df['bouwjaar'].mean().round(2)}")
    st.write(f"**Median:** {df['bouwjaar'].median()}")
    st.write(f"**Mode:** {df['bouwjaar'].mode()[0]}")
    st.write(f"**Max:** {max(df['bouwjaar'])}")

with col3:
    # Display metrics for `inhoud`
    st.markdown("### Inhoud Metrics")
    st.write(f"**Min:** {min(df['inhoud'])} m³")
    st.write(f"**Mean:** {df['inhoud'].mean().round(2)} m³")
    st.write(f"**Median:** {df['inhoud'].median()} m³")
    st.write(f"**Mode:** {df['inhoud'].mode()[0]} m³")
    st.write(f"**Max:** {max(df['inhoud'])} m³")

with col4:
    # Display metrics for `oppervlakte_wonen`
    st.markdown("### Opp Wonen Metrics")
    st.write(f"**Min:** {min(df['oppervlakte_wonen'])} m²")
    st.write(f"**Mean:** {df['oppervlakte_wonen'].mean().round(2)} m²")
    st.write(f"**Median:** {df['oppervlakte_wonen'].median()} m²")
    st.write(f"**Mode:** {df['oppervlakte_wonen'].mode()[0]} m²")
    st.write(f"**Max:** {max(df['oppervlakte_wonen'])} m²")

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
### About K-Means Clustering
K-Means clustering is an unsupervised machine learning algorithm used to identify patterns or groupings within the data. 

- **Purpose**: In this analysis, we cluster rental prices (`Huurprijs`) based on locations (`Plaatsnaam`) to uncover trends and groupings within the dataset. 
- **Benefits**: This can help identify areas with similar rental price ranges, detect anomalies, and provide insights for strategic decision-making.
- **Insights**: The resulting clusters highlight patterns in the relationship between location and rental pricing, enabling a better understanding of the market segmentation.
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
