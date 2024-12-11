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

# Load the data
file = 'data/rental_data.csv'
df = pd.read_csv(file)

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