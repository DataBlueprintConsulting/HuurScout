import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import streamlit as st

def plot_distribution(df, column='huurmaand_woning'):
    """Plot distribution with annotations and KDE, and return the figure."""
    if column not in df.columns:
        raise KeyError(f'The column "{column}" is not in the dataset.')

    data_column = df[column]

    # Create the histogram with KDE
    g = sns.displot(data=data_column, kde=True, stat='count', height=5, aspect=1.5)
    for ax in g.axes.flat:
        for bar in ax.patches:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height, 
                    f'{int(height)}', ha='center', va='bottom', fontsize=10
                )
    # Create the Matplotlib figure to return
    fig = g.fig
    return fig

def plot_histogram_and_normal_probability(df, column='huurmaand_woning'):
    """Plot histogram and normal probability plot, and return the figures."""
    if column not in df.columns:
        raise KeyError(f'The column "{column}" is not in the dataset.')

    data_column = df[column]

    # Histogram with Seaborn
    fig1, ax1 = plt.subplots()
    sns.histplot(data_column, kde=True, ax=ax1)
    ax1.set_title(f'Histogram of {column}')
    ax1.set_xlabel(column)
    ax1.set_ylabel('Frequency')

    # Normal probability plot
    fig2 = plt.figure()
    res = stats.probplot(data_column, plot=plt)
    plt.title(f'Normal Probability Plot of {column}')

    return fig1, fig2

def plot_scatter(df, x_var, y_var='huurmaand_woning'):
    """Plot scatter plot for two variables."""
    if x_var not in df.columns or y_var not in df.columns:
        raise KeyError(f'One or both of the columns "{x_var}" and "{y_var}" are not in the dataset.')

    # Scatter plot
    data = pd.concat([df[y_var], df[x_var]], axis=1)
    fig, ax = plt.subplots()
    data.plot.scatter(x=x_var, y=y_var, ax=ax)
    ax.set_title(f'Scatter Plot of {y_var} vs {x_var}')
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    return fig

import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import streamlit as st

def kmeans_clustering(df):
    """Perform K-Means clustering on plaatsnaam_encoded and huurmaand_woning with interactive hoverable plot."""
    if 'plaatsnaam' not in df.columns or 'huurmaand_woning' not in df.columns:
        raise KeyError('Required columns "plaatsnaam" and "huurmaand_woning" are not in the dataset.')

    # Encode 'plaatsnaam' into numerical values
    label_encoder = LabelEncoder()
    df['plaatsnaam_encoded'] = label_encoder.fit_transform(df['plaatsnaam'])

    # Select features for clustering
    features = df[['plaatsnaam_encoded', 'huurmaand_woning']].dropna()

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    features['cluster'] = kmeans.fit_predict(features)

    # Merge cluster assignments back with original data for hover information
    features['plaatsnaam'] = df['plaatsnaam'].iloc[features.index]

    # Create an interactive plot using Plotly
    fig = px.scatter(
        features,
        x='plaatsnaam_encoded',
        y='huurmaand_woning',
        color='cluster',
        hover_data={'plaatsnaam': True, 'cluster': True, 'huurmaand_woning': True},
        title='K-Means Clustering: Plaatsnaam vs Huurprijs',
        labels={'plaatsnaam_encoded': 'Plaatsnaam (Encoded)', 'huurmaand_woning': 'Huurmaand (€)'},
        color_continuous_scale='Viridis'
    )
    fig.update_traces(marker=dict(size=10))  # Customize marker size
    fig.update_layout(coloraxis_colorbar=dict(title='Cluster'))

    # Display the plot in Streamlit
    st.plotly_chart(fig)

def plaatsnaam_statistics(df):
    """Calculate and plot statistics for plaatsnaam vs huurmaand."""
    if 'plaatsnaam' not in df.columns or 'huurmaand_woning' not in df.columns:
        raise KeyError('Required columns "plaatsnaam" and "huurmaand_woning" are not in the dataset.')

    plaatsnaam_stats = df.groupby('plaatsnaam')['huurmaand_woning'].agg(['mean', 'count'])
    filtered_stats = plaatsnaam_stats[plaatsnaam_stats['count'] >= 10]

    top_10_expensive = filtered_stats['mean'].sort_values(ascending=False).head(10)
    top_10_cheapest = filtered_stats['mean'].sort_values(ascending=True).head(10)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Top 10 Most Expensive Plaatsnamen (Average Rent):")
        st.write(top_10_expensive.round(0))
    
    with col2:
        st.write("\nTop 10 Cheapest Plaatsnamen (Average Rent):")
        st.write(top_10_cheapest.round(0))

    # Plot statistics
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    top_10_expensive.plot(kind='bar', color='red')
    plt.title('Top 10 Most Expensive Plaatsnamen')
    plt.ylabel('Average Huurmaand (€)')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    top_10_cheapest.plot(kind='bar', color='green')
    plt.title('Top 10 Cheapest Plaatsnamen')
    plt.ylabel('Average Huurmaand (€)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

def pairplot_analysis(df, columns):
    """Generate pairplot for specified columns."""
    sns.set()
    fig = sns.pairplot(df[columns], height=2.5).fig
    st.pyplot(fig)

def correlation_analysis(df):
    """Perform correlation analysis on numerical and one-hot encoded data."""
    numerical_df = df.select_dtypes(include=['number'])

    # Correlation matrix for numerical data
    corrmat = numerical_df.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.8, square=True, cmap='coolwarm')
    plt.title('Correlation Matrix - Numerical Data')
    st.pyplot(f)

    # One-hot encoding
    one_hot_labels = pd.get_dummies(df[['energielabel', 'soort_bouw', 'soort_woonhuis', 'soort_dak', 'cv_ketel', 'plaatsnaam']])
    numerical_df_with_labels = pd.concat([numerical_df, one_hot_labels], axis=1)

    # Correlation matrix with one-hot encoding
    corrmat_with_labels = numerical_df_with_labels.corr()
    # f, ax = plt.subplots(figsize=(12, 9))
    # sns.heatmap(corrmat_with_labels, vmax=0.8, square=True, cmap='coolwarm')
    # plt.title('Correlation Matrix with One-Hot Encoding')
    # st.pyplot(f)

    # Correlation with huurmaand_woning
    huurmaand_corr = corrmat_with_labels['huurmaand_woning'].dropna().sort_values(ascending=False)
    col1, col2 = st.columns(2)
    with col1:    
        st.write("Top correlations with 'huurmaand_woning':")
        st.write(huurmaand_corr.head(25))
    with col2:    
        st.write("\nLowest correlations with 'huurmaand_woning':")
        st.write(huurmaand_corr.tail(25))