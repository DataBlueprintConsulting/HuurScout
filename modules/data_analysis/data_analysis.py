import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import plotly.express as px

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

def kmeans_clustering(df):
    """Perform K-Means clustering on plaatsnaam_encoded and huurmaand_woning with interactive hoverable plot."""
    if 'plaatsnaam' not in df.columns or 'huurmaand_woning' not in df.columns:
        raise KeyError('Required columns "plaatsnaam" and "huurmaand_woning" are not in the dataset.')

    # Encode 'plaatsnaam' into numerical values
    label_encoder = LabelEncoder()
    df['plaatsnaam_encoded'] = label_encoder.fit_transform(df['plaatsnaam'])

    # Select features for clustering, dropping rows with missing values
    features = df[['plaatsnaam_encoded', 'huurmaand_woning']].dropna()

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    features['cluster'] = kmeans.fit_predict(features)

    # Create a cluster column in the original dataframe
    df['cluster'] = pd.NA
    df.loc[features.index, 'cluster'] = features['cluster']

    # Map the encoded plaatsnamen back to their original names, ensuring unique names in each cluster
    clustered_plaatsnamen = features.groupby('cluster')['plaatsnaam_encoded'].apply(
        lambda x: list(label_encoder.inverse_transform(x.unique()))
    )

    # Create an interactive plot using Plotly
    fig = px.scatter(
        features,
        x='plaatsnaam_encoded',
        y='huurmaand_woning',
        color='cluster',
        hover_data={'cluster': True, 'huurmaand_woning': True},
        title='K-Means Clustering: Plaatsnaam vs Huurprijs',
        labels={'plaatsnaam_encoded': 'Plaatsnaam', 'huurmaand_woning': 'Huurprijs/maand (€)'},
        color_continuous_scale='Viridis'
    )
    fig.update_traces(marker=dict(size=10))  # Customize marker size
    fig.update_layout(coloraxis_colorbar=dict(title='Cluster'))

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Calculate the average rent for each cluster
    cluster_averages = df.groupby('cluster')['huurmaand_woning'].median().round(2)
    cluster_averages_rounded = cluster_averages.round(0).astype(int)

    # Predefined categories for clusters
    predefined_categories = {
        0: "Expensive",
        2: "Medium",
        1: "Cheap"
    }

    # Create a DataFrame for visualization
    cluster_results = pd.DataFrame({
        "Cluster": cluster_averages_rounded.index,
        "Mediaan Huurprijs (€)": cluster_averages_rounded.values,
        "Categorie": [predefined_categories[cluster] for cluster in cluster_averages_rounded.index]
    })
    cluster_results = cluster_results.sort_values(by="Mediaan Huurprijs (€)", ascending=False)

    # Add color-coding for categories
    def highlight_category(row):
        if row["Categorie"] == "Expensive":
            return ["background-color: #FFCCCC"] * len(row)  # Light red
        elif row["Categorie"] == "Cheap":
            return ["background-color: #CCFFCC"] * len(row)  # Light green
        else:
            return ["background-color: #FFFFCC"] * len(row)  # Light yellow

    # Display the reordered cluster results
    st.subheader("Clustering Analyse Resultaten")
    st.dataframe(cluster_results.style.apply(highlight_category, axis=1), hide_index=True)
    
    # Define the desired cluster order
    cluster_order = [0, 2, 1]

    # Create a DataFrame for visualization
    cluster_results = pd.DataFrame({
        "Cluster": cluster_averages_rounded.index,
        "Mediaan Huurprijs (€)": cluster_averages_rounded.values,
        "Categorie": [predefined_categories[cluster] for cluster in cluster_averages_rounded.index]
    })

    # Reorder the clustered_plaatsnamen dictionary
    ordered_clustered_plaatsnamen = {key: clustered_plaatsnamen[key] for key in cluster_order}

    # Interactive search for locations within expanders
    for cluster, plaatsnamen in ordered_clustered_plaatsnamen.items():
        with st.expander(f"Cluster {cluster} ({len(plaatsnamen)} locations)"):
            # Search bar for each cluster
            search_query = st.text_input(f"Zoek plaatsnaam in Cluster {cluster}", key=f"search_{cluster}")
            
            # Filter locations based on the search query
            filtered_locations = [
                plaatsnaam for plaatsnaam in plaatsnamen 
                if search_query.lower() in plaatsnaam.lower()
            ]
            
            # Display the filtered or all locations
            if filtered_locations:
                st.write(", ".join(sorted(filtered_locations)))
            else:
                st.write("No matches found.")

def plaatsnaam_statistics(df):
    """Calculate and plot statistics for plaatsnaam vs huurmaand."""
    if 'plaatsnaam' not in df.columns or 'huurmaand_woning' not in df.columns:
        raise KeyError('Required columns "plaatsnaam" and "huurmaand_woning" are not in the dataset.')

    plaatsnaam_stats = df.groupby('plaatsnaam')['huurmaand_woning'].agg(['mean', 'count'])
    filtered_stats = plaatsnaam_stats[plaatsnaam_stats['count'] >= 10]

    top_10_expensive = filtered_stats['mean'].sort_values(ascending=False).head(10)
    top_10_cheapest = filtered_stats['mean'].sort_values(ascending=True).head(10)
    

    # Plot statistics
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    top_10_expensive.plot(kind='bar', color='red')
    plt.title('Top 10 Duurste Plaatsnamen Huur')
    plt.ylabel('Gemiddelde Huurprijs/maand (€)')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    top_10_cheapest.plot(kind='bar', color='green')
    plt.title('Top 10 Goedkoopste Plaatsnamen Huur')
    plt.ylabel('Gemiddelde Huurprijs/maand (€)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Top 10 Duurste Plaatsnamen (Gemiddelde huur):")
        st.write(top_10_expensive.round(0))
    
    with col2:
        st.write("\nTop 10 Goedkoopste Plaatsnamen (Gemiddelde huur):")
        st.write(top_10_cheapest.round(0))


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
        st.write("Top correlaties met huurprijs:")
        st.write(huurmaand_corr.head(25))
    with col2:    
        st.write("\nLaagste correlaties met huurprijs':")
        st.write(huurmaand_corr.tail(25))