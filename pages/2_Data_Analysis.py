import streamlit as st
import pandas as pd
from modules.data_analysis.data_analysis import (
    kmeans_clustering,
    plaatsnaam_statistics,
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
    st.session_state["selected_places"] = ["Alle locaties"]

# Reset button
if st.button("Reset Filters"):
    st.session_state["selected_places"] = ["Alle locaties"]

# Meerdere locaties selecteren via dropdown
selected_places = st.multiselect(
    "Selecteer een of meerdere locaties:",
    options=["Alle locaties"] + list(df['plaatsnaam'].unique()),
    default=st.session_state["selected_places"],  # Gebruik sessiestatus om selectie te bewaren/resetten
    help="Zoek of selecteer locaties om het dataset te filteren. Kies 'Alle locaties' om alles te includeren."
)


# Update session state with the user's selection
st.session_state["selected_places"] = selected_places

# Apply filter logic
if "Alle locaties" in selected_places:
    filtered_df = df
else:
    filtered_df = df[df['plaatsnaam'].isin(selected_places)]

# Display filtered data
st.write(f"Gefiltereerde Dataset: {filtered_df.shape[0]} rijen")
st.dataframe(filtered_df, hide_index=True)

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    # Dynamic Key Metrics
    st.subheader('🔑 Key Metrics')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### Huurprijs/maand")
        st.write(f"**Min:** {int(filtered_df['huurmaand_woning'].min())} €")
        st.write(f"**Mean:** {int(filtered_df['huurmaand_woning'].mean().round(2))} €")
        st.write(f"**Median:** {int(filtered_df['huurmaand_woning'].median())} €")
        st.write(f"**Mode:** {int(filtered_df['huurmaand_woning'].mode()[0].round(2))} €")
        st.write(f"**Max:** {int(filtered_df['huurmaand_woning'].max())} €")

    with col2:
        st.markdown("### Bouwjaar")
        st.write(f"**Min:** {int(filtered_df['bouwjaar'].min())}")
        st.write(f"**Mean:** {int(filtered_df['bouwjaar'].mean())}")
        st.write(f"**Median:** {int(filtered_df['bouwjaar'].median())}")
        st.write(f"**Mode:** {int(filtered_df['bouwjaar'].mode()[0])}")
        st.write(f"**Max:** {int(filtered_df['bouwjaar'].max())}")

    with col3:
        st.markdown("### Inhoud")
        st.write(f"**Min:** {int(filtered_df['inhoud'].min())} m³")
        st.write(f"**Mean:** {int(filtered_df['inhoud'].mean().round(2))} m³")
        st.write(f"**Median:** {int(filtered_df['inhoud'].median())} m³")
        st.write(f"**Mode:** {int(filtered_df['inhoud'].mode()[0])} m³")
        st.write(f"**Max:** {int(filtered_df['inhoud'].max())} m³")

    with col4:
        st.markdown("### Opp Wonen")
        cleaned_oppervlakte_wonen = filtered_df['oppervlakte_wonen'].dropna()
        st.write(f"**Min:** {int(cleaned_oppervlakte_wonen.min())} m²")
        st.write(f"**Mean:** {int(cleaned_oppervlakte_wonen.mean().round(2))} m²")
        st.write(f"**Median:** {int(cleaned_oppervlakte_wonen.median())} m²")
        st.write(f"**Mode:** {int(cleaned_oppervlakte_wonen.mode()[0])} m²")
        st.write(f"**Max:** {int(cleaned_oppervlakte_wonen.max())} m²")

    st.markdown("""
    ### Over de Metrics
    De bovenstaande statistieken bieden een overzicht van de belangrijkste kenmerken van huurwoningen in de dataset. Wanneer u filters toepast, worden de statistieken dynamisch aangepast op basis van uw selectie.

    - **Huurprijs/maand Woning**: Een samenvatting van huurprijzen (in euro's) per maand, die een idee geeft van het prijsbereik, het gemiddelde en typische prijspatronen.
    - **Bouwjaar**: Inzichten in de bouwjaren van woningen, met nadruk op de leeftijdsverdeling en trends in de tijd.
    - **Inhoud**: Details over het volume van woningen (in kubieke meters), nuttig voor het begrijpen van de ruimtelijkheid van woningen.
    - **Oppervlakte Wonen**: Informatie over de woonoppervlakte (in vierkante meters), waarmee de grootte van woningen kan worden beoordeeld en verschillende aanbiedingen kunnen worden vergeleken.
    """)


    st.markdown("## Advanced Analytics")

    # K-Means Clustering
    st.subheader('🔗 K-Means Clustering: Plaatsnaam vs Huurprijs')
    try:
        kmeans_clustering(df)
    except KeyError as e:
        st.text(str(e))

    st.markdown("""
    #### Over K-Means Clustering
    K-Means clustering is een unsupervised machine learning-algoritme dat wordt gebruikt om patronen of groeperingen in de gegevens te identificeren.
    - **Doel**: In deze analyse clusteren we huurprijzen (`Huurprijs`) op basis van locaties (`Plaatsnaam`) om trends en groeperingen in de dataset te ontdekken.
    - **Waarom**: Dit kan helpen om gebieden met vergelijkbare huurprijsklassen te identificeren, afwijkingen op te sporen en inzichten te bieden voor strategische besluitvorming.
    - **Inzichten**: De resulterende clusters benadrukken patronen in de relatie tussen locatie en huurprijzen, waardoor een beter begrip van de marktsegmentatie mogelijk wordt.

    #### Inzichten uit de Clusteranalyse
    - `Cluster4` == **'Premium Segment' Clusters**: Woningen in exclusieve gebieden.
    - `Cluster3` == **'Luxe Segment' Clusters**: Premium woningen in gewilde en populaire locaties.
    - `Cluster1` == **'Hoog Segment' Clusters**: Woningen die een balans bieden tussen kwaliteit en comfort.
    - `Cluster2` == **'Midden Segment' Clusters**: Betaalbare woningen met een goede prijs-kwaliteitverhouding.
    - `Cluster0` == **'Laag Segment' Clusters**: De meest toegankelijke en budgetvriendelijke woningen.

    """)
    
    # Plaatsnaam Statistics
    st.subheader('📍 Plaatsnaam Statistics')
    try:
        plaatsnaam_statistics(df)
    except KeyError as e:
        st.text(str(e))
    
    st.markdown("""
    ### Over Plaatsnaamstatistieken
    Deze sectie biedt een gedetailleerde statistische uitsplitsing van huurgegevens op basis van verschillende locaties (`Plaatsnaam`).

    - **Doel**: Verkennen hoe huurprijzen variëren tussen verschillende steden of dorpen. Deze analyse helpt bij het identificeren van patronen en trends die specifiek zijn voor elke locatie.
    - **Waarom**: Biedt een vergelijkend overzicht van locaties, zodat gebruikers datagedreven beslissingen kunnen nemen, of het nu gaat om het kiezen van een betaalbare regio of het begrijpen van premium locaties.
    - **Inzichten**: Gebruikers kunnen gemiddelde huurkosten, veelvoorkomende trends en uitschieters binnen specifieke gebieden ontdekken, wat waardevolle informatie biedt voor huurders, vastgoedinvesteerders en analisten.
    """)


    # Correlation Analysis
    st.subheader('📈 Correlation Analysis')
    try:
        correlation_analysis(df)
    except KeyError as e:
        st.text(str(e))
    st.markdown(
        """
        ### Over de correlatiematrix
        De **correlatieanalyse** onderzoekt hoe numerieke kenmerken in de dataset met elkaar samenhangen.
        Dit helpt te begrijpen welke variabelen een sterke positieve of negatieve relatie hebben, wat 
        nuttig kan zijn voor voorspellende modellen en feature-engineering.
        """
    )


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
