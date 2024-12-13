import streamlit as st

# Set the page configuration
st.set_page_config(page_title="About Me 🙋🏽‍♂️", layout="wide", page_icon='favicon.ico')

# Main heading
st.title("About Me 🙋🏽‍♂️")
st.logo('logo.png')
st.html("""
    <style>
    [alt=Logo] {
        height: 2,5rem;
    }
    </style>
""")

st.markdown("""
## Hallo! 👋  
Mijn naam is **Adam Asbai Halifa**, een **Data Engineer** die momenteel werkt bij **Volantis**.  

Ik heb een **bacheloropleiding in Bouwkunde en Informatica** afgerond, wat mij een unieke combinatie van technische en analytische vaardigheden heeft gegeven.  

Momenteel volg ik een **Pre-Master Data Science** aan de Radboud Universiteit, waar ik mijn expertise uitbreid op het gebied van data analyses, data science, machine learning en data mining.  

Naast mijn rol als **data engineer** bij Volantis, werk ik parttime als **freelancer**, gespecialiseerd in **data-gedreven oplossingen**. Met mijn expertise in **Python-programmeren** ben ik sterk geïnteresseerd in **machine learning**, **data science**, **data-analyse** en het toepassen van data om beslissingen en innovaties te stimuleren.  

Wanneer ik kan niet achter mijn laptop zit, blijf ik graag actief door te sporten, te reizen op mijn motor en op de hoogte te blijven van de nieuwste technologische trends.
""")


# Contact details
st.markdown("### **Contact Me**")
st.markdown(f"""
- **Website**: [datablueprintconsulting.nl](https://datablueprintconsulting.nl/)  
- **LinkedIn**: https://www.linkedin.com/in/adam-asbai-halifa/ 
- **Email**: [a.asbaihalifa@dbp-c.nl](mailto:a.asbaihalifa@dbp-c.nl)  
- **Datablueprint Consulting**: [Github](https://github.com/DataBlueprintConsulting)  
- **Telefoonnummer**: +31 6 43801509
""")

# Add a call-to-action for collaboration
st.markdown("""
---
🌟 Voel je vrij om contact op te nemen. Ik sta altijd open voor connecties en samenwerkingen aan interessante data projecten!
""")

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