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
## Hi there! 👋  
I’m **Adam Asbai Halifa**, a **Data Engineer** currently working at **Volantis**.  

I hold a **Bachelor’s degree in Structural Engineering and Computer Science**, which has given me a unique blend of technical and analytical skills.  

Currently, I am pursuing a **Pre-Master's program in Data Science** at Radboud University, expanding my expertise in advanced analytics, machine learning, and cutting-edge data technologies.  

In addition to my role at Volantis as a data **engineer**, I work part-time as a **freelancer**, specializing in **data-driven solutions**. With expertise in **Python programming**, I’m deeply interested in **machine learning**, **data science**, **data analysis**, and the transformative power of data to drive decisions and innovations.  

When I’m not building data pipelines or exploring datasets, I enjoy staying active through workouts, traveling on my motorbike, and keeping up with the latest technology trends.
""")

# Contact details
st.markdown("### **Contact Me**")
st.markdown(f"""
- **Website**: [datablueprintconsulting.nl](https://datablueprintconsulting.nl/)  
- **LinkedIn**: https://www.linkedin.com/in/adam-asbai-halifa/ 
- **Email**: [a.asbaihalifa@dbp-c.nl](mailto:a.asbaihalifa@dbp-c.nl)  
- **Phone**: +31 6 43801509
""")

# Add a call-to-action for collaboration
st.markdown("""
---
🌟 Feel free to reach out—I’m always open to connecting and collaborating on exciting projects!
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