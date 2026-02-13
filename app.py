import streamlit as st
import pandas as pd
from datetime import datetime

st.title("Émargement digital")

uploaded_file = st.file_uploader("Importer un fichier Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if "Présent" not in df.columns:
        df["Présent"] = False
    if "Heure" not in df.columns:
        df["Heure"] = ""

    search = st.text_input("Rechercher un participant")

    for i, row in df.iterrows():
        text = " ".join([str(x) for x in row.values]).lower()
        
        if search.lower() in text:
            col1, col2 = st.columns([3,1])
            
            with col1:
                st.write(row)
            
            with col2:
                if not row["Présent"]:
                    if st.button("Émarger", key=i):
                        df.at[i, "Présent"] = True
                        df.at[i, "Heure"] = datetime.now().strftime("%H:%M:%S")
                        st.experimental_rerun()
                else:
                    st.write("✅")

    st.download_button(
        "Télécharger la liste",
        df.to_csv(index=False),
        "emargement.csv"
    )
