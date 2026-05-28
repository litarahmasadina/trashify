import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard EDA", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")
st.success("**Pertanyaan Bisnis:** Bagaimana proporsi kelas sampah pada dataset saat ini, dan apakah keseimbangan data sudah cukup baik untuk melatih model AI?")

# Load data hasil ekstraksi folder
@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_info.csv")

df = load_data()

# 1. Total Gambar Keseluruhan
total_images = df['Jumlah_Gambar'].sum()
st.metric("Total Keseluruhan Gambar Dataset", f"{total_images} File")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.write("**1. Proporsi Kategori Keseluruhan**")
    df_kategori = df.groupby('Kategori')['Jumlah_Gambar'].sum().reset_index()
    fig_pie = px.pie(df_kategori, names='Kategori', values='Jumlah_Gambar', 
                     color='Kategori', hole=0.4,
                     color_discrete_map={'Organik':'#2ecc71', 'Anorganik':'#3498db', 'Residu':'#e74c3c'})
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.write("**2. Distribusi Data (Train, Val, Test)**")
    fig_bar = px.bar(df, x='Split', y='Jumlah_Gambar', color='Kategori', barmode='group',
                     color_discrete_map={'Organik':'#2ecc71', 'Anorganik':'#3498db', 'Residu':'#e74c3c'})
    st.plotly_chart(fig_bar, use_container_width=True)
    
