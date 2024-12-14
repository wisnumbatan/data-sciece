import streamlit as st
import pandas as pd
import numpy as np
# Import library lainnya yang diperlukan dari notebook Anda

# Judul aplikasi
st.title("Sistem Rekomendasi Wisata")

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
menu = st.sidebar.selectbox("Pilih Menu", ["Home", "Eksplorasi Data", "Sistem Rekomendasi"])

# Fungsi untuk memuat data
def load_data():
    # Ganti dengan kode pemrosesan data Anda
    data = pd.DataFrame({"Contoh": ["Data1", "Data2", "Data3"]})  # Placeholder
    return data

# Home
if menu == "Home":
    st.write("Selamat datang di aplikasi Sistem Rekomendasi Wisata!")

# Eksplorasi Data
elif menu == "Eksplorasi Data":
    st.header("Eksplorasi Data")
    data = load_data()
    st.write("Dataset:")
    st.dataframe(data)

# Sistem Rekomendasi
elif menu == "Sistem Rekomendasi":
    st.header("Sistem Rekomendasi")

    # Input pengguna, misalnya kota atau preferensi wisata
    user_input = st.text_input("Masukkan preferensi wisata Anda:", "Pantai")

    # Logika rekomendasi placeholder (gantikan dengan model Anda)
    recommendations = ["Pantai Kuta", "Pantai Sanur", "Pantai Jimbaran"]  # Placeholder

    st.write("Rekomendasi untuk Anda:")
    for rec in recommendations:
        st.write(f"- {rec}")
