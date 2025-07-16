import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Setup halaman
st.set_page_config(page_title="Dashboard Klasterisasi", layout="wide")
st.title("👩‍🏫 Aplikasi Klasterisasi Pelanggan Mall")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def list_csv_files():
    return [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

def load_data(filename):
    return pd.read_csv(os.path.join(DATA_DIR, filename))

def save_data(df, filename):
    df.to_csv(os.path.join(DATA_DIR, filename), index=False)

def delete_row_by_id(filename, identifier):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    if "CustomerID" in df.columns:
        df = df[df["CustomerID"] != identifier]
    elif "Nama Lengkap" in df.columns:
        df = df[df["Nama Lengkap"] != identifier]
    df.to_csv(path, index=False)

tabs = st.tabs([
    "Beranda",
    "Dashboard Data",
    "Tambah Data",
    "Hapus Data"
])

# Tab 1: Beranda
with tabs[0]:
    st.header("📌 Selamat Datang")
    st.write("""
        Aplikasi ini membantu proses **klasterisasi pelanggan mall** menggunakan metode **K-Means Clustering**.
        Gunakan menu di tab atas untuk mengelola data, menambahkan, menghapus, dan menganalisis file CSV.
    """)

# Tab 2: Dashboard Data
with tabs[1]:
    st.header("📊 Dashboard Klasterisasi")

    csv_files = list_csv_files()
    if not csv_files:
        st.warning("⚠️ Tidak ada file CSV ditemukan. Silakan upload file CSV ke folder 'data/' secara manual.")
        st.stop()
    filename = st.selectbox("Pilih File CSV", csv_files, key="dashboard_select")
    data = load_data(filename)

    # Paksa konversi kolom 'Usia' ke numerik jika ada
    if 'Usia' in data.columns:
        data['Usia'] = pd.to_numeric(data['Usia'], errors='coerce')

    fitur_numerik = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not fitur_numerik:
        st.warning("⚠️ File ini tidak memiliki kolom numerik untuk klasterisasi.")
        st.stop()

    st.subheader("⚙️ Klasterisasi Pelanggan")
    fitur = st.multiselect("Pilih Fitur untuk Klasterisasi", fitur_numerik, key="fitur_klaster")

    k = st.slider("Pilih Jumlah Klaster (k)", 2, 10, 3, key="slider_k")

    if len(fitur) >= 1:
        X = data[fitur]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_scaled)
        data['Cluster'] = kmeans.labels_

        st.success("✅ Klasterisasi berhasil dilakukan!")
    else:
        st.info("Pilih minimal dua fitur numerik untuk proses klasterisasi.")
        data['Cluster'] = None

    st.subheader("🔎 Pilih Kolom untuk Ditampilkan")
    semua_kolom = list(data.columns)
    kolom_dipilih = st.multiselect("Pilih kolom", semua_kolom, default=semua_kolom, key="pilih_kolom")

    st.subheader("🔍 Cari Data")
    kata_kunci = st.text_input("Masukkan kata kunci pencarian")

    if kata_kunci and kolom_dipilih:
        filter_mask = pd.Series(False, index=data.index)
        for col in kolom_dipilih:
            filter_mask = filter_mask | data[col].astype(str).str.contains(kata_kunci, case=False, na=False)
        data_filtered = data[filter_mask]
    else:
        data_filtered = data

    if kolom_dipilih:
        st.dataframe(data_filtered[kolom_dipilih])
    else:
        st.warning("⚠️ Pilih minimal satu kolom untuk ditampilkan.")

    # Visualisasi jika klasterisasi berhasil
    if len(fitur) >= 2 and data['Cluster'].notna().all():
        fig, ax = plt.subplots()
        sns.scatterplot(data=data_filtered, x=fitur[0], y=fitur[1], hue="Cluster", palette="Set2", ax=ax)
        plt.title("Visualisasi Klaster")
        st.pyplot(fig)

# Tab 3: Tambah Data
with tabs[2]:
    st.header("➕ Tambah Data (Manual)")

    st.info("Fitur tambah data hanya mendukung CSV dengan kolom standar seperti 'Usia' dan 'Nama Lengkap'.")

    csv_files = list_csv_files()
    if not csv_files:
        st.warning("⚠️ Tidak ada file CSV tersedia.")
    else:
        filename = st.selectbox("Pilih File CSV", csv_files, key="tambah_select")

        with st.form("form_tambah"):
            nama = st.text_input("Nama Lengkap")
            usia = st.number_input("Usia", min_value=1, max_value=100)

            submitted = st.form_submit_button("Simpan")

            if submitted:
                new_row = {
                    "Nama Lengkap": nama,
                    "Usia": usia
                }
                df = load_data(filename)
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                save_data(df, filename)
                st.success("✅ Data berhasil ditambahkan!")

# Tab 4: Hapus Data
with tabs[3]:
    st.header("🗑️ Hapus Data")

    csv_files = list_csv_files()
    if not csv_files:
        st.warning("⚠️ Tidak ada file CSV.")
    else:
        filename = st.selectbox("Pilih File CSV", csv_files, key="hapus_select")
        df = load_data(filename)
        st.dataframe(df)

        identifier = st.text_input("Masukkan Nama Lengkap yang ingin dihapus", key="hapus_id")
        if st.button("Hapus Data"):
            if identifier in df["Nama Lengkap"].values:
                delete_row_by_id(filename, identifier)
                st.success(f"✅ Data untuk '{identifier}' berhasil dihapus.")
            else:
                st.warning("⚠️ Nama tidak ditemukan.")
