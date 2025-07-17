import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman
st.set_page_config(page_title="Klasterisasi Data", layout="wide")
st.title("🛍️ Aplikasi Klasterisasi Data Pelanggan")

# Direktori data
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def list_csv_files():
    return [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

def load_data(filename):
    return pd.read_csv(os.path.join(DATA_DIR, filename))

def save_data(df, filename):
    df.to_csv(os.path.join(DATA_DIR, filename), index=False)

def delete_row_by_id(filename, customer_id):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    df = df[df["CustomerID"] != customer_id]
    df.to_csv(path, index=False)

def delete_csv_file(filename):
    os.remove(os.path.join(DATA_DIR, filename))

# Menu utama
menu = st.sidebar.selectbox("📂 Menu", [
    "Beranda",
    "Kluster Data",
    "Tambah Data",
    "Hapus Data",
    "Manajemen File CSV"
])

# Beranda
if menu == "Beranda":
    st.header("📌 Selamat Datang")
    st.write("""
        Aplikasi ini membantu proses **Klasterisasi Data** menggunakan metode **K-Means Clustering**.
        Gunakan menu di sidebar untuk menambahkan data, menghapus data, dan melakukan klasterisasi.
    """)

# Klaster Data
elif menu == "Kluster Data":
    st.header("📊 Klasterisasi")

    csv_files = list_csv_files()
    if not csv_files:
        st.warning("⚠️ Tidak ada file CSV ditemukan.")
        st.stop()

    filename = st.selectbox("Pilih File CSV", csv_files, key="klaster_file")
    data = load_data(filename)

    fitur_numerik = data.select_dtypes(include='number').columns.tolist()
    fitur = st.multiselect("Pilih Fitur untuk Klasterisasi", fitur_numerik, default=fitur_numerik[:2], key="fitur_klaster")

    k = st.number_input("Masukkan Jumlah Klaster (k)", min_value=2, value=3, step=1, format="%d")

    if len(fitur) == 2:
        X = data[fitur]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=int(k), random_state=0)
        kmeans.fit(X_scaled)
        data['Cluster'] = kmeans.labels_

        st.success("✅ Klasterisasi berhasil dilakukan!")

        st.subheader("🔎 Pilih Kolom untuk Pencarian")
        semua_kolom = list(data.columns)
        kolom_difilter = st.multiselect("Pilih Kolom Target Filter", semua_kolom, default=fitur)

        st.subheader("🔍 Masukkan Kata Kunci")
        kata_kunci = st.text_input("Masukkan kata kunci untuk filter")

        if kata_kunci and kolom_difilter:
            filter_mask = pd.Series(False, index=data.index)
            for col in kolom_difilter:
                filter_mask |= data[col].astype(str).str.contains(kata_kunci, case=False, na=False)
            data_filtered = data[filter_mask]
        else:
            data_filtered = data

        st.subheader("📄 Tabel Hasil Klaster")
        page_size = 10
        total_pages = (len(data_filtered) - 1) // page_size + 1
        page = st.number_input("Halaman", min_value=1, max_value=total_pages, step=1)
        start = (page - 1) * page_size
        end = start + page_size
        st.dataframe(data_filtered.iloc[start:end])

        fig, ax = plt.subplots()
        sns.scatterplot(data=data_filtered, x=fitur[0], y=fitur[1], hue="Cluster", palette="Set2", ax=ax)
        plt.title("Visualisasi Klasterisasi")
        st.pyplot(fig)
    else:
        st.info("Pilih **dua fitur numerik** untuk melanjutkan proses klasterisasi.")

# Tambah Data
elif menu == "Tambah Data":
    st.header("➕ Tambah Data Pelanggan")

    csv_files = list_csv_files()
    if not csv_files:
        st.warning("⚠️ Tidak ada file CSV tersedia.")
    else:
        filename = st.selectbox("Pilih File CSV", csv_files, key="tambah_select")
        df = load_data(filename)

        with st.form("form_tambah"):
            st.subheader("📝 Form Tambah Data")
            input_data = {}

            for col in df.columns:
                dtype = df[col].dtype

                if pd.api.types.is_numeric_dtype(dtype):
                    min_val = int(df[col].min()) if not df[col].isnull().all() else 0
                    max_val = int(df[col].max()) if not df[col].isnull().all() else 100
                    value = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=min_val)
                else:
                    value = st.text_input(f"{col}")
                input_data[col] = value

            submitted = st.form_submit_button("Simpan")
            if submitted:
                new_df = pd.DataFrame([input_data])
                df = pd.concat([df, new_df], ignore_index=True)
                save_data(df, filename)
                st.success("✅ Data berhasil ditambahkan.")

# Hapus Data
elif menu == "Hapus Data":
    st.header("🗑️ Hapus Data Pelanggan")

    csv_files = list_csv_files()
    if not csv_files:
        st.warning("⚠️ Tidak ada file CSV.")
    else:
        filename = st.selectbox("Pilih File CSV", csv_files, key="hapus_select")
        df = load_data(filename)
        st.dataframe(df)

        customer_id = st.number_input("Masukkan Customer ID yang ingin dihapus", min_value=1, key="hapus_id")
        if st.button("Hapus Data"):
            if customer_id in df["CustomerID"].values:
                delete_row_by_id(filename, customer_id)
                st.success(f"✅ Customer ID {customer_id} berhasil dihapus.")
            else:
                st.warning("⚠️ Customer ID tidak ditemukan.")

# Manajemen File CSV
elif menu == "Manajemen File CSV":
    st.header("📁 Manajemen File CSV")

    st.subheader("📤 Tambah File CSV Baru")
    uploaded_file = st.file_uploader("Upload file CSV", type="csv")
    if uploaded_file:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ File {uploaded_file.name} berhasil ditambahkan.")

    st.subheader("🗑️ Hapus File CSV")
    csv_files = list_csv_files()
    if csv_files:
        file_to_delete = st.selectbox("Pilih File yang Ingin Dihapus", csv_files)
        if st.button("Hapus File"):
            delete_csv_file(file_to_delete)
            st.success(f"✅ File {file_to_delete} berhasil dihapus.")
    else:
        st.info("Tidak ada file CSV tersedia.")
