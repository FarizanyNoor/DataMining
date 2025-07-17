import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Setup halaman
st.set_page_config(page_title="Klasterisasi", layout="wide")
st.title("🛍️ Aplikasi Klasterisasi Data")

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

# ====================== MENU SIDEBAR ============================
menu = st.sidebar.selectbox("📂 Menu", [
    "📁 Upload File CSV",
    "Beranda",
    "Kluster Data",
    "Tambah Data",
    "Hapus Data",
])

# ===================== MENU 0: Upload File ========================
if menu == "📁 Upload File CSV":
    st.header("📤 Upload File CSV Baru")

    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded_file:
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"✅ File '{uploaded_file.name}' berhasil diunggah ke folder data/")

# ===================== MENU 1: Beranda ============================
elif menu == "Beranda":
    st.header("📌 Selamat Datang")
    st.write("""
        Aplikasi ini membantu proses **Klasterisasi Data** menggunakan metode **K-Means Clustering**.
        Gunakan menu di samping untuk mengelola data, menambahkan, menghapus, dan melakukan klasterisasi.
    """)

# ===================== MENU 2: Klasterisasi ========================
elif menu == "Kluster Data":
    st.header("📊 Klasterisasi")

    csv_files = list_csv_files()
    if not csv_files:
        st.warning("⚠️ Tidak ada file CSV ditemukan.")
        st.stop()

    filename = st.selectbox("Pilih File CSV", csv_files, key="dashboard_select")
    data = load_data(filename)

    st.subheader("⚙️ Klasterisasi Pelanggan")
    fitur_numerik = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    fitur = st.multiselect("Pilih Fitur untuk Klasterisasi", fitur_numerik, default=fitur_numerik[:2], key="fitur_klaster")
    k = st.slider("Pilih Jumlah Klaster (k)", 2, 10, 3, key="slider_k")

    if len(fitur) == 2:
        X = data[fitur]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_scaled)
        data['Cluster'] = kmeans.labels_

        st.success("✅ Klasterisasi berhasil dilakukan!")
    else:
        st.info("Hanya bisa memilih dua fitur numerik untuk proses klasterisasi.")
        st.stop()

    st.subheader("🔍 Filter Data Berdasarkan Kolom yang Dipilih")
    semua_kolom = list(data.columns)
    kolom_filter = st.multiselect("Pilih Kolom Target Pencarian", semua_kolom, default=semua_kolom)
    kata_kunci = st.text_input("Masukkan Kata Kunci Pencarian")

    if kata_kunci and kolom_filter:
        filter_mask = pd.Series(False, index=data.index)
        for kolom in kolom_filter:
            filter_mask |= data[kolom].astype(str).str.contains(kata_kunci, case=False, na=False)
        data_filtered = data[filter_mask]
    else:
        data_filtered = data

    # Pagination
    st.subheader("📄 Hasil Data (dengan Pagination)")
    rows_per_page = st.selectbox("Jumlah baris per halaman", [5, 10, 20, 50], index=1)
    total_rows = len(data_filtered)
    total_pages = (total_rows - 1) // rows_per_page + 1
    page_number = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, step=1)

    start_idx = (page_number - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    st.write(f"Menampilkan baris {start_idx + 1} - {min(end_idx, total_rows)} dari {total_rows}")
    st.dataframe(data_filtered.iloc[start_idx:end_idx], use_container_width=True)

    fig, ax = plt.subplots()
    sns.scatterplot(data=data_filtered, x=fitur[0], y=fitur[1], hue="Cluster", palette="Set1", ax=ax)
    plt.title("Hasil Klasterisasi")
    st.pyplot(fig)

# ===================== MENU 3: Tambah Data =========================
elif menu == "Tambah Data":
    st.header("➕ Tambah Data Pelanggan")

    csv_files = list_csv_files()
    if not csv_files:
        st.warning("⚠️ Tidak ada file CSV tersedia.")
    else:
        filename = st.selectbox("Pilih File CSV", csv_files, key="tambah_select")

        with st.form("form_tambah"):
            customer_id = st.number_input("Customer ID", min_value=1)
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=10, max_value=100)
            income = st.number_input("Annual Income (k$)", min_value=1)
            score = st.slider("Spending Score (1-100)", 1, 100, 50)

            submitted = st.form_submit_button("Simpan")

            if submitted:
                new_row = {
                    "CustomerID": customer_id,
                    "Gender": gender,
                    "Age": age,
                    "Annual Income (k$)": income,
                    "Spending Score (1-100)": score
                }
                df = load_data(filename)
                if customer_id in df["CustomerID"].values:
                    st.warning(f"⚠️ Customer ID {customer_id} sudah ada.")
                else:
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    save_data(df, filename)
                    st.success("✅ Data berhasil ditambahkan!")

# ===================== MENU 4: Hapus Data =========================
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
