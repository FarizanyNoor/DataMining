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

menu = st.sidebar.selectbox("📂 Menu", [
    "Beranda",
    "Kluster Data",
    "Tambah Data",
    "Hapus Data",
])


# Tab 1: Beranda
if menu == "Beranda":
    st.header("📌 Selamat Datang")
    st.write("""
        Aplikasi ini membantu proses **Klasterisasi Data** menggunakan metode **K-Means Clustering**.
        Gunakan menu di tab atas untuk mengelola data, menambahkan, menghapus, dan upload file CSV.
    """)

# Tab 2: Dashboard Data
elif menu == "Kluster Data":
    st.header("📊Klasterisasi")

    # 1. Pilih file CSV
    csv_files = list_csv_files()
    if not csv_files:
        st.warning("⚠️ Tidak ada file CSV ditemukan. Silakan upload file CSV terlebih dahulu.")
        st.stop()
    filename = st.selectbox("Pilih File CSV", csv_files, key="dashboard_select")
    data = load_data(filename)

    # 2. Klasterisasi
    st.subheader("⚙️ Klasterisasi Pelanggan")
    fitur_numerik = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    fitur = st.multiselect("Pilih Fitur untuk Klasterisasi", fitur_numerik, default=fitur_numerik[:2], key="fitur_klaster")
    
    k = st.slider("Pilih Jumlah Klaster (k)", 2, 10, 3, key="slider_k")

    if len(fitur) >= 2:
        X = data[fitur]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_scaled)
        data['Cluster'] = kmeans.labels_

        st.success("✅ Klasterisasi berhasil dilakukan!")
    else:
        st.info("Pilih minimal dua fitur numerik untuk proses klasterisasi.")

    # 3. Pilih kolom yang ingin ditampilkan
    st.subheader("🔎 Pilih Kolom untuk Ditampilkan")
    semua_kolom = list(data.columns)
    kolom_dipilih = st.multiselect("Pilih kolom", semua_kolom, default=semua_kolom, key="pilih_kolom")

    # 4. Input pencarian teks
    st.subheader("🔍 Cari Data (Filter Baris)")
    kata_kunci = st.text_input("Masukkan kata kunci pencarian")

    # Filter data berdasarkan pencarian di kolom yang dipilih saja
    if kata_kunci and kolom_dipilih:
        filter_mask = pd.Series(False, index=data.index)
        for col in kolom_dipilih:
            filter_mask = filter_mask | data[col].astype(str).str.contains(kata_kunci, case=False, na=False)
        data_filtered = data[filter_mask]
    else:
        data_filtered = data

    # Tampilkan data hasil filter dan kolom terpilih
    if kolom_dipilih:
        st.dataframe(data_filtered[kolom_dipilih])
    else:
        st.warning("⚠️ Pilih minimal satu kolom untuk ditampilkan.")

    # Jika klasterisasi berhasil, tampilkan scatterplot
    if len(fitur) = 2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=data_filtered, x=fitur[0], y=fitur[1], hue="Cluster", palette="Set1", ax=ax)
        plt.title("Hasil Klasterisasi")
        st.pyplot(fig)

# Tab 3: Tambah Data
elif menu == "Tambah Data":
    st.header("➕ Tambah Data Pelanggan")

    csv_files = list_csv_files()
    if not csv_files:
        st.warning("⚠️ Tidak ada file CSV tersedia. Silakan upload terlebih dahulu.")
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
                    st.warning(f"⚠️ Customer ID {customer_id} sudah ada. Gunakan ID lain.")
                else:
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    save_data(df, filename)
                    st.success("✅ Data berhasil ditambahkan!")

# Tab 4: Hapus Data
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

