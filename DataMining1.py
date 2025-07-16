import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Klasterisasi Mall Customers", layout="wide")

st.title("Aplikasi Klasterisasi Data Mall Customers")

# Load default dataset
@st.cache_data
def load_default_data():
    data = '''CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
6,Female,22,17,76
7,Female,35,18,6
8,Female,23,18,94
9,Male,64,19,3
10,Female,30,19,72'''
    return pd.read_csv(io.StringIO(data))

# Upload CSV
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Menggunakan data default karena tidak ada file yang diupload.")
    df = load_default_data()

st.subheader("Data Customer")
st.dataframe(df, use_container_width=True)

# Identifikasi fitur numerik
fitur_numerik = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
fitur_numerik = [col for col in fitur_numerik if col not in ['CustomerID']]

if not fitur_numerik:
    st.error("⚠️ Tidak ada kolom numerik yang bisa digunakan untuk klasterisasi.")
else:
    st.sidebar.header("Pengaturan Klasterisasi")
    fitur = st.sidebar.multiselect("Pilih Fitur", fitur_numerik, default=fitur_numerik[:2], key="fitur_klaster")
    k = st.sidebar.slider("Jumlah Klaster (K)", min_value=2, max_value=10, value=3)

    if fitur:
        X = df[fitur]
        model = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = model.fit_predict(X)

        st.subheader("Hasil Klasterisasi")
        st.write(df.head())

        # Visualisasi
        if len(fitur) == 2:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=fitur[0], y=fitur[1], hue='Cluster', palette='tab10', ax=ax)
            ax.set_title("Visualisasi Klaster")
            st.pyplot(fig)
        elif len(fitur) == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(df[fitur[0]], df[fitur[1]], df[fitur[2]], c=df['Cluster'], cmap='tab10')
            ax.set_xlabel(fitur[0])
            ax.set_ylabel(fitur[1])
            ax.set_zlabel(fitur[2])
            ax.set_title("Visualisasi Klaster 3D")
            st.pyplot(fig)
        else:
            st.info("Pilih maksimal 3 fitur untuk visualisasi.")

        # Download hasil klasterisasi
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Klasterisasi CSV",
            data=csv,
            file_name='hasil_klasterisasi.csv',
            mime='text/csv'
        )

# Tambah Data Manual
st.subheader("Tambah Data Baru (Opsional)")
with st.expander("📌 Klik untuk Tambah Data"):
    with st.form("form_tambah"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        income = st.number_input("Annual Income (k$)", min_value=1, value=50)
        score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
        submit = st.form_submit_button("Tambah")

        if submit:
            new_data = pd.DataFrame({
                'CustomerID': [df['CustomerID'].max() + 1 if 'CustomerID' in df.columns else len(df)+1],
                'Gender': [gender],
                'Age': [age],
                'Annual Income (k$)': [income],
                'Spending Score (1-100)': [score]
            })
            df = pd.concat([df, new_data], ignore_index=True)
            st.success("✅ Data berhasil ditambahkan. Silakan ulangi klasterisasi jika perlu.")
