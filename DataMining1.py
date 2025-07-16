import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Klasterisasi Mall Customers", layout="wide")

st.title("Mall Customers Clustering")

# Load CSV
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Awal")
    st.write(df.head())

    # Tambah Data Baru
    st.subheader("Tambah Data Baru")
    with st.form("form_tambah"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=1, max_value=100, value=30)
        income = st.number_input("Annual Income (k$)", min_value=1, value=50)
        score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
        submitted = st.form_submit_button("Tambah")

        if submitted:
            new_row = {
                'CustomerID': df['CustomerID'].max() + 1 if 'CustomerID' in df.columns else len(df) + 1,
                'Gender': gender,
                'Age': age,
                'Annual Income (k$)': income,
                'Spending Score (1-100)': score
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.success("Data berhasil ditambahkan!")

    # Identifikasi fitur numerik
    fitur_numerik = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    fitur_numerik = [col for col in fitur_numerik if col != 'CustomerID']

    if not fitur_numerik:
        st.error("⚠️ File ini tidak memiliki kolom numerik untuk klasterisasi.")
    else:
        # Pilih fitur numerik untuk klasterisasi
        fitur = st.multiselect("Pilih Fitur untuk Klasterisasi", fitur_numerik, default=fitur_numerik[:2], key="fitur_klaster")

        if fitur:
            X = df[fitur]
            # Pilih jumlah klaster
            k = st.slider("Jumlah Klaster (K)", min_value=2, max_value=10, value=3)
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
else:
    st.info("Silakan upload file CSV terlebih dahulu.")
