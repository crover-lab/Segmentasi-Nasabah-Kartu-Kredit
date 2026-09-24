import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# Konfigurasi Halaman
# ==========================================================
st.set_page_config(
    page_title="Segmentasi Nasabah Kartu Kredit",
    page_icon=":bar_chart:",
    layout="wide"
)

# ==========================================================
# Load Model & Scaler
# ==========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("kmeans_cc_model.pkl")
    scaler = joblib.load("scaler_cc_model.pkl")
    return model, scaler

model, scaler = load_artifacts()

# Nama segmen sesuai hasil profiling di notebook
CLUSTER_LABELS = {
    0: "High-Value Spender",
    1: "Cash Advance Dependent (Berisiko)",
    2: "Standard User",
    3: "Nasabah Pasif / Dormant"
}

CLUSTER_DESC = {
    0: "Saldo dan pembelian tinggi, limit kredit tertinggi. Nasabah aktif berbelanja dengan nilai besar.",
    1: "Pembelian rendah namun cash advance tinggi. Berpotensi berisiko dari sisi kredit.",
    2: "Seluruh metrik berada di level moderat-rendah. Aktivitas standar.",
    3: "Saldo dan pembelian paling rendah. Aktivitas paling minim."
}

# Warna konsisten untuk tiap segmen pada seluruh grafik
CLUSTER_COLORS = {
    "High-Value Spender": "#2E86AB",
    "Cash Advance Dependent (Berisiko)": "#D9534F",
    "Standard User": "#5CB85C",
    "Nasabah Pasif / Dormant": "#A6A6A6"
}

# Fitur yang di-log-transform di tahap Data Preparation (harus sama persis dengan notebook)
SKEWED_FEATURES = [
    "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"
]

# Urutan fitur final persis seperti saat scaler.fit() di notebook
FEATURE_ORDER = [
    "BALANCE_log", "BALANCE_FREQUENCY", "PURCHASES_log", "ONEOFF_PURCHASES_log",
    "INSTALLMENTS_PURCHASES_log", "CASH_ADVANCE_log", "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT_log", "PAYMENTS_log", "MINIMUM_PAYMENTS_log",
    "PRC_FULL_PAYMENT", "TENURE"
]

# Kolom mentah yang wajib ada di file CSV yang di-upload
RAW_COLUMNS = [
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"
]

# Metrik utama yang ditampilkan pada grafik ringkasan
KEY_METRICS = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]


def preprocess(df_raw: pd.DataFrame) -> np.ndarray:
    """Menerapkan pipeline preprocessing yang sama persis dengan notebook
    (log transform fitur skewed -> susun sesuai FEATURE_ORDER -> scaling)."""
    df = df_raw.copy()

    for col in SKEWED_FEATURES:
        df[f"{col}_log"] = np.log1p(df[col])

    X = df[FEATURE_ORDER].values
    X_scaled = scaler.transform(X)
    return X_scaled


def predict(df_raw: pd.DataFrame) -> pd.DataFrame:
    X_scaled = preprocess(df_raw)
    clusters = model.predict(X_scaled)
    result = df_raw.copy()
    result["Cluster"] = clusters
    result["Segmen"] = result["Cluster"].map(CLUSTER_LABELS)
    return result


# ==========================================================
# Header
# ==========================================================
st.title("Segmentasi Nasabah Kartu Kredit Berdasarkan Pola Penggunaan")
st.markdown(
    "Aplikasi ini memprediksi segmen nasabah kartu kredit menggunakan model "
    "**K-Means (K=4)** yang dilatih pada dataset "
    "[Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)."
)

tab_manual, tab_csv, tab_info = st.tabs(["Input Manual", "Upload CSV", "Tentang Segmen"])

# ==========================================================
# TAB 1: Input Manual
# ==========================================================
with tab_manual:
    st.subheader("Masukkan Data Nasabah")
    st.caption("Cukup isi 6 data utama di bawah — sisanya sudah diisi otomatis dengan nilai wajar.")

    col1, col2 = st.columns(2)

    with col1:
        balance = st.number_input("Saldo Nasabah (BALANCE)", min_value=0.0, value=1500.0, step=100.0)
        purchases = st.number_input("Total Pembelian (PURCHASES)", min_value=0.0, value=800.0, step=50.0)
        cash_advance = st.number_input("Total Tarik Tunai (CASH_ADVANCE)", min_value=0.0, value=0.0, step=50.0)

    with col2:
        credit_limit = st.number_input("Limit Kartu Kredit (CREDIT_LIMIT)", min_value=0.0, value=5000.0, step=100.0)
        payments = st.number_input("Total Pembayaran (PAYMENTS)", min_value=0.0, value=1200.0, step=100.0)
        tenure = st.number_input("Lama Jadi Nasabah / TENURE (bulan)", min_value=1, max_value=12, value=12, step=1)

    with st.expander("Pengaturan Lanjutan (opsional, sudah ada nilai default)"):
        st.caption("Field ini jarang dipakai untuk keputusan segmentasi utama — biarkan default kalau tidak yakin.")
        adv1, adv2, adv3 = st.columns(3)

        with adv1:
            balance_frequency = st.slider("BALANCE_FREQUENCY", 0.0, 1.0, 0.9)
            oneoff_purchases = st.number_input("ONEOFF_PURCHASES", min_value=0.0, value=purchases * 0.5, step=50.0)
            installments_purchases = st.number_input("INSTALLMENTS_PURCHASES", min_value=0.0, value=purchases * 0.5, step=50.0)

        with adv2:
            purchases_frequency = st.slider("PURCHASES_FREQUENCY", 0.0, 1.0, 0.8)
            oneoff_purchases_frequency = st.slider("ONEOFF_PURCHASES_FREQUENCY", 0.0, 1.0, 0.3)
            purchases_installments_frequency = st.slider("PURCHASES_INSTALLMENTS_FREQUENCY", 0.0, 1.0, 0.5)

        with adv3:
            cash_advance_frequency = st.slider("CASH_ADVANCE_FREQUENCY", 0.0, 1.0, 0.0 if cash_advance == 0 else 0.3)
            cash_advance_trx = st.number_input("CASH_ADVANCE_TRX", min_value=0, value=0, step=1)
            purchases_trx = st.number_input("PURCHASES_TRX", min_value=0, value=15, step=1)

        minimum_payments = st.number_input("MINIMUM_PAYMENTS", min_value=0.0, value=payments * 0.25, step=50.0)
        prc_full_payment = st.slider("PRC_FULL_PAYMENT", 0.0, 1.0, 0.2)

    if st.button("Prediksi Segmen", type="primary"):
        input_df = pd.DataFrame([{
            "BALANCE": balance,
            "BALANCE_FREQUENCY": balance_frequency,
            "PURCHASES": purchases,
            "ONEOFF_PURCHASES": oneoff_purchases,
            "INSTALLMENTS_PURCHASES": installments_purchases,
            "CASH_ADVANCE": cash_advance,
            "PURCHASES_FREQUENCY": purchases_frequency,
            "ONEOFF_PURCHASES_FREQUENCY": oneoff_purchases_frequency,
            "PURCHASES_INSTALLMENTS_FREQUENCY": purchases_installments_frequency,
            "CASH_ADVANCE_FREQUENCY": cash_advance_frequency,
            "CASH_ADVANCE_TRX": cash_advance_trx,
            "PURCHASES_TRX": purchases_trx,
            "CREDIT_LIMIT": credit_limit,
            "PAYMENTS": payments,
            "MINIMUM_PAYMENTS": minimum_payments,
            "PRC_FULL_PAYMENT": prc_full_payment,
            "TENURE": tenure,
        }])

        result = predict(input_df)
        cluster_id = int(result.loc[0, "Cluster"])
        segmen = result.loc[0, "Segmen"]
        warna = CLUSTER_COLORS[segmen]

        st.success(f"Segmen: {segmen} (Cluster {cluster_id})")
        st.info(CLUSTER_DESC[cluster_id])

        st.markdown("#### Ringkasan Metrik Utama Nasabah")
        metric_df = pd.DataFrame({
            "Metrik": KEY_METRICS,
            "Nilai": input_df[KEY_METRICS].iloc[0].values
        })

        fig_manual = px.bar(
            metric_df, x="Metrik", y="Nilai",
            text="Nilai", title=f"Metrik Utama — {segmen}",
            color_discrete_sequence=[warna]
        )
        fig_manual.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_manual.update_layout(yaxis_title="Nilai", xaxis_title="", showlegend=False)
        st.plotly_chart(fig_manual, use_container_width=True)

# ==========================================================
# TAB 2: Upload CSV
# ==========================================================
with tab_csv:
    st.subheader("Prediksi Banyak Nasabah Sekaligus")
    st.caption(
        "File CSV harus memiliki kolom berikut (nama & urutan boleh berbeda, "
        "yang penting nama kolom sama persis): "
        + ", ".join(RAW_COLUMNS)
    )

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            missing_cols = [c for c in RAW_COLUMNS if c not in df_upload.columns]

            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan di file: {', '.join(missing_cols)}")
            else:
                result = predict(df_upload)
                st.success(f"Berhasil memprediksi segmen untuk {len(result)} nasabah.")

                st.dataframe(result, use_container_width=True)

                csv_out = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Hasil Prediksi (CSV)",
                    data=csv_out,
                    file_name="hasil_segmentasi_nasabah.csv",
                    mime="text/csv"
                )

                st.divider()
                st.markdown("### Visualisasi Hasil Segmentasi")

                counts = result["Segmen"].value_counts().reset_index()
                counts.columns = ["Segmen", "Jumlah"]

                chart_col1, chart_col2 = st.columns(2)

                # --- Distribusi jumlah nasabah per segmen (bar interaktif) ---
                with chart_col1:
                    fig_count = px.bar(
                        counts, x="Segmen", y="Jumlah", color="Segmen",
                        text="Jumlah", color_discrete_map=CLUSTER_COLORS,
                        title="Jumlah Nasabah per Segmen"
                    )
                    fig_count.update_traces(textposition="outside")
                    fig_count.update_layout(showlegend=False, xaxis_title="", yaxis_title="Jumlah Nasabah")
                    st.plotly_chart(fig_count, use_container_width=True)

                # --- Proporsi segmen (donut interaktif) ---
                with chart_col2:
                    fig_donut = px.pie(
                        counts, names="Segmen", values="Jumlah", hole=0.45,
                        color="Segmen", color_discrete_map=CLUSTER_COLORS,
                        title="Proporsi Segmen Nasabah"
                    )
                    fig_donut.update_traces(textinfo="percent+label")
                    st.plotly_chart(fig_donut, use_container_width=True)

                # --- Profil rata-rata tiap segmen (radar chart) ---
                st.markdown("**Profil Rata-rata Tiap Segmen (Radar Chart)**")
                avg_by_cluster = result.groupby("Segmen")[KEY_METRICS].mean()
                # Normalisasi 0-1 per metrik agar skala antar metrik sebanding pada radar
                norm = (avg_by_cluster - avg_by_cluster.min()) / (avg_by_cluster.max() - avg_by_cluster.min() + 1e-9)

                fig_radar = go.Figure()
                for segmen in norm.index:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=norm.loc[segmen].values.tolist() + [norm.loc[segmen].values[0]],
                        theta=KEY_METRICS + [KEY_METRICS[0]],
                        fill="toself",
                        name=segmen,
                        line_color=CLUSTER_COLORS.get(segmen, "#888888")
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Perbandingan Karakteristik Segmen (nilai dinormalisasi)"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # --- Sebaran Balance vs Purchases per segmen (scatter interaktif) ---
                st.markdown("**Sebaran Saldo vs Pembelian per Segmen**")
                fig_scatter = px.scatter(
                    result, x="BALANCE", y="PURCHASES", color="Segmen",
                    color_discrete_map=CLUSTER_COLORS, opacity=0.65,
                    hover_data=["CREDIT_LIMIT", "PAYMENTS", "TENURE"],
                    title="Saldo vs Pembelian per Nasabah"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

# ==========================================================
# TAB 3: Info Segmen
# ==========================================================
with tab_info:
    st.subheader("Penjelasan Tiap Segmen Nasabah")
    for cid, label in CLUSTER_LABELS.items():
        st.markdown(f"**Cluster {cid} — {label}**")
        st.write(CLUSTER_DESC[cid])
        st.divider()