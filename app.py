import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
from streamlit_folium import st_folium
import plotly.express as px

# 기본 설정
st.set_page_config(layout="centered")
st.title("📍 배달 위치 클러스터링")

# 데이터 로드
try:
    df = pd.read_csv("Delivery.csv")
except FileNotFoundError:
    st.error("⚠️ 'Delivery.csv' 파일이 필요합니다.")
    st.stop()

# 필수 컬럼 확인
if not {'Latitude', 'Longitude'}.issubset(df.columns):
    st.error("CSV에 'Latitude'와 'Longitude' 컬럼이 필요합니다.")
    st.stop()

df = df.dropna(subset=['Latitude', 'Longitude'])
if df.empty:
    st.error("유효한 위치 정보가 없습니다.")
    st.stop()

# 클러스터 수 조정
k = st.slider("군집 수 (K)", 2, 10, 5)

# KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])

# --- Plotly: 클러스터 중심에 점이 따닥따닥 모인 정렬된 시각화 ---
st.subheader("🎯 클러스터 중심 기준 정렬 시각화 (위치 무관)")

fig2 = px.strip(
    df,
    x='Cluster',
    y='Latitude',  # 그냥 Y축은 뭔가 기준을 줘야 해서 하나 씀
    color='Cluster',
    orientation='v',
    stripmode='group'
)
fig2.update_layout(height=400, yaxis_title="(단순 시각화)", xaxis_title="클러스터")
st.plotly_chart(fig2)

# --- Folium 지도 시각화 ---
st.subheader("🗺️ 실제 지도 위치에 표시된 군집 마커")

# 색상 정의
cluster_colors = [
    'red', 'blue', 'green', 'purple', 'orange',
    'darkred', 'cadetblue', 'pink', 'black', 'gray'
]

m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)

for _, row in df.iterrows():
    cluster_id = int(row['Cluster'])
    color = cluster_colors[cluster_id % len(cluster_colors)]
    popup = f"Cluster: {cluster_id}"
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup,
        icon=folium.Icon(color=color)
    ).add_to(marker_cluster)

# 중심점도 표시
centroids = kmeans.cluster_centers_
for i, (lat, lon) in enumerate(centroids):
    folium.CircleMarker(
        location=[lat, lon],
        radius=10,
        color='black',
        fill=True,
        fill_color=cluster_colors[i % len(cluster_colors)],
        fill_opacity=0.7,
        popup=f"Center {i}"
    ).add_to(m)

st_folium(m, width=700, height=500)
