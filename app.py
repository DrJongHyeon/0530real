import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
from streamlit_folium import st_folium

# 설정
st.set_page_config(layout="centered")
st.title("📍 배달 위치 자동 클러스터링")

# CSV 파일 읽기
try:
    df = pd.read_csv("Delivery.csv")
except FileNotFoundError:
    st.error("⚠️ 'Delivery.csv' 파일이 필요합니다.")
    st.stop()

# 위도/경도 유효성 검사
if not {'Latitude', 'Longitude'}.issubset(df.columns):
    st.error("CSV 파일에 'Latitude'와 'Longitude' 컬럼이 존재해야 합니다.")
    st.stop()

df = df.dropna(subset=['Latitude', 'Longitude'])
if df.empty:
    st.error("유효한 위치 정보가 없습니다.")
    st.stop()

# 사용자가 K만 조절할 수 있도록 슬라이더 제공
k = st.slider("군집 수 (K)", min_value=2, max_value=10, value=5)

# KMeans 클러스터링
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])

# 클러스터 색상 지정
cluster_colors = [
    'red', 'blue', 'green', 'purple', 'orange',
    'darkred', 'cadetblue', 'pink', 'black', 'gray'
]

# 지도 생성
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)

# 마커 추가
for _, row in df.iterrows():
    cluster_id = int(row['Cluster'])
    color = cluster_colors[cluster_id % len(cluster_colors)]
    popup = f"Cluster: {cluster_id}"
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup,
        icon=folium.Icon(color=color)
    ).add_to(marker_cluster)

# 클러스터 중심 표시
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

# 지도 표시
st_folium(m, width=700, height=500)
