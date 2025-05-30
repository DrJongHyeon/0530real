import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
from streamlit_folium import st_folium

# 색상 매핑 (Folium에서 지원하는 색상 중 최대 10개)
cluster_colors = [
    'red', 'blue', 'green', 'purple', 'orange',
    'darkred', 'cadetblue', 'pink', 'black', 'gray'
]

# 타이틀 및 설명
st.title("📍 배달 위치 군집 분석")
st.markdown("위도/경도 데이터를 기반으로 KMeans 클러스터링 및 지도 시각화를 수행합니다.")

# 데이터 로드
try:
    df = pd.read_csv("Delivery.csv")
except FileNotFoundError:
    st.error("⚠️ 'Delivery.csv' 파일이 현재 디렉토리에 없습니다.")
    st.stop()

# 필수 컬럼 확인
required_cols = {'Latitude', 'Longitude'}
if not required_cols.issubset(df.columns):
    st.error("⚠️ CSV 파일에 'Latitude'와 'Longitude' 컬럼이 있어야 합니다.")
    st.stop()

# NaN 제거
df = df.dropna(subset=['Latitude', 'Longitude'])

# 유효한 데이터 존재 확인
if df.empty:
    st.error("⚠️ 유효한 위도/경도 데이터를 가진 행이 없습니다.")
    st.stop()

# KMeans 클러스터링
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])

# -------------------------------
# Plotly 시각화
# -------------------------------
st.subheader("📊 Plotly 기반 클러스터 시각화")

fig = px.scatter_mapbox(
    df,
    lat='Latitude',
    lon='Longitude',
    color='Cluster',
    hover_name='Num' if 'Num' in df.columns else None,
    zoom=10,
    height=600,
    mapbox_style='carto-positron'
)
st.plotly_chart(fig)

# -------------------------------
# Folium 지도 시각화
# -------------------------------
st.subheader("🗺️ Folium 기반 지도 시각화")

# 평균 좌표
avg_lat = df['Latitude'].mean()
avg_lon = df['Longitude'].mean()

# 지도 생성
m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)

# 각 마커 추가
for _, row in df.iterrows():
    cluster_id = int(row['Cluster'])
    color = cluster_colors[cluster_id % len(cluster_colors)]
    popup_text = f"Cluster: {cluster_id}"
    if 'Num' in df.columns:
        popup_text = f"Num: {row['Num']}<br>Cluster: {cluster_id}"
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup_text,
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
        popup=f"Cluster Center {i}"
    ).add_to(m)

# 지도 출력
st_data = st_folium(m, width=700, height=500)
