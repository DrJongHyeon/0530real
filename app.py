import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
from streamlit_folium import st_folium

# 색상 매핑 (Folium에서 지원하는 기본 색상)
cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'pink', 'black', 'gray']

# 데이터 로드
df = pd.read_csv("Delivery.csv")

# KMeans 군집
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])

# --- Plotly 시각화 ---
st.title("📍 배달 위치 군집 분석")
st.subheader("📊 Plotly 기반 산점도")

fig = px.scatter_mapbox(
    df,
    lat='Latitude',
    lon='Longitude',
    color='Cluster',
    hover_name='Num',
    zoom=10,
    height=600,
    mapbox_style='carto-positron'
)
st.plotly_chart(fig)

# --- Folium 지도 시각화 ---
st.subheader("🗺️ Folium 기반 지도 시각화")

m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)

# 각 포인트에 마커 추가 (군집 색상 적용)
for idx, row in df.iterrows():
    cluster_id = row['Cluster']
    color = cluster_colors[cluster_id % len(cluster_colors)]
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Num: {row['Num']}<br>Cluster: {cluster_id}",
        icon=folium.Icon(color=color)
    ).add_to(marker_cluster)

# 클러스터 중심점 추가
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

# 지도 렌더링
st_data = st_folium(m, width=700, height=500)
