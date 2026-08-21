import streamlit as st
from preprocess import load_data, preprocess
from models import run_kmeans, run_classification
import folium
from streamlit_folium import st_folium

st.title("Crime Hotspot Prediction")

df = load_data("data/crime_data.csv")
X = preprocess(df)

st.write("### Data", df)

labels, score = run_kmeans(X)
results = run_classification(X)

st.write("### K-Means Silhouette Score:", score)
st.write("### Classification Results", results)

# Map
m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)

for i, row in df.iterrows():
    color = "red" if labels[i] == 1 else "blue"
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=color,
        fill=True
    ).add_to(m)

st_folium(m, width=700, height=500)