from preprocess import load_data, preprocess
from models import run_kmeans, run_classification

df = load_data("data/crime_data.csv")
X = preprocess(df)

labels, score = run_kmeans(X)
print("Silhouette Score:", score)

results = run_classification(X)
print(results)