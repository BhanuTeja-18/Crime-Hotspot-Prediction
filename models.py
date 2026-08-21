from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split

def run_kmeans(X):
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    return labels, score

def run_classification(X):
    y = [1 if x > X['hour'].mean() else 0 for x in X['hour']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()

    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    dt_pred = dt.predict(X_test)
    rf_pred = rf.predict(X_test)

    return {
        "Decision Tree Accuracy": accuracy_score(y_test, dt_pred),
        "Random Forest Accuracy": accuracy_score(y_test, rf_pred)
    }