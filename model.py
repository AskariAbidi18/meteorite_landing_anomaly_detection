from sklearn.ensemble import IsolationForest
from data_preprocessing import preprocess_data, load_data
import matplotlib.pyplot as plt

def model():
    # Load raw dataset
    df = load_data()

    # Preprocess features
    features = preprocess_data()

    # Train Isolation Forest
    clf = IsolationForest(contamination=0.01, random_state=42)
    features["anomaly"] = clf.fit_predict(features)
    features["anomaly"] = features["anomaly"].map({1: 0, -1: 1})

    # Apply same filtering to original df to match features shape
    df = df.dropna(subset=["mass (g)", "reclat", "reclong", "GeoLocation"])
    df = df[df['year'].notnull()]
    df = df[df['year'] > 1000]
    df['year'] = df['year'].astype(int)

    # Add anomaly labels
    df["anomaly"] = features["anomaly"].values

    # Return detected anomalies
    anomalies = df[df["anomaly"] == 1]
    return anomalies[["name", "mass (g)", "year", "reclat", "reclong"]]

def plot_anomalies():
    # Load and preprocess data
    df = load_data()
    features = preprocess_data()

    clf = IsolationForest(contamination=0.01, random_state=42)
    features["anomaly"] = clf.fit_predict(features)
    features["anomaly"] = features["anomaly"].map({1: 0, -1: 1})

    df = df.dropna(subset=["mass (g)", "reclat", "reclong", "GeoLocation"])
    df = df[df['year'].notnull()]
    df = df[df['year'] > 1000]
    df['year'] = df['year'].astype(int)
    df["anomaly"] = features["anomaly"].values

    normal = df[df["anomaly"] == 0]
    anomalies = df[df["anomaly"] == 1]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(normal['year'], normal['mass (g)'], color='blue', label='Normal', alpha=0.5)
    plt.scatter(anomalies['year'], anomalies['mass (g)'], color='red', label='Anomaly', alpha=0.8)
    plt.yscale('log')
    plt.xlabel('Year')
    plt.ylabel('Mass (g)')
    plt.title('Meteorite Mass vs Year (Red = Anomalies)')
    plt.legend()
    plt.show()
