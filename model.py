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

    # Fit Isolation Forest
    clf = IsolationForest(contamination=0.01, random_state=42)
    features["anomaly"] = clf.fit_predict(features)
    features["anomaly"] = features["anomaly"].map({1: 0, -1: 1})

    # Filter df to align with features
    df = df.dropna(subset=["mass (g)", "reclat", "reclong", "GeoLocation"])
    df = df[df['year'].notnull()]
    df = df[df['year'] > 1000]
    df['year'] = df['year'].astype(int)

    # Add anomaly labels back to df
    df["anomaly"] = features["anomaly"].values

    # Separate normal and anomalous data
    normal = df[df["anomaly"] == 0]
    anomalies = df[df["anomaly"] == 1]

    # --- Start Plot ---
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-whitegrid')  # clean background

    # Plot normal points
    plt.scatter(normal['year'], normal['mass (g)'],
                color='blue', label='Normal', alpha=0.4)

    # Plot anomalies
    plt.scatter(anomalies['year'], anomalies['mass (g)'],
                color='red', label='Anomaly', alpha=0.7)

    # Label top 5 anomalies by mass
    for i, row in anomalies.nlargest(5, 'mass (g)').iterrows():
        plt.text(row['year'], row['mass (g)'], row['name'],
                 fontsize=8, color='darkred', alpha=0.8)

    # Log scale for better mass visibility
    plt.yscale('log')

    # Titles and labels
    plt.xlabel('Year')
    plt.ylabel('Mass (g)')
    plt.title('Meteorite Mass vs Year (Red = Anomalies)', fontsize=14)
    plt.legend()

    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.3)

    # Annotate number of anomalies
    plt.annotate(f"Anomalies: {len(anomalies)}",
                 xy=(0.75, 0.95), xycoords='axes fraction',
                 fontsize=12, color='red', fontweight='bold')

    # Save plot
    plt.savefig("enhanced_anomaly_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Optional: Export anomalies to CSV
    anomalies.to_csv("detected_anomalies.csv", index=False)
