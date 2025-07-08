import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("Meteorite_Landings.csv")
    return df

def preprocess_data():
    df = load_data()
    df = df.dropna(subset=["mass (g)","reclat","reclong","GeoLocation"])
    df = df[df['year'].notnull()]
    df = df[df['year'] > 1000]  # Filter out bad values like 0.0
    df['year'] = df['year'].astype(int)
    df["age"] = 2025 - df["year"]
    df['log_mass'] = np.log1p(df['mass (g)'])
    df["fall"] = df["fall"].map({"Fell": 1, "Found": 0})
    features = df[['log_mass', 'age', 'reclat', 'reclong', 'fall']]
    return features
