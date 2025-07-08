from data_preprocessing import preprocess_data, load_data
from model import model, plot_anomalies

anomalies = model()
print(anomalies)

plot_anomalies()
anomalies.to_csv("anomalies.csv", index=False)