
# Meteorite Landing Anomaly Detection

## 🌌 Project Overview

This project focuses on detecting anomalies in meteorite landing data using machine learning techniques. By analyzing historical meteorite landing records, the model identifies unusual patterns that could indicate significant events or errors in data collection.

---

## 📦 Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AskariAbidi18/meteorite_landing_anomaly_detection.git
   cd meteorite_landing_anomaly_detection
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

To run the anomaly detection pipeline:

```bash
python src/run.py
```

This script will:

- Load and preprocess the meteorite landing dataset.
- Train an anomaly detection model.
- Output detected anomalies and save the results.

---

## 🧪 Jupyter Notebook

For an interactive exploration of the data and model:

1. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Open `notebooks/logic.ipynb` to follow along with the analysis.

---

## 📈 Visualizations

The project includes visual representations of detected anomalies:

- `anomaly_plot.png`: Initial anomaly detection results.
- `enhanced_anomaly_plot.png`: Improved visualization with additional insights.

---

## 🛠️ Dependencies

The project requires the following Python libraries:

- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a new Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> Made with 💫 by Askari Abidi
