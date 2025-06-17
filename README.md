# 🩺 Disease Prediction Web App

This is a Machine Learning-based **Disease Prediction Tool** built using **Streamlit**. It takes common health metrics as input and predicts the likelihood of diseases such as **Diabetes**, **Heart Disease** or **No Disease** using a trained Random Forest model.

---

## 🚀 Features

- 🧠 Predicts disease based on health parameters
- 📊 Shows top 3 probable outcomes with confidence scores
- 🌐 Built with Streamlit — works directly in the browser
- 🎯 Custom UI with dark theme and real-time form input
- 💾 Trained on synthetic data with labeled encoders

---

## 💻 Demo

Live App: [Click here to use the app](https://diseasepredictionnabadiganta.streamlit.app/)  

---

## 📁 Project Structure

```
disease-prediction-app/
├── app.py                 # Streamlit application
├── model.pkl              # Trained ML model
├── encoders.pkl           # Label encoders for categorical features
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 💻 Run Locally

1. Clone the repository:

```bash
git clone https://github.com/yourusername/disease-prediction-app.git
cd disease-prediction-app
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## ⚙️ Requirements

- Python 3.7+
- streamlit
- pandas
- scikit-learn
- joblib

If you’re using additional visualization (like `plotly`), make sure to include that in `requirements.txt`:

```bash
pip install plotly
```

---


## ⚠️ Disclaimer

This project is for **educational/demo purposes only** and is **not** intended for real-world medical use. Always consult a professional for medical diagnosis.

---

## 📫 Author

Made by [Nabadiganta Acharjee]  
GitHub: [github.com/naba-diganta03](https://github.com/naba-diganta03)
