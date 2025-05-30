﻿# DocAppSentiment
# 🧠 Sentiment Analysis Web App

This is a simple web app to perform sentiment analysis on text input using a Naive Bayes classifier trained on a Twitter dataset. It classifies text as **positive** or **negative**.

Built using **Streamlit**, with support for **Docker** and **Render** deployment.

---

## 🚀 Features

- Input text through the web interface
- Predict sentiment using a trained model
- Displays results with color-coded feedback

---

## 🖥️ Run Locally (Streamlit)

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (if not already trained)
```bash
python main.py
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🐳 Run with Docker

### 1. Build the image
```bash
docker build -t sentiment-app .
```

### 2. Run the container
```bash
docker run -p 8501:8501 sentiment-app
```

The app will be available at [http://localhost:8501](http://localhost:8501)

---

## ☁️ Deploy to Render

### 1. Setup repository

Push your code (including `app.py`, `main.py`, `requirements.txt`, and `Dockerfile`) to GitHub.

### 2. Create a new Render Web Service

- Go to [https://render.com](https://render.com)
- Click **"New" → "Web Service"**
- Connect your GitHub repo
- Set the environment:
  - **Environment:** Docker
  - **Branch:** `main` (or your branch name)
  - **Port:** `8501`
- Click **"Create Web Service"**

Render will build and deploy the app automatically.

---

## 📁 File Structure

```
├── app.py                  # Streamlit UI
├── main.py                 # Trains and saves the model
├── predict.py              # (Optional) Old Flask API version
├── requirements.txt
├── Dockerfile
└── data/
    └── model.dat.gz        # Saved trained model
```

---

## 🧠 Model

- Dataset: Stanford Twitter Sentiment Corpus
- Algorithm: Multinomial Naive Bayes
- Features: Bag of Words using `CountVectorizer`

---

## 📜 License

MIT License – free to use, share, and modify.

```
