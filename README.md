# Live on : 
https://wine-streamlit-ml-latest-2.onrender.com/
🍷 Wine Quality Prediction – ML + Docker + CI/CD

A **production-style Machine Learning project** that predicts wine quality using multiple ML models, deployed as an interactive **Streamlit web app**, fully **Dockerized** and integrated with **CI/CD using GitHub Actions**.

---

## 🚀 Project Overview

This project demonstrates how real-world ML applications are built and shipped in companies:

- Train multiple ML models
- Serve predictions via a web UI
- Containerize the app using Docker
- Automate builds using CI/CD pipelines

---

## 🧠 Machine Learning Models Used

The following models are trained and evaluated on the Wine Quality dataset:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

All models are saved as `.pkl` files and loaded dynamically inside the app.

---

## 🖥️ Web Application (Streamlit)

**Features:**
- User-friendly input form for wine features
- Model selection dropdown
- Real-time prediction
- Clean and responsive UI
- Runs consistently on any machine using Docker

---

## 🐳 Docker Integration

The application is fully containerized.

### Why Docker?
- No environment issues
- Same behavior on all systems
- Easy deployment to cloud platforms

### Run locally with Docker:
```bash
docker build -t wine-quality-ml .
docker run -p 8501:8501 wine-quality-ml
