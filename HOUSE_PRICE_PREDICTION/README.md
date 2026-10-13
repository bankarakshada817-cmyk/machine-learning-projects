# 🏠 House Price Prediction System

## About the Project
This project predicts estimated house prices using Machine Learning. It provides a simple and interactive web interface built with Streamlit.

## Features
- Predict estimated house prices
- Enter property details such as location, BHK, size and property age
- Consider locality features and amenities
- Interactive and user-friendly interface

## Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- Pickle

## Project Files
- `app.py` – Main Streamlit application
- `house_price_model.pkl` – Trained prediction model
- `vectorizer.pkl` – Feature vectorizer
- `encoder.pkl` – Categorical feature encoder
- `features.pkl` – Stored feature information

## Installation

Install the required libraries:

```bash
python -m pip install streamlit pandas scikit-learn
```

## How to Run

Run the following command in the project directory:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.

## Author
Created as a Machine Learning project for house price prediction.