# 🛒 Super Store Predictive Analytics

## 📌 Project Overview

**Super Store Predictive Analytics** is a Machine Learning project that analyzes Super Store sales data and predicts sales based on different product and outlet-related features.

The project includes data preprocessing, exploratory data analysis, feature engineering, machine learning model training, and an interactive Streamlit web application.

## 🎯 Objectives

* Analyze Super Store sales data
* Handle missing values and categorical data
* Perform data preprocessing
* Train a machine learning regression model
* Predict sales using user-provided features
* Build an interactive web application using Streamlit

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn
* Pickle

## 📊 Dataset

The project uses Super Store / Big Mart-style sales data containing information such as:

* Item Identifier
* Item Weight
* Item Fat Content
* Item Visibility
* Item Type
* Item MRP
* Outlet Identifier
* Outlet Establishment Year
* Outlet Size
* Outlet Location Type
* Outlet Type
* Item Outlet Sales

## 🤖 Machine Learning

The project follows these major steps:

1. Load the dataset
2. Explore the data
3. Handle missing values
4. Encode categorical features
5. Prepare input and target variables
6. Split the dataset into training and testing sets
7. Train the regression model
8. Evaluate the model
9. Save the trained model
10. Use the model for prediction through Streamlit

## 🌐 Streamlit Web Application

The Streamlit application provides an easy-to-use interface where users can enter product and outlet information and get a predicted sales value.

### Main Features

* 📝 User-friendly input form
* 📊 Sales prediction
* ⚡ Fast prediction
* 🎨 Interactive web interface
* 📈 Machine learning based results

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Super-Store-Predictive-Analytics.git
```

### 2. Navigate to the Project

```bash
cd Super-Store-Predictive-Analytics
```

### 3. Install Required Libraries

```bash
python -m pip install -r requirements.txt
```

### 4. Train the Model

```bash
python train_model.py
```

This will create the trained model file.

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📂 Project Files

| File               | Description                      |
| ------------------ | -------------------------------- |
| `app.py`           | Streamlit web application        |
| `train_model.py`   | Model training and preprocessing |
| `requirements.txt` | Required Python libraries        |
| `model.pkl`        | Trained machine learning model   |
| `data/`            | Dataset files                    |
| `screenshots/`     | Application screenshots          |

## 📈 Future Improvements

* Improve model accuracy
* Add multiple machine learning algorithms
* Add sales visualization dashboards
* Add model comparison
* Deploy the application online
* Add prediction history

## 👩‍💻 Author

**Akshada Bankar**

AI & Data Science Student

## ⭐ Acknowledgement

This project was developed as a practical Machine Learning and Predictive Analytics project using Python and Streamlit.
