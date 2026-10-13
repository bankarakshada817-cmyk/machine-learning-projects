
# Auto Analytics Dashboard

## About the Project

Auto Analytics Dashboard is a data analysis project built using Python and Streamlit. It helps users explore and understand automobile data through interactive filters, key performance indicators, and visualizations.

## Features

- View and explore automobile data.
- Filter cars by brand and fuel type.
- Display key metrics such as total cars, unique brands, average price, and average horsepower.
- Analyze price distribution and fuel type distribution.
- Compare average prices across brands.
- Explore the relationship between engine size and price.
- View correlations between numerical features.
- Check missing values and dataset information.
- Download the dataset as a CSV file.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook

## Project Structure

```text
Auto_Analytics_Dashboard/
│
├── app.py
├── autos_dataset.csv
├── requirements.txt
├── README.md
├── .gitignore
└── your_notebook.ipynb
```

Note: Replace `your_notebook.ipynb` with the actual name of your notebook file.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd Auto_Analytics_Dashboard
```

### 3. Install the Required Libraries

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Application

```bash
python -m streamlit run app.py
```

The dashboard will open in your web browser.

## Dataset

The application uses `autos_dataset.csv`. Keep this file in the same folder as `app.py`.

## Learning Outcomes

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Interactive dashboard development
- Working with Python data analysis libraries

## Author

AI & Data Science Student

## Future Improvements

- Add more interactive filters.
- Include additional automobile insights.
- Deploy the dashboard online.
