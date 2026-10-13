from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load Dataset & Train K-NN Model
try:
    df = pd.read_csv('Iris.csv')
except FileNotFoundError:
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    )
    df['Species'] = [iris.target_names[i] for i in iris.target]

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    active_tab = 'prediction'  # Default active tab

    if request.method == 'POST':
        active_tab = request.form.get('active_tab', 'prediction')
        
        if active_tab == 'prediction':
            try:
                sl = float(request.form['sepal_length'])
                sw = float(request.form['sepal_width'])
                pl = float(request.form['petal_length'])
                pw = float(request.form['petal_width'])
                
                input_data = np.array([[sl, sw, pl, pw]])
                prediction = model.predict(input_data)[0]
            except Exception as e:
                prediction = f"Error in inputs: {e}"

    # Prepare summary data for the Dashboard tab view
    sample_data = df.head(10).to_html(classes='table table-striped', index=False)
    species_counts = df['Species'].value_counts().to_dict()

    return render_template(
        'index.html',
        prediction=prediction,
        active_tab=active_tab,
        sample_data=sample_data,
        species_counts=species_counts,
        min_sl=float(df['SepalLengthCm'].min()), max_sl=float(df['SepalLengthCm'].max()),
        min_sw=float(df['SepalWidthCm'].min()), max_sw=float(df['SepalWidthCm'].max()),
        min_pl=float(df['PetalLengthCm'].min()), max_pl=float(df['PetalLengthCm'].max()),
        min_pw=float(df['PetalWidthCm'].min()), max_pw=float(df['PetalWidthCm'].max())
    )

if __name__ == '__main__':
    app.run(debug=True)