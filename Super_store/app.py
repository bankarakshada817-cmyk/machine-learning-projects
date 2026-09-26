from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "knn_regression_model.pkl"
)

COLUMNS_PATH = os.path.join(
    BASE_DIR,
    "training_columns.pkl"
)


with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


with open(COLUMNS_PATH, "rb") as file:
    training_columns = pickle.load(file)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # GET VALUES FROM FORM
        # ----------------------------------------------------

        item_weight = float(
            request.form["item_weight"]
        )

        item_visibility = float(
            request.form["item_visibility"]
        )

        item_mrp = float(
            request.form["item_mrp"]
        )

        outlet_establishment_year = int(
            request.form["outlet_establishment_year"]
        )

        item_fat_content = request.form[
            "item_fat_content"
        ]

        item_type = request.form[
            "item_type"
        ]

        outlet_identifier = request.form[
            "outlet_identifier"
        ]

        outlet_size = request.form[
            "outlet_size"
        ]

        outlet_location_type = request.form[
            "outlet_location_type"
        ]

        outlet_type = request.form[
            "outlet_type"
        ]


        # ----------------------------------------------------
        # CREATE EMPTY DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame(
    0.0,
    index=[0],
    columns=training_columns
)

        # ----------------------------------------------------
        # NUMERICAL FEATURES
        # ----------------------------------------------------

        if "Item_Weight" in input_data.columns:
            input_data.loc[0, "Item_Weight"] = item_weight

        if "Item_Visibility" in input_data.columns:
            input_data.loc[0, "Item_Visibility"] = item_visibility

        if "Item_MRP" in input_data.columns:
            input_data.loc[0, "Item_MRP"] = item_mrp

        if "Outlet_Establishment_Year" in input_data.columns:
            input_data.loc[
                0,
                "Outlet_Establishment_Year"
            ] = outlet_establishment_year


        # ----------------------------------------------------
        # ITEM FAT CONTENT
        # ----------------------------------------------------

        fat_column = (
            "Item_Fat_Content_"
            + item_fat_content
        )

        if fat_column in input_data.columns:
            input_data.loc[
                0,
                fat_column
            ] = 1


        # ----------------------------------------------------
        # ITEM TYPE
        # ----------------------------------------------------

        item_column = (
            "Item_Type_"
            + item_type
        )

        if item_column in input_data.columns:
            input_data.loc[
                0,
                item_column
            ] = 1


        # ----------------------------------------------------
        # OUTLET IDENTIFIER
        # ----------------------------------------------------

        outlet_column = (
            "Outlet_Identifier_"
            + outlet_identifier
        )

        if outlet_column in input_data.columns:
            input_data.loc[
                0,
                outlet_column
            ] = 1


        # ----------------------------------------------------
        # OUTLET SIZE
        # ----------------------------------------------------

        size_column = (
            "Outlet_Size_"
            + outlet_size
        )

        if size_column in input_data.columns:
            input_data.loc[
                0,
                size_column
            ] = 1


        # ----------------------------------------------------
        # OUTLET LOCATION
        # ----------------------------------------------------

        location_column = (
            "Outlet_Location_Type_"
            + outlet_location_type
        )

        if location_column in input_data.columns:
            input_data.loc[
                0,
                location_column
            ] = 1


        # ----------------------------------------------------
        # OUTLET TYPE
        # ----------------------------------------------------

        type_column = (
            "Outlet_Type_"
            + outlet_type
        )

        if type_column in input_data.columns:
            input_data.loc[
                0,
                type_column
            ] = 1


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]

        prediction = round(
            float(prediction),
            2
        )


        # ----------------------------------------------------
        # SHOW RESULT
        # ----------------------------------------------------

        return render_template(
            "index.html",
            prediction=prediction
        )


    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )