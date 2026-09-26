import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "KNN_reg_outlet_sales - KNN_reg_outlet_sales.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nOriginal Columns:")
print(df.columns.tolist())


# ============================================================
# 2. REMOVE UNNECESSARY COLUMN
# ============================================================

# Item_Identifier is a product ID, not useful for KNN prediction
df = df.drop("Item_Identifier", axis=1)


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

# Fill numerical missing values
df["Item_Weight"] = df["Item_Weight"].fillna(
    df["Item_Weight"].median()
)

# Fill categorical missing values
df["Outlet_Size"] = df["Outlet_Size"].fillna(
    df["Outlet_Size"].mode()[0]
)


# ============================================================
# 4. CONVERT CATEGORICAL COLUMNS
# ============================================================

categorical_columns = [
    "Item_Fat_Content",
    "Item_Type",
    "Outlet_Identifier",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)


# ============================================================
# 5. CONVERT BOOLEAN TO INTEGER
# ============================================================

boolean_columns = df.select_dtypes(
    include=["bool"]
).columns

for column in boolean_columns:
    df[column] = df[column].astype(int)


# ============================================================
# 6. REMOVE ANY REMAINING MISSING VALUES
# ============================================================

df = df.dropna()


# ============================================================
# 7. FEATURES AND TARGET
# ============================================================

X = df.drop(
    "Item_Outlet_Sales",
    axis=1
)

y = df["Item_Outlet_Sales"]


print("\nFinal Features:")
print(X.columns.tolist())

print("\nNumber of Features:", X.shape[1])


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ============================================================
# 9. CREATE KNN REGRESSION MODEL
# ============================================================

model = KNeighborsRegressor(
    n_neighbors=5
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\nTraining KNN model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ============================================================
# 11. SAVE MODEL
# ============================================================

with open(
    "knn_regression_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# ============================================================
# 12. SAVE TRAINING COLUMNS
# ============================================================

with open(
    "training_columns.pkl",
    "wb"
) as file:

    pickle.dump(
        X.columns.tolist(),
        file
    )


# ============================================================
# 13. MODEL SCORE
# ============================================================

score = model.score(
    X_test,
    y_test
)

print("\n==========================================")
print("       KNN MODEL CREATED SUCCESSFULLY")
print("==========================================")

print("\nR² Score:", round(score, 4))

print("\nCreated Files:")
print("1. knn_regression_model.pkl")
print("2. training_columns.pkl")

print("\nDone!")