import joblib
import numpy as np
from sklearn.svm import SVC

# 4 Features: [CGPA, Projects, Internships, Resume_Score]
X = np.array([
    [8.5, 3, 2, 80],
    [6.0, 1, 0, 40],
    [9.0, 4, 3, 90],
    [5.5, 0, 0, 30],
    [7.2, 2, 1, 60],
    [8.0, 2, 1, 75],
    [6.5, 1, 0, 50],
    [9.5, 5, 2, 95],
    [5.0, 0, 0, 20],
    [7.8, 3, 2, 70]
])

# Target: 1 = Placed, 0 = Not Placed
y = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])

# Train SVC Model
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X, y)

# Overwrite existing model.pkl
joblib.dump(model, 'model.pkl')
print("✅ Model successfully trained with 4 features and saved to 'model.pkl'!")