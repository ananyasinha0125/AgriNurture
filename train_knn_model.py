import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the dataset
df = pd.read_csv('fertilizers.csv')

# Encode 'soil type' if it's a string
if df['soil type'].dtype == object:
    df['soil type'] = LabelEncoder().fit_transform(df['soil type'])

# Features and target
X = df[['temperature', 'humidity', 'moisture', 'soil type', 'n', 'p', 'k']]
y = df['fertilizer']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'fertilizer_knn_model.pkl')
