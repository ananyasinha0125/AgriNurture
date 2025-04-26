import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# -------------------- Crop Data --------------------
crop_data = pd.read_csv('crop_data.csv')
crop_data.columns = crop_data.columns.str.strip().str.lower()

crop_features = crop_data[['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']]
crop_labels = crop_data['label']

X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(
    crop_features, crop_labels, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier()
rf_model.fit(X_train_crop, y_train_crop)
crop_preds = rf_model.predict(X_test_crop)

print("Random Forest Accuracy for Crop Data:", accuracy_score(y_test_crop, crop_preds))

# Save crop model
joblib.dump(rf_model, 'models/crop_random_forest_model.pkl')


# -------------------- Fertilizer Data --------------------
fertilizers_data = pd.read_csv('fertilizers.csv')
fertilizers_data.columns = fertilizers_data.columns.str.strip().str.lower()

# Debug: Print column names
print("Fertilizer columns:", fertilizers_data.columns.tolist())

# Encode 'soil type'
le_soil = LabelEncoder()
fertilizers_data['soil type'] = le_soil.fit_transform(fertilizers_data['soil type'])

# Prepare features and labels
fertilizer_features = fertilizers_data[['temperature', 'humidity', 'moisture', 'n', 'p', 'k', 'soil type']]
fertilizer_labels = fertilizers_data['fertilizer']

X_train_fert, X_test_fert, y_train_fert, y_test_fert = train_test_split(
    fertilizer_features, fertilizer_labels, test_size=0.2, random_state=42)

ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
ann_model.fit(X_train_fert, y_train_fert)
fertilizer_preds = ann_model.predict(X_test_fert)

print("ANN Accuracy for Fertilizer Data:", accuracy_score(y_test_fert, fertilizer_preds))

# Save fertilizer model and label encoder
joblib.dump(ann_model, 'models/fertilizer_ann_model.pkl')
joblib.dump(le_soil, 'models/soil_label_encoder.pkl')
