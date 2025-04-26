import streamlit as st
import pandas as pd
import numpy as np

from app1 import load_crop_data

# Example data for testing (replace with actual data files)
# Simulated crop data for demonstration
crop_data = pd.DataFrame({
    "label": ["wheat", "rice", "maize"],
    "n": [90, 110, 100],
    "p": [60, 50, 40],
    "k": [40, 40, 50],
    "ph": [6.5, 6.0, 6.8],
    "moisture": [50, 70, 60],
    "temperature": [25, 28, 30],
    "humidity": [80, 85, 75],
    "rainfall": [200, 250, 150]
})

# Simulated soil data (to be replaced with actual slider inputs)
soil_data = {
    "n": 70,
    "p": 30,
    "k": 20,
    "ph": 6.0,
    "moisture": 40,
    "temperature": 24,
    "humidity": 75,
    "rainfall": 150
}

# Function to compare soil data with ideal crop requirements
def compare_parameters_with_crop_requirements(soil_data, crop_name, crop_data):
    crop_info = crop_data[crop_data["label"] == crop_name].mean(numeric_only=True)
    comparison = {}

    for param in ["n", "p", "k", "ph", "moisture", "temperature", "humidity", "rainfall"]:
        if param in soil_data and param in crop_info.index:
            difference = soil_data[param] - crop_info[param]
            comparison[param] = difference
        else:
            comparison[param] = None
    return comparison

# Function to recommend fertilizers based on deficiencies
def recommend_fertilizers(soil_data, crop_name, crop_data):
    crop_info = crop_data[crop_data["label"] == crop_name].mean(numeric_only=True)
    recommendations = []

    for param in ["n", "p", "k"]:
        if param in soil_data and param in crop_info.index:
            difference = crop_info[param] - soil_data[param]
            if difference > 0:  # Deficiency
                recommendations.append(f"{param.upper()}: Add {difference:.2f} units.")
    
    return recommendations if recommendations else ["No fertilizer adjustments needed."]
def show_analysis_page():
    st.title("Soil and Fertilizer Data Analysis")

    # Load Crop Data
    crop_data = load_crop_data()  # Ensure `crop_data.csv` exists in your directory

    if crop_data is not None:
        st.title("Crop and Soil Data Summary:")
        st.dataframe(crop_data)

        # Visualize Ideal Soil Parameters for Selected Crop
        st.subheader("Visualize Ideal Parameters for Crops")
        crop_to_compare = st.selectbox("Select Crop to Analyze", crop_data["label"].unique())

        if crop_to_compare:
            comparison = compare_parameters_with_crop_requirements(soil_data, crop_to_compare, crop_data)  # Ensure this function is adjusted for fertilizer
            st.write(f"**Comparison for {crop_to_compare}:**")
            
            for param, diff in comparison.items():
                if diff is not None:
                    st.write(f"{param.capitalize()}: {diff:.2f} (Adjust {'Up' if diff < 0 else 'Down'})")

# Streamlit app layout
st.title("Fertilizer Recommendation System")

# Input: Crop Type
crop_type = st.selectbox("Select Crop Type", options=crop_data["label"].unique())
# Input: Location
location = st.text_input("Enter Location (City/Town/Region)", "Example: Ghaziabad")

# Process recommendations
if st.button("Recommend Fertilizers") and crop_type:
    # Compare soil parameters with crop requirements
    comparison = compare_parameters_with_crop_requirements(soil_data, crop_type, crop_data)
    st.subheader(f"Adjustments for {crop_type}:")
    for param, diff in comparison.items():
        if diff is not None and abs(diff) > 0:
            adjustment = "Increase" if diff < 0 else "Decrease"
            st.write(f"- {param.capitalize()}: {adjustment} by {abs(diff):.2f} units.")

    # Recommend fertilizers
    st.subheader("Fertilizer Recommendations:")
    fertilizer_recommendations = recommend_fertilizers(soil_data, crop_type, crop_data)
    for rec in fertilizer_recommendations:
        st.write(f"- {rec}")