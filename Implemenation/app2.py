# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Fertilizer Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('Implemenation/crop_data.csv')
    return df

def train_model(df):
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return rf_model, X.columns

#Sample data for demo
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

def create_gauge_chart(value, title, max_value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_value/3], 'color': "lightgray"},
                {'range': [max_value/3, 2*max_value/3], 'color': "gray"},
                {'range': [2*max_value/3, max_value], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

def recommend_fertilizer(crop_type, location, soil_data):
    if soil_data["n"] < 20:
        return f"Use Urea fertilizer to improve nitrogen levels for {crop_type} in {location}."
    elif soil_data["ph"] < 6.0:
        return f"Use lime-based fertilizers to balance the pH for {crop_type} in {location}."
    else:
        return f"Use a balanced NPK fertilizer for {crop_type} in {location}."

def load_crop_data():
    try:
        crop_data = pd.read_csv("Implemenation/crop_data.csv")
        crop_data.columns = crop_data.columns.str.strip().str.lower()
        return crop_data
    except FileNotFoundError:
        st.error("Crop data file not found. Ensure 'crop_data.csv' exists.")
        return None

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

def recommend_soil_parameters(crop_type, location, crop_data):
    crop_info = crop_data[crop_data["label"].str.lower() == crop_type.lower()].mean(numeric_only=True)

    if crop_info.empty:
        return None, f"Data for {crop_type} not found. Please ensure crop data is available."
    
    # Recommended Soil Parameters
    recommendations = {
        "Nitrogen (N)": crop_info["n"],
        "Phosphorus (P)": crop_info["p"],
        "Potassium (K)": crop_info["k"],
        "pH Level": crop_info["ph"],
        "Moisture (%)": crop_info["moisture"],
        "Temperature (°C)": crop_info["temperature"],
        "Humidity (%)": crop_info["humidity"],
        "Rainfall (mm)": crop_info["rainfall"]
    }

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home","Soil Parameter Recommeder", "Fertilizer Recommendation", "Data Analysis"])

if page == "Home":
    st.title("AgriNurture: Fertilizer Recommendation System")
    st.write("Welcome to the AgriNurture platform.")
elif page == "Fertilizer Recommendation":
    st.title("Fertilizer Recommendation")
    crop_type = st.selectbox("Select Crop Type", ["Wheat", "Rice", "Maize", "Sugarcane", "Cotton", "Other"])
    location = st.text_input("Enter Location (City/Town/Region)", "Ghaziabad")
    if st.button("Recommend Fertilizer"):
        st.write(recommend_fertilizer(crop_type, location, soil_data))
elif page == "Soil Parameters Recommendation":
    st.title("Soil Parameters Recommendation")
    
    # Load Crop Data
    crop_data = load_crop_data()
    
    if crop_data is not None:
        # Input: Crop Type
        crop_type = st.selectbox("Select Crop Type", crop_data["label"].unique())
        
        # Input: Location
        location = st.text_input("Enter Location (City/Town/Region)", "Example: Ghaziabad")
        
        # Generate Recommendations
        if st.button("Get Soil Recommendations"):
            recommendations, error = recommend_soil_parameters(crop_type, location, crop_data)
            
            if error:
                st.error(error)
            else:
                st.success(f"Recommended Soil Parameters for {crop_type} in {location}:")
                for param, value in recommendations.items():
                    st.write(f"- **{param}:** {value:.2f}")
    else:
        st.write("Crop data is unavailable. Ensure `crop_data.csv` is present in the directory.")                
elif page == "Data Analysis":
    st.title("Data Analysis")
    crop_data = load_crop_data()
    if crop_data is not None:
        st.dataframe(crop_data)
def main():
 
    def show_home_page():
      st.write("\n"
             "    ### Features:\n"
             "    - Soil health assessment\n"
             "    - Suggesting soil parameters based on assessment\n"
             "    - Crop recommendations tailored to soil parameters\n"
             "    - Data visualization and analysis\n"
             "    - Machine learning-powered predictions\n"
             "    \n"
             "    ### How to use:\n"
             "    1. Navigate to the 'Fertilizer Recommendation' page to get personalized suggestions\n"
             "    2. Input your crop type and location\n"
             "    3. View the recommended fertilizers and detailed analysis\n"
             "    4. Explore data patterns in the 'Data Analysis' page\n"
             "    ")
        
    def show_recommendation_page():
       st.title("Fertilizer Recommendation")

    # Input: Crop Type
    crop_type = st.selectbox("Select Crop Type", ["Wheat", "Rice", "Maize", "Sugarcane", "Cotton", "Other"])

    # Input: Location
    location = st.text_input("Enter Location (City/Town/Region)", "Example: Ghaziabad")

    # Input Validation and Fertilizer Recommendation
    if st.button("Recommend Fertilizer"):
        optimal_fertilizer = recommend_fertilizer(crop_type, location, soil_data)  # Define your logic here
        st.title(f"**Optimal Fertilizer Recommendation:** {optimal_fertilizer}")
        st.write("Adjust soil parameters for better crop yield if needed.")
    else:
        st.write("Enter all required details and press the button to get a recommendation.")
def show_analysis_page():
    st.title("Soil and Fertilizer Data Analysis")

    # Load Crop Data
    crop_data = load_crop_data() 

    if crop_data is not None:
        st.title("Crop and Soil Data Summary:")
        st.dataframe(crop_data)

        # Visualization of Ideal Soil Parameters
        st.subheader("Visualize Ideal Parameters for Crops")
        crop_to_compare = st.selectbox("Select Crop to Analyze", crop_data["label"].unique())

        if crop_to_compare:
            comparison = compare_parameters_with_crop_requirements(soil_data, crop_to_compare, crop_data) 
            st.write(f"**Comparison for {crop_to_compare}:**")
            
            for param, diff in comparison.items():
                if diff is not None:
                    st.write(f"{param.capitalize()}: {diff:.2f} (Adjust {'Up' if diff < 0 else 'Down'})")

            # Visual Comparison
            st.subheader("Visualize Current vs. Ideal Soil Conditions")
            soil_values = [val for val in soil_data.values()]
            ideal_values = [
                crop_data[crop_data["label"] == crop_to_compare][col].mean() if col in crop_data.columns else None
                for col in soil_data.keys()
            ]

            valid_soil_values = [v for v, iv in zip(soil_values, ideal_values) if iv is not None]
            valid_ideal_values = [iv for iv in ideal_values if iv is not None]
            valid_params = [param for param, iv in zip(soil_data.keys(), ideal_values) if iv is not None]

            fig, ax = px.subplots()
            bar_width = 0.35
            index = np.arange(len(valid_params))
            ax.bar(index, valid_soil_values, bar_width, label="Current Soil")
            ax.bar(index + bar_width, valid_ideal_values, bar_width, label=f"Ideal for {crop_to_compare}")
            ax.set_xlabel("Soil Parameters")
            ax.set_ylabel("Values")
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels([param.capitalize() for param in valid_params], rotation=45)
            ax.legend()
            st.pyplot(fig)
    else:
        st.write("No data available for analysis. Ensure `crop_data.csv` is present.")
    
        
if __name__ == "__main__":
    main()