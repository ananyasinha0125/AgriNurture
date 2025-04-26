import pandas as pd
import streamlit as st
import numpy as np
import requests
import joblib
import plotly.graph_objects as go

# Load fertilizers.csv into a DataFrame
fertilizers_df = pd.read_csv("fertilizers.csv")

def get_fertilizer_recommendation(temperature, humidity, moisture, soil_type, n, p, k, model):
    soil_mapping = {"sandy": 0, "loamy": 1, "black": 2, "red": 3, "clayey": 4}
    input_soil_code = soil_mapping.get(soil_type.strip().lower(), -1)

    # Normalize numeric columns in the dataset
    numeric_cols = ['temperature', 'humidity', 'moisture', 'n', 'p', 'k']
    df_normalized = fertilizers_df.copy()

    # Normalize the columns if they exist
    for col in numeric_cols:
        if col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val + 1e-5)
        else:
            print(f"Warning: Column '{col}' not found in the dataset!")

    # Normalize input
    input_vals = {
        'temperature': temperature,
        'humidity': humidity,
        'moisture': moisture,
        'n': n,
        'p': p,
        'k': k
    }

    input_normalized = {}
    for col in numeric_cols:
        if col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            input_normalized[col] = (input_vals[col] - min_val) / (max_val - min_val + 1e-5)

    # Prepare input for prediction with the correct feature count (7 features)
    input_data = np.array([[
        input_normalized.get('temperature', 0),
        input_normalized.get('humidity', 0),
        input_normalized.get('moisture', 0),
        input_normalized.get('n', 0),
        input_normalized.get('p', 0),
        input_normalized.get('k', 0),
        input_soil_code
    ]])

    # Get fertilizer prediction
    fertilizer_prediction = model.predict(input_data)[0]
    return fertilizer_prediction

# 🔑 API Key
WEATHER_API_KEY = "ea234ca963df621a5a93413217dce69b"

# 🌐 Language setup
language = st.sidebar.selectbox("🌍 Choose Language / भाषा चुनें", ["English", "हिन्दी"])
def t(en, hi): return hi if language == "हिन्दी" else en

# 🌱 App title
st.markdown(f"""
    <div style="background-color:#7b6d3b; padding:10px; text-align:center;">
        <h1 style="color:white;">🌱 {t("AgriNature", "अग्रीनेचर")} 🌾</h1>
    </div>
""", unsafe_allow_html=True)

# 📍 Location from IP
def get_location_by_ip():
    try:
        res = requests.get("https://ipinfo.io/json")
        return res.json().get("city", "Delhi")
    except:
        return "Delhi"

# 🌦️ Weather data
def get_weather_data(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},India&appid={WEATHER_API_KEY}&units=metric"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "rainfall": round(np.random.uniform(80, 200), 2),
            "ph": round(np.random.uniform(5.5, 7.5), 1)
        }
    return None

# 🚜 Load models
@st.cache_resource
def load_crop_model():
    return joblib.load("models/crop_random_forest_model.pkl")

@st.cache_resource
def load_fertilizer_knn_model():
    return joblib.load("models/fertilizer_knn_model.pkl")

crop_model = load_crop_model()
fertilizer_knn_model = load_fertilizer_knn_model()

# 📍 User inputs
st.sidebar.subheader(t("📍 Select Your City", "📍 अपना शहर चुनें"))
city = st.sidebar.text_input(t("City Name", "शहर का नाम"), value=get_location_by_ip())

weather = get_weather_data(city)
if not weather:
    st.error(t("❌ Could not fetch weather.", "❌ मौसम डेटा नहीं मिला।"))
    st.stop()

st.success(f"📍 {t('Weather data loaded for', 'के लिए मौसम डेटा लोड हुआ')}: {city}")

# User Input Section
st.subheader("🧪 " + t("Soil Nutrients & Soil Type", "मिट्टी पोषक तत्व और प्रकार"))

col1, col2, col3 = st.columns(3)
with col1: n = st.slider("Nitrogen (N)", 0, 100, 50)
with col2: p = st.slider("Phosphorus (P)", 0, 100, 50)
with col3: k = st.slider("Potassium (K)", 0, 100, 50)

soil_type = st.selectbox("🧱 " + t("Select Soil Type", "मिट्टी का प्रकार चुनें"),
                         ["Sandy", "Loamy", "Black", "Red", "Clayey"])

# Show weather
st.markdown(f"""
**🌡️ {t("Temperature", "तापमान")}:** {weather['temperature']} °C  
**💧 {t("Humidity", "नमी")}:** {weather['humidity']} %  
**☁️ {t("Rainfall", "वर्षा")}:** {weather['rainfall']} mm  
**🧪 {t("pH Level", "pH स्तर")}:** {weather['ph']}  
""")

# Conversion
def convert_mgkg_to_kgha(val): return round(val * 1.95, 2)

# Prediction Logic
if st.button(t("🌾 Recommend Crop & Fertilizer", "🌾 फसल और उर्वरक की सिफारिश")):
    converted_n = convert_mgkg_to_kgha(n)
    converted_p = convert_mgkg_to_kgha(p)
    converted_k = convert_mgkg_to_kgha(k)

    st.markdown(f"""
    **🧪 N :** {n} mg/kg → {converted_n} kg/ha  
    **🧪 P :** {p} mg/kg → {converted_p} kg/ha  
    **🧪 K :** {k} mg/kg → {converted_k} kg/ha  
    """)

    # Crop prediction
    input_df = pd.DataFrame([{
        "n": converted_n, "p": converted_p, "k": converted_k,
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "ph": weather["ph"],
        "rainfall": weather["rainfall"]
    }])
    crop = crop_model.predict(input_df)[0]
    st.success(f"✅ {t('Recommended Crop', 'सुझाई गई फसल')}: **{crop.upper()}**")

    # Fertilizer prediction using KNN model
    moisture = round(np.random.uniform(25, 45), 1)
    st.markdown(f"**💧 Soil Moisture (estimated):** {moisture} %")

    fertilizer = get_fertilizer_recommendation(
        weather["temperature"], weather["humidity"], moisture, soil_type, converted_n, converted_p, converted_k, fertilizer_knn_model
    )
    st.success(f"✅ {t('Recommended Fertilizer', 'सुझाया गया उर्वरक')}: **{fertilizer}**")

    required_fertilizer_amount = round((converted_n + converted_p + converted_k) * 0.1, 2)
    st.markdown(f"**💼 Amount to apply (kg/ha):** {required_fertilizer_amount} kg/ha")

    # 📊 Plotly Bar Chart
    required_vals = [25, 60, 6.5, 150]
    actual_vals = [weather['temperature'], weather['humidity'], weather['ph'], weather['rainfall']]
    labels = ['Temperature (°C)', 'Humidity (%)', 'pH', 'Rainfall (mm)']

    fig = go.Figure(data=[ 
        go.Bar(name='Required', x=labels, y=required_vals, marker_color='seagreen'),
        go.Bar(name='Actual', x=labels, y=actual_vals, marker_color='coral')
    ])
    fig.update_layout(
        title=f"📊 Actual vs Required Parameters for {crop.upper()}",
        yaxis_title="Values",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown(""" 
<hr style="border:1px solid #ccc;"/>
<div style='text-align: center; color: gray; font-size: 14px;'>
    Made with ❤️ for Indian farmers. | कृपया सुझाव साझा करें 🌾
</div>
""", unsafe_allow_html=True)