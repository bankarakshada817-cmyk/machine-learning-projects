
import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Custom Light Theme
# -----------------------------
custom_css = """
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    font-family: Arial, sans-serif;
    color: #0f172a !important;
}

/* Main Content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Header */
.center-header {
    text-align: center;
    margin-bottom: 30px;
    padding-top: 10px;
}

.center-header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    color: #0f172a !important;
}

.center-header p {
    font-size: 1.05rem;
    color: #475569 !important;
}

/* Form Card */
div[data-testid="stForm"] {
    background-color: #ffffff !important;
    padding: 30px !important;
    border-radius: 16px !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.06) !important;
}

/* Section Headings */
.box-section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1e293b !important;
    margin-bottom: 16px;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 10px;
}

/* Labels */
div[data-testid="stForm"] label,
div[data-testid="stForm"] p {
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* Text Input Background */
div[data-baseweb="input"] {
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* Selectbox Outer Background */
div[data-baseweb="select"] > div {
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* Number Input Background */
div[data-testid="stNumberInputContainer"] {
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* Input Fields */
input,
textarea {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    caret-color: #0f172a !important;
}

/* Selected Dropdown Text */
div[data-baseweb="select"] span {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
}

/* Dropdown Arrow */
div[data-baseweb="select"] svg {
    fill: #475569 !important;
    color: #475569 !important;
}

/* Focus Effects */
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"] > div:focus-within,
div[data-testid="stNumberInputContainer"]:focus-within {
    border-color: #2563eb !important;
    background-color: #ffffff !important;
}

/* Number Input Plus and Minus Buttons */
div[data-testid="stNumberInputContainer"] button {
    background-color: #e2e8f0 !important;
    color: #0f172a !important;
    border: none !important;
}

/* Dropdown Menu */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #ffffff !important;
}

/* Dropdown Options */
ul[role="listbox"] li {
    background-color: #ffffff !important;
    color: #0f172a !important;
}

/* Dropdown Option Hover */
ul[role="listbox"] li:hover {
    background-color: #dbeafe !important;
    color: #0f172a !important;
}

/* Predict Button */
.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    min-height: 48px;
    border-radius: 10px;
    background-color: #2563eb !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    transition: all 0.2s ease;
}

/* Button Hover */
.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
    border: none !important;
}

/* Result Card */
.result-card {
    background-color: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 14px;
    padding: 25px;
    text-align: center;
    margin-top: 25px;
}

/* Result Heading */
.result-card h3 {
    color: #0369a1 !important;
    font-size: 1.2rem;
    margin-bottom: 8px;
}

/* Predicted Price */
.result-card h1 {
    color: #0c4a6e !important;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
}

/* Success and Error Messages */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Hide Streamlit Header and Footer */
#MainMenu,
footer {
    visibility: hidden;
}

/* Responsive Heading */
@media (max-width: 768px) {
    .center-header h1 {
        font-size: 1.8rem;
    }

    div[data-testid="stForm"] {
        padding: 18px !important;
    }
}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


# -----------------------------
# Load Pickle Files
# -----------------------------
@st.cache_resource
def load_assets():
    with open("house_price_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        dv = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("features.pkl", "rb") as f:
        features = pickle.load(f)

    return model, dv, encoder, features


try:
    model, dv, encoder, features = load_assets()
except FileNotFoundError as e:
    st.error(f"Required model file not found: {e.filename}")
    st.info(
        "Keep house_price_model.pkl, vectorizer.pkl, "
        "encoder.pkl and features.pkl in the project folder."
    )
    st.stop()


# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="center-header">
    <h1>🏠 House Price Prediction System</h1>
    <p>Predict estimated property prices using Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Property Form
# -----------------------------
with st.form("property_form", border=False):

    col1, col2 = st.columns(2, gap="large")

    # Basic Property Details
    with col1:
        st.markdown(
            '<div class="box-section-title">'
            '📍 Basic Property Specifications</div>',
            unsafe_allow_html=True
        )

        state = st.selectbox(
            "State",
            ["maharashtra", "karnataka", "gujarat", "delhi", "tamil nadu"]
        )

        city = st.text_input("City", value="Mumbai")

        property_type = st.selectbox(
            "Property Type",
            ["Apartment", "Independent House", "Villa"]
        )

        bhk = st.number_input(
            "BHK", min_value=1, max_value=10, value=2
        )

        size_sqft = st.number_input(
            "Size (Sq.Ft)", min_value=300, max_value=10000, value=1200
        )

        price_sqft = st.number_input(
            "Price Per Sq.Ft", min_value=1000, value=5000
        )

        year = st.number_input(
            "Year Built", min_value=1980, max_value=2026, value=2018
        )

        furnished = st.selectbox(
            "Furnished Status",
            ["Unfurnished", "Semi-furnished", "Furnished"]
        )

        floor = st.number_input(
            "Floor Number", min_value=0, value=2
        )

        total_floor = st.number_input(
            "Total Floors", min_value=1, value=10
        )

    # Locality and Additional Features
    with col2:
        st.markdown(
            '<div class="box-section-title">'
            '🌟 Locality & Additional Features</div>',
            unsafe_allow_html=True
        )

        age = st.number_input(
            "Age of Property (Years)", min_value=0, value=5
        )

        school = st.number_input(
            "Nearby Schools", min_value=0, value=5
        )

        hospital = st.number_input(
            "Nearby Hospitals", min_value=0, value=3
        )

        transport = st.selectbox(
            "Public Transport Accessibility",
            ["Low", "Medium", "High"]
        )

        parking = st.selectbox(
            "Parking Space", ["Yes", "No"]
        )

        security = st.selectbox(
            "Security", ["No", "Yes"]
        )

        amenities = st.text_input(
            "Amenities", value="gym, pool"
        )

        facing = st.selectbox(
            "Facing Direction",
            ["South", "East", "West", "North"]
        )

        owner = st.text_input(
            "Owner Type", value="First Owner"
        )

        availability = st.text_input(
            "Availability Status", value="Ready to Move"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "🏡 Predict House Price",
        use_container_width=True
    )


# -----------------------------
# Prediction Logic
# -----------------------------
if submitted:

    try:
        ordinal_df = pd.DataFrame({
            "Property_Type": [property_type],
            "Furnished_Status": [furnished],
            "Public_Transport_Accessibility": [transport],
            "Facing": [facing],
            "Security": [security]
        })

        ordinal_encoded = encoder.transform(ordinal_df)

        input_data = {
            "State": state.lower(),
            "City": city.strip().lower(),
            "Property_Type": ordinal_encoded[0][0],
            "BHK": bhk,
            "Size_in_SqFt": size_sqft,
            "Price_per_SqFt": price_sqft,
            "Year_Built": year,
            "Furnished_Status": ordinal_encoded[0][1],
            "Floor_No": floor,
            "Total_Floors": total_floor,
            "Age_of_Property": age,
            "Nearby_Schools": school,
            "Nearby_Hospitals": hospital,
            "Public_Transport_Accessibility": ordinal_encoded[0][2],
            "Parking_Space": parking.lower(),
            "Security": ordinal_encoded[0][3],
            "Amenities": amenities.lower(),
            "Facing": ordinal_encoded[0][4],
            "Owner_Type": owner.strip().lower(),
            "Availability_Status": availability.strip().lower()
        }

        X = dv.transform([input_data])
        prediction = model.predict(X)[0]

        st.markdown(
            f"""
            <div class="result-card">
                <h3>Estimated Property Price</h3>
                <h1>₹ {prediction:.2f} Lakhs</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error("Prediction failed. Please check your input values and model files.")
        st.exception(e)
