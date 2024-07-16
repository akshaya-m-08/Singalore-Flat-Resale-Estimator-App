import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
from streamlit_option_menu import option_menu

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="Singapore Resale Flat Price Estimator",
    page_icon="asset\home_tab2.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the sidebar and main content
custom_css = """
<style>
.image-container img {
    height: 200px;
    width: 200px;
    display: block;
    margin-left: auto;
    margin-right: auto;
}
.css-1l02zno {
    text-align: center;
}
.side-image-container img {
    height: 100px;
    width: 100px;
    display: block;
    margin-left: auto;
    margin-right: auto;
}
[data-testid="stSidebar"] {
    background-color: #2F4F4F; 
    text-align: center;
    color: white !important;
}
[data-testid="stMetric"] 
{
    background-color: #4682B4; 
    text-align: center;
    padding: 5px 0;
    font-weight: bold;
    border-radius: 15px;
    color: White; 
    width: auto;
}
[data-testid="stTitle"] 
{
    text-align: center;
    padding: 5px 0;
    font-weight: bold;
    border-radius: 15px;
    color: #4682B4; 
    width: auto;
}
[data-testid="stMetricValue"] {
    font-size: 30px;
}
</style>
"""

# Embed custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Function to get the base64 encoded string of the image
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('best_model.pkl')

model = load_model()

# Function to preprocess user input and predict resale price
def preprocess_input(year, town, flat_type, flat_model, storey_range, floor_area_sqm, lease_commence_date):
    current_year = 2024
    remaining_lease = 99 - (year - lease_commence_date)
    current_remaining_lease = remaining_lease - (current_year - year)
    years_holding = year - lease_commence_date

    # Create a dataframe from user input
    input_data = pd.DataFrame({
        'year': [year],
        'town': [town],
        'flat_type': [flat_type],
        'flat_model': [flat_model],
        'storey_range': [storey_range],
        'floor_area_sqm': [floor_area_sqm],
        'lease_commence_date': [lease_commence_date],
        'remaining_lease': [remaining_lease],
        'current_remaining_lease': [current_remaining_lease],
        'years_holding': [years_holding]
    })

    # Preprocess the input data in the same way as the training data
    input_data['flat_model'] = input_data['flat_model'].str.upper()
    input_data['flat_type'] = input_data['flat_type'].str.upper()
    input_data[['lower_bound', 'upper_bound']] = input_data['storey_range'].str.split(' TO ', expand=True)
    input_data['lower_bound'] = pd.to_numeric(input_data['lower_bound'])
    input_data['upper_bound'] = pd.to_numeric(input_data['upper_bound'])

    return input_data

def predict_resale_price(year, town, flat_type, flat_model, storey_range, floor_area_sqm, lease_commence_date):
    input_data = preprocess_input(year, town, flat_type, flat_model, storey_range, floor_area_sqm, lease_commence_date)
    prediction = model.predict(input_data)
    return prediction[0]

@st.cache_data
def load_static_data():
    towns = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'LIM CHU KANG', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN']
    flat_types = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI GENERATION', 'MULTI-GENERATION', '3Gen']
    flat_models = ['ADJOINED FLAT', 'APARTMENT', 'DBSS', 'IMPROVED', 'IMPROVED-MAISONETTE', 'MAISONETTE', 'MODEL A', 'MODEL A2', 'NEW GENERATION', 'PREMIUM APARTMENT', 'PREMIUM APARTMENT LOFT', 'SIMPLIFIED', 'STUDIO APARTMENT', 'TERRACE', 'TYPE S1', 'TYPE S2', 'Model A-Maisonette', 'Multi Generation', 'Premium Maisonette', 'Standard', '2-room']
    storey_ranges = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51', '06 TO 10', '01 TO 05', '11 TO 15', '16 TO 20', '21 TO 25', '26 TO 30', '36 TO 40', '31 TO 35']
    return towns, flat_types, flat_models, storey_ranges

# Sidebar content
with st.sidebar:
    logo_path = "asset\home_tab2.png"
    logo_base64 = get_image_base64(logo_path)
    st.markdown(f'<div class="side-image-container"><img src="data:image/png;base64,{logo_base64}" alt="Company Logo" /></div>', unsafe_allow_html=True)   

    # Default tab selection
    st.write("")
    st.write("")
    selected = option_menu(
        menu_title="Singapore",
        options=["Home", "Application"],
        icons=["house", "app-indicator"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"background-color": "#2F4F4F"},
            "icon": {"color": "white", "font-size": "16px"},
            "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#4682B4"},
            "nav-link-selected": {"background-color": "#4682B4"},
            "menu-title": {"color": "white", "font-size": "20px", "text-align": "center", "margin-bottom": "10px"},
        }
    )

if selected == "Home":
    logo_path = "asset\home_tab1.png"
    logo_base64 = get_image_base64(logo_path)
    st.markdown(f'<div class="image-container"><img src="data:image/png;base64,{logo_base64}" alt="Company Logo" /></div>', unsafe_allow_html=True)   
    st.title("Welcome to Singapore Resale Flat Price Estimator")
    st.header("Singapore - A Vibrant City-State")
    st.write("""
        Singapore, an island city-state in Southeast Asia, is renowned for its rapid development, excellent infrastructure, and vibrant economy.
        With a diverse culture, world-class education, and a high standard of living, Singapore attracts people from all over the world.
        The Housing and Development Board (HDB) flats are a popular choice among residents, and their resale value is a topic of great interest.
    """)
    col1, col2, col3 = st.columns((2,8,2),gap="large")
    with col2:
        VIDEO_URL = "https://www.youtube.com/watch?v=R2Yk6UmXxMM"
        st.video(VIDEO_URL)
elif selected == "Application":
    logo_path = "asset\logo.gif"
    logo_base64 = get_image_base64(logo_path)
    st.markdown(f'<div class="image-container"><img src="data:image/png;base64,{logo_base64}" alt="Company Logo" /></div>', unsafe_allow_html=True)   
    st.title('Singapore Resale Flat Price Estimator')

    # Load data for dropdown options
    towns, flat_types, flat_models, storey_ranges = load_static_data()

    # User input
    year = st.selectbox('Year of Sale', options=list(range(1990, 2025)), index=34)
    town = st.selectbox('Town', towns)
    flat_type = st.selectbox('Flat Type', flat_types)
    flat_model = st.selectbox('Flat Model', flat_models)
    storey_range = st.selectbox('Storey Range', storey_ranges)
    floor_area_sqm = st.text_input('Floor Area (sqm)', '90.0', help="Enter a value between 10 and 1000")
    lease_commence_date = st.selectbox('Lease Commence Date', options=list(range(1960, 2025)), index=40)

    # Validate floor_area_sqm input
    try:
        floor_area_sqm = float(floor_area_sqm)
        if not 10.0 <= floor_area_sqm <= 1000.0:
            floor_area_sqm = None
    except ValueError:
        floor_area_sqm = None

    predicted_price = ""
    if st.button('Estimate Resale Price'):
        if floor_area_sqm is not None:
            predicted_price = predict_resale_price(year, town, flat_type, flat_model, storey_range, floor_area_sqm, lease_commence_date)
            col1,col2,col3 = st.columns((2,5,2),gap="large")
            with col2:
                st.markdown('<style>div[data-testid="stMetric"] > div {text-align: center;}</style>', unsafe_allow_html=True)
                st.metric(label="Estimated Resale Price", value=f"${predicted_price:,.2f}")
        else:
            st.warning("Please enter a valid floor area between 10 and 200 sqm.")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")       
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")           
    st.write("### About Application")
    st.write("""
        This application predicts the resale price of HDB flats in Singapore based on various input parameters.
        The model was trained using historical transaction data and incorporates features such as the year of sale,
        town, flat type, flat model, storey range, floor area, and lease commencement date.
        The predictions aim to provide an estimate of the resale value, aiding potential buyers and sellers in making informed decisions.
    """)
    st.write("")
    st.write("")
    st.write("### Developer Information")
    st.write("""
        This application was developed by Akshaya Muralidharan. Akshaya is a dedicated professional with a diverse background transitioning into Data Science. Proficient in Python, data preprocessing, and visualization, with hands-on experience in machine learning algorithms. Committed to leveraging advanced analytics to derive actionable insights and drive business growth. Connect with Akshaya on [LinkedIn](https://www.linkedin.com/in/akshayam08/).
    """)
