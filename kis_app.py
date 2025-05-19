import streamlit as st
import base64
import os
from pymongo import MongoClient
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import re
import io
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
import gdown


# -------------- Set Background --------------
def set_bg(jpg_file):
    with open(jpg_file, "rb") as f:
        img_data = f.read()
    b64_img = base64.b64encode(img_data).decode()
    css = f""" 
    <style>
    .stApp {{
        background: url('data:image/jpeg;base64,{b64_img}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

IMAGE_PATH = "field2.png"
if not os.path.exists(IMAGE_PATH):
    st.error("Background image not found!")
else:
    set_bg(IMAGE_PATH)

# -------------- Add Styling --------------
st.markdown("""
<style>
.st-emotion-cache-1y4p8pa, .st-emotion-cache-1gulkj5, .st-emotion-cache-1hynsf2 {
    background-color: rgba(255, 255, 255, 0.95) !important;
    color: #2c3e50 !important;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
h1, h2, h3, h4, h5, h6 {
    color: #2c3e50 !important;
    font-family: 'Arial Rounded MT Bold', sans-serif;
}
body {
    color: #2c3e50 !important;
    font-size: 16px;
}
.stButton>button {
    background-color: #27ae60 !important;
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #1e874b !important;
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# -------------- Configuration --------------
class_names = ["Alluvial", "Black", "Clay", "Red"]

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)
mongo_uri="mongodb+srv://bt22csd040:abhi1234@cluster0.am4cwfw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_client = MongoClient(mongo_uri)
mongo_db = mongo_client["kisaan_chakshu"]
soil_collection = mongo_db["soil_images"]

# -------------- Download Model from Google Drive --------------
@st.cache_resource
def download_model():
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        # Direct download link from Google Drive
        url = "https://drive.google.com/uc?id=1vS9zVymxsf5S1ZYpGblXt8P432rr2NnC"
        gdown.download(url, model_path, quiet=False)
    return model_path

# -------------- Model Loading --------------
@st.cache_resource
def load_model():
    model_path = download_model()

    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.LayerNorm(num_ftrs),
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 256),
        nn.GELU(),
        nn.LayerNorm(256),
        nn.Dropout(0.3),
        nn.Linear(256, len(class_names))  # Make sure class_names is defined globally
    )

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# -------------- Helper Functions --------------
def predict_soil(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)
    return class_names[predicted_class.item()]

def call_llm(query_text):
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[{"role": "user", "content": query_text}],
            extra_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Kisaan Chakshu"
            }
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def get_crop_recommendation(soil_type, state, season):
    query_text = (
        f"The soil is {soil_type}. The location is {state}, India. Season is {season}. "
        "Suggest the top 5 best crops to grow considering soil type, season, and economic profitability. "
        "Use the format strictly as:\n"
        "1. CropName - very short description\n"
        "2. CropName - very short description\n"
        "Only return the list, no other text."
    )
    response = call_llm(query_text)
    if response:
        cleaned_text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        crops = re.findall(r"\d+\.\s(.*?)\s-\s", cleaned_text)
        if not crops:
            crops = [line.split("-")[0].strip() for line in cleaned_text.split("\n") if "-" in line]
        return crops
    return []

def store_prediction_mongo(name, email, image_file, soil_type, crops, state, season):
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    doc = {
        "farmer_name": name,
        "farmer_email": email,
        "soil_type": soil_type,
        "crops": crops,
        "state": state,
        "season": season,
        "image": encoded_image,
        "timestamp": datetime.utcnow()
    }
    soil_collection.insert_one(doc)

def send_email(name, email, soil_type, crops):
    subject = "Crop Recommendation from Kisaan Chakshu"
    body = f"Hi Mr. {name},\n\nThank you for using Kisaan Chakshu.\nYour soil type is {soil_type}, and based on the current season, you should grow: {', '.join(crops)}.\n\nHappy Farming!\nTeam Kisaan Chakshu"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = "abhimail708@gmail.com"
    msg['To'] = email

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login("abhimail708@gmail.com", "fifwaozhnpnmzldo")
        server.sendmail("abhimail708@gmail.com", email, msg.as_string())

# -------------- Streamlit UI --------------
st.title("🌾 Kisaan Chakshu")
st.markdown("### Your Smart Farming Companion", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("<div style='text-align: center; margin: 20px 0;'>🌾🚜🌻</div>", unsafe_allow_html=True)

if 'user' not in st.session_state:
    st.session_state.user = {
        'email': '',
        'registered': False,
        'soil_type': None,
        'crops': [],
        'state': '',
        'season': '',
        'selected_crop': None
    }

email = st.text_input("📧 Enter your email to get started:")

if email:
    st.session_state.user['email'] = email
    cursor = soil_collection.find({"farmer_email": email}).sort("timestamp", -1).limit(1)
    existing_user = next(cursor, None)

    if existing_user and not st.session_state.user['registered'] and not st.session_state.get("ignore_existing_user", False):
        st.success("✅ Welcome back! We found your previous records")      
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Use Previous Analysis"):
                st.session_state.user.update({
                    'soil_type': existing_user['soil_type'],
                    'crops': existing_user['crops'],
                    'state': existing_user['state'],
                    'season': existing_user['season'],
                    'registered': True
                })
        with col2:
            if st.button("🆕 New Analysis"):
                st.session_state.user.update({
                    'registered': False,
                    'soil_type': None,
                    'crops': [],
                    'state': '',
                    'season': '',
                    'selected_crop': None
                })
                st.session_state.ignore_existing_user = True
                st.rerun()

        with st.expander("📚 View Previous Analysis"):
            st.image(Image.open(io.BytesIO(base64.b64decode(existing_user["image"]))), caption="Previous Soil Image")
            st.markdown(f"""
                - **Soil Type**: {existing_user['soil_type']}
                - **Recommended Crops**: {', '.join(existing_user['crops'])}
                - **Last Updated**: {existing_user['timestamp'].strftime('%d %b %Y')}
            """)
    else:
        with st.form("user_details"):
            st.subheader("👨‍🌾 Farmer Details")
            name = st.text_input("Full Name")
            state = st.text_input("State/Region")
            season = st.selectbox("Current Season", ["Kharif", "Rabi", "Zaid", "Monsoon", "Winter", "Summer"])
            soil_img = st.file_uploader("Upload Soil Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

            if st.form_submit_button("🔍 Analyze Soil"):
                if all([name, state, season, soil_img]):
                    with st.spinner("Analyzing soil..."):
                        img = Image.open(soil_img).convert("RGB")
                        soil_type = predict_soil(img)
                        crops = get_crop_recommendation(soil_type, state, season)

                        soil_img.seek(0)
                        store_prediction_mongo(name, email, soil_img, soil_type, crops, state, season)
                        send_email(name, email, soil_type, crops)
                        st.session_state.user.update({
                            'registered': True,
                            'soil_type': soil_type,
                            'crops': crops,
                            'state': state,
                            'season': season
                        })
                        st.rerun()
                else:
                    st.error("Please fill all fields")

if st.session_state.user['registered']:
    st.subheader("🌾 Crop Recommendations")
    st.markdown(f"""
    **Soil Type:** {st.session_state.user['soil_type']}  
    **Location:** {st.session_state.user['state']}  
    **Season:** {st.session_state.user['season']}
    """)

    selected_crop = st.selectbox("Select the crop you want to grow", st.session_state.user['crops'])

    if st.button("📖 Get Cultivation Guide"):
        with st.spinner("Generating cultivation guide..."):
            query = (
                f"Provide detailed cultivation instructions for {selected_crop} in {st.session_state.user['state']} during {st.session_state.user['season']} season. "
                "Include:\n1. Soil preparation\n2. Fertilizer requirements\n3. Planting schedule\n"
                "4. Irrigation needs\n5. Pest control\n6. Harvesting timeline\n"
                "7. Expected yield\n8. Market value in Indian Rupees\n"
                "Use bullet points and simple language."
            )
            guide = call_llm(query)
            if guide:
                st.subheader(f"🌱 {selected_crop} Cultivation Guide")
                st.markdown(guide)

    if st.button("📊 Do you want to get the market price of your crop?"):
        st.subheader("💱 Market Price Info")
        st.markdown(f"""
            - *State:* {st.session_state.user['state']}
            - *Commodity (Crop):* {selected_crop}

            👉 Click below to visit the official Agmarknet portal:  
            [🔗 Check Market Prices on Agmarknet](https://agmarknet.gov.in/)

            On the portal:
            - Choose your *State*
            - Select your *Crop (Commodity)*
            - 📅 Choose *From* and *To* dates to check crop price trends.
            - Click *"GO"* to get current prices
        """)
        st.info("This link opens a government resource for daily mandi (market) prices in India.")

    # -------------- Feedback Section --------------
    st.markdown("---")
    st.subheader("📝 We Value Your Feedback")
    st.markdown("""
    Please take a moment to share your experience using **Kisaan Chakshu** or let us know if you faced any issues.  
    Your feedback helps us improve!

    👉 [**Click here to fill out the feedback form**](https://docs.google.com/forms/d/e/1FAIpQLSctVkGa0KacDGGAv09TNlvkWkfYS-XHOMtVP6QTGvfbwTSkQQ/viewform?usp=sharing&ouid=106426102093727668295)  
    """)
# -------------- Footer --------------
st.markdown("---")
st.markdown("""
**📢 Note:**

* Recommendations are based on AI analysis and should be verified with local experts  
* Always test soil nutrients before final crop selection  
* Market prices may vary based on quality and demand  
""")
