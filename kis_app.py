import streamlit as st
st.set_page_config(page_title="Kisaan Chakshu", page_icon="🌱")  # ✅ FIRST Streamlit command

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


# ----------------- Configuration -----------------
class_names = ["Alluvial", "Black", "Clay", "Red"]

# ✅ Initialize OpenAI client with OpenRouter
client = OpenAI(
    api_key="my_api_key",
    base_url="https://openrouter.ai/api/v1"
)

# ----------------- MongoDB setup -----------------
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["kisaan_chakshu"]
soil_collection = mongo_db["soil_images"]

# ----------------- Model Loading -----------------
@st.cache_resource
def load_model():
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.LayerNorm(num_ftrs),
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 256),
        nn.GELU(),
        nn.LayerNorm(256),
        nn.Dropout(0.3),
        nn.Linear(256, len(class_names))
    )
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ----------------- Helper Functions -----------------
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
        f"The soil is {soil_type}. State in India is {state}. Season is {season}. "
        "Suggest best crops to grow with little description and considering economic profit. Do not give any other tip"
    )
    response = call_llm(query_text)
    if response:
        cleaned_text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        crops = re.findall(r"\d+\.\s\*\*(.*?)\*\*", cleaned_text)
        if not crops:
            crops = [crop.strip().title() for crop in cleaned_text.split(",") if crop.strip()]
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
        server.login("abhimail708@gmail.com", "my_password")
        server.sendmail("abhimail708@gmail.com", email, msg.as_string())

    print(f"📤 Email sent to {name} at {email}")

# ----------------- Streamlit App -----------------
st.title("🌱 Kisaan Chakshu: Smart Farming Assistant")

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
                st.session_state.user['registered'] = False
                st.session_state.user['soil_type'] = None
                st.session_state.user['crops'] = []
                st.session_state.user['state'] = ''
                st.session_state.user['season'] = ''
                st.session_state.user['selected_crop'] = None
                st.session_state.ignore_existing_user = True  # 👈 add a flag
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
            - **State:** `{st.session_state.user['state']}`
            - **Commodity (Crop):** `{selected_crop}`

            👉 Click below to visit the official Agmarknet portal:  
            [🔗 Check Market Prices on Agmarknet](https://agmarknet.gov.in/)

            On the portal:
            - Choose your **State**
            - Select your **Crop (Commodity)**
            - 📅 Choose **From** and **To** dates to check crop price trends.
            - Click **"GO"** to get current prices
        """)
        st.info("This link opens a government resource for daily mandi (market) prices in India.")



# ----------------- Footer -----------------
st.markdown("---")
st.markdown("""
**📢 Note:**  
- Recommendations are based on AI analysis and should be verified with local experts  
- Always test soil nutrients before final crop selection  
- Market prices may vary based on quality and demand
""")
