# Kisaan_Chakshu

## Overview
**Kisaan_Chakshu** is an AI-powered web application designed to assist farmers in selecting the best crops based on soil type and seasonal conditions. The system utilizes:
- **Vision Transformers (ViT)** to classify soil type from an uploaded image.
- **LLM-based analysis** to suggest the most suitable crops based on soil type and current weather (summer/winter).

## Features
- Upload a soil image for classification.
- Select the current weather condition (summer or winter).
- Get AI-powered recommendations for the best crops to grow.

## Tech Stack
- **Machine Learning:** Vision Transformers (ViT) for soil classification
- **LLM Integration:** Used for crop recommendation
- **Backend:** Flask (or FastAPI if preferred)
- **Frontend:** HTML, CSS, JavaScript (React optional for a better UI)
- **Deployment:** Docker (optional), Cloud-based hosting

## How It Works
1. **User Input:**
   - Upload a soil image.
   - Select the current weather condition.
2. **Soil Classification:**
   - The uploaded image is processed using a fine-tuned Vision Transformer model.
   - Model Used: `models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)`, fine-tuned on the [Soil Image Classification Dataset](https://www.kaggle.com/datasets/faisalkhaan/soil-image-classification/data).
   - The dataset contains 4 classes:
     - Alluvial Soil
     - Black Soil
     - Clay Soil
     - Red Soil
3. **Crop Prediction:**
   - The soil type and weather condition are sent to an LLM model.
   - LLM API used: `https://api.together.xyz/v1/chat/completions`
   - The LLM suggests the top suitable crops.
4. **Output Display:**
   - The user receives a list of recommended crops.

## Setup & Installation
### Prerequisites
- Python 3.8+
- Pip and Virtual Environment
- Kaggle API (if training models on Kaggle)
- Required Python libraries:
  ```sh
  pip install torch torchvision transformers flask openai numpy pandas opencv-python
  ```

### Running the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/Kisaan_Chakshu.git
   cd Kisaan_Chakshu
   ```
2. Run the backend server:
   ```sh
   python app.py
   ```
3. Access the web application in your browser.

## Future Enhancements
- Integration of real-time weather data
- Mobile-friendly UI
- Support for regional languages

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to improve.

## License
MIT License
