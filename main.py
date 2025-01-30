from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict, Union
from PIL import Image
import requests
from io import BytesIO
import base64
import json



# Load environment variables
load_dotenv()

app = FastAPI(
    title="URL to HuggingFace Inference",
    description="API for processing images through HuggingFace model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["GET","POST"],
    allow_headers=['*'],
    expose_headers=["*"]
)

# Security configurations
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME = "gpt-4o-mini"


client = InferenceClient(
	provider="hf-inference",
	api_key=HUGGINGFACE_API_KEY,
    model=MODEL_NAME
)

openAIClient = OpenAI()

try:
    AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
    AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY")
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()

# Create an Azure Image Analysis client
azureImageClient = ImageAnalysisClient(
    endpoint="https://razorgroup.cognitiveservices.azure.com/",
    credential=AzureKeyCredential(AZURE_VISION_KEY)
)

maskingPayloadStub = {
  "prompt": "Product",
  "image": '',
  "threshold": 0.2,
  "invert_mask": False,
  "return_mask": True,
  "grow_mask": 20,
  "seed": 468685,
  "base64": True
}

def smart_crop_image(url):
    result = azureImageClient.analyze_from_url(
        image_url = url,
        visual_features = [VisualFeatures.SMART_CROPS],
        smart_crops_aspect_ratios = [1.0]
    )
    return result

def get_image_mask(base64image):
    api_key = os.getenv('SEGMIND_API_KEY')
    url = "https://api.segmind.com/v1/automatic-mask-generator"

    # Request payload
    data = {
      "prompt": "Product",
      "image": base64image,
      "threshold": 0.2,
      "invert_mask": True,
      "return_mask": True,
      "grow_mask": 10,
      "seed": 468685,
      "base64": True
    }

    headers = {'x-api-key': api_key}
    response = requests.post(url, json=data, headers=headers)
    print(response)
    print('xxxxxxxxxxxxxxxxxxxxxxxxx Agli line madarchod. xxxxxxxxxxxxxxxxxxxxx')
    print(response.content)  # The response is the generated image
    return json.loads(response.content)



class ImageURL(BaseModel):
    url: HttpUrl


class URLRequest(BaseModel):
    urls: List[HttpUrl]


class InferenceResponse(BaseModel):
    model_response: str
    input_messages: List[Dict[str, Any]]

class ExpectedModelOutput(BaseModel):
    dimension: bool
    angle: bool
    lifestyle: bool


class ImageAnalysisResult(BaseModel):
    x: int
    y: int
    w: int
    h: int


# class MaskingModelResponse(BaseModel):
#     base64

# def create_inference_client():
#     """Create a HuggingFace inference client"""
#     if not HUGGINGFACE_API_KEY:
#         raise HTTPException(
#             status_code=500,
#             detail="HuggingFace API key not configured"
#         )
#
#
#     return InferenceClient(
#         provider="hf-inference",
#         token=HUGGINGFACE_API_KEY,
#         model="Qwen/Qwen2-VL-7B-Instruct"
#     )



def build_messages(urls: List[HttpUrl]) -> List[Dict[str, Any]]:
    """Build the messages structure for the model input"""
    text_content = {
        "type": "text",
        "text": "You are a panel of three experts on eCommerce conversion rate optimisation - Alice, Bob, "
                "and Charles. You will be provided a set of images from product listings, you need to determine if "
                "they meet a set of deadlines. You will do this via a panel discussion, trying to solve it step by "
                "step and make sure that the result is correct."
                "The guidelines are: 1. There should be one image with dimensions and details on it. 2. There should "
                "be images from multiple angles. 3. There should be an image showing the product used in a lifestyle "
                "setting. Once you have completed the evaluation, present the answer as a JSON object with parameters "
                "as dimension, angle, and lifestyle with Boolean values. Only reply with the JSON object, nothing else."
    }

    image_contents = [
        {
            "type": "image_url",
            "image_url": {
                "url": str(url)
            }
        }
        for url in urls
    ]

    return [{
        "role": "user",
        "content": [text_content] + image_contents
    }]




def crop_image_from_url(image_url, bbox):
    """
    Fetch an image from URL, crop it, and return as base64 string.

    Args:
        image_url (str): URL of the input image
        bbox (dict): Dictionary containing x, y, w, h values for cropping

    Returns:
        str: Base64 encoded string of the cropped image
    """
    try:
        # Fetch the image from URL
        response = requests.get(image_url)
        response.raise_for_status()

        # Create an image object from the downloaded bytes
        img = Image.open(BytesIO(response.content))

        # Calculate the coordinates for cropping
        left = bbox['x']
        top = bbox['y']
        right = left + bbox['w']
        bottom = top + bbox['h']

        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))

        # Convert the cropped image to base64
        buffered = BytesIO()
        cropped_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # return f"data:image/png;base64,{img_str}"
        return img_str

    except requests.RequestException as e:
        print(f"Error fetching image: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

# @app.post("/smart-crop", response_model=dict)
@app.post("/smart-crop", response_model = ImageAnalysisResult)
async def process_smart_crop(image_data: ImageURL):
    """
    Process an image through Azure's smart crop functionality

    Args:
        image_data: Pydantic model containing the image URL

    Returns:
        dict: The smart crop analysis results from Azure
    """
    try:
        # Call the smart crop function with the provided URL
        result = smart_crop_image(str(image_data.url))
        print(result)
        actualResult = result['smartCropsResult']['values'][0]['boundingBox']
        print(actualResult)
        base64_cropped_image = (crop_image_from_url(image_url=(str(image_data.url)), bbox = actualResult))
        base64_cropped_mask = get_image_mask(base64_cropped_image)
        print(base64_cropped_mask.get('image',''))
        print("Pahonch gaye neeche tak nacho BC")

        # Convert the Azure response to a dictionary
        # Note: We're converting to dict to ensure serializable response
        return ImageAnalysisResult(
            actualResult
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
@app.post("/process-images", response_model=InferenceResponse)
def process_images(
        request: URLRequest) -> InferenceResponse:
    """
    Process images through HuggingFace model
    """
    try:
        # Build messages structure
        messages = build_messages(request.urls)
        # Make inference request
        completion = openAIClient.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=messages,
            response_format=ExpectedModelOutput
        )

        # Extract model response
        model_response = completion.choices[0].message.content

        return InferenceResponse(
            model_response=model_response,
            input_messages=messages
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ye phata: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "name": "URL to HuggingFace Inference API",
        "version": "1.0.0",
        "endpoints": {
            "/process-images": "POST - Process images through HuggingFace model",
            "/": "GET - This information"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=10000)