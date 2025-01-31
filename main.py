from PIL.ImageFile import ImageFile
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
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict, Union
from PIL import Image
import requests
from io import BytesIO
import base64
import json
import numpy as np
import logging
import logging.handlers
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a custom logger
logger = logging.getLogger('image-processing-api')

# Create handlers
file_handler = logging.handlers.RotatingFileHandler(
    'api.log',
    maxBytes=10485760,  # 10MB
    backupCount=5
)
console_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

app = FastAPI(
    title="URL to HuggingFace Inference",
    description="API for processing images through HuggingFace model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=['*'],
    expose_headers=["*"]
)

# Security configurations
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME = "gpt-4o-mini"

logger.info(f"Initializing HuggingFace client with model: {MODEL_NAME}")
client = InferenceClient(
    provider="hf-inference",
    api_key=HUGGINGFACE_API_KEY,
    model=MODEL_NAME
)

logger.info("Initializing OpenAI client")
openAIClient = OpenAI()

try:
    AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
    AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY")
    logger.info("Azure Vision credentials loaded successfully")
except KeyError as e:
    logger.error("Missing Azure Vision environment variables", exc_info=True)
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    exit()

# Create an Azure Image Analysis client
logger.info("Initializing Azure Image Analysis client")
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
    logger.info(f"Starting smart crop for image: {url}")
    try:
        result = azureImageClient.analyze_from_url(
            image_url=url,
            visual_features=[VisualFeatures.SMART_CROPS],
            smart_crops_aspect_ratios=[1.0]
        )
        logger.info("Smart crop completed successfully")
        logger.debug(f"Smart crop result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in smart_crop_image: {str(e)}", exc_info=True)
        raise


def get_image_mask(base64image):
    logger.info("Starting image mask generation")
    try:
        api_key = os.getenv('SEGMIND_API_KEY')
        url = "https://api.segmind.com/v1/automatic-mask-generator"

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
        logger.debug(f"Sending request to Segmind API")
        response = requests.post(url, json=data, headers=headers)
        logger.info(f"Mask generation completed with status code: {response.status_code}")
        return json.loads(response.content)
    except Exception as e:
        logger.error(f"Error in get_image_mask: {str(e)}", exc_info=True)
        raise


def base64_to_png_bytes(base64_string: str, is_mask: bool = False) -> BytesIO:
    logger.info(f"Converting base64 to PNG bytes (is_mask: {is_mask})")
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        base64_string = base64_string.replace('\\n', '')

        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))

        if is_mask:
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            img_array = np.array(image)
            luminance = (img_array[..., 0] * 0.299 +
                         img_array[..., 1] * 0.587 +
                         img_array[..., 2] * 0.114)
            alpha = np.where(luminance > 50, 0, 255).astype(np.uint8)
            img_array[..., 3] = alpha
            image = Image.fromarray(img_array)
        else:
            if image.mode != 'RGBA':
                image = image.convert('RGBA')

        png_buffer = BytesIO()
        image.save(png_buffer, format='PNG')
        png_buffer.seek(0)
        png_buffer.name = 'image.png'

        logger.info("Successfully converted base64 to PNG bytes")
        return png_buffer
    except Exception as e:
        logger.error(f"Error in base64_to_png_bytes: {str(e)}", exc_info=True)
        raise


def crop_image_from_url(image_url, bbox):
    logger.info(f"Starting image crop from URL: {image_url}")
    try:
        response = requests.get(image_url)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))

        left = bbox['x']
        top = bbox['y']
        right = left + bbox['w']
        bottom = top + bbox['h']

        logger.debug(f"Cropping coordinates: left={left}, top={top}, right={right}, bottom={bottom}")

        cropped_img = img.crop((left, top, right, bottom))
        buffered = BytesIO()
        cropped_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        logger.info("Successfully cropped and encoded image")
        return img_str
    except requests.RequestException as e:
        logger.error(f"Error fetching image: {str(e)}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return None


@app.post("/generate_image_basic")
async def process_smart_crop(image_data: ImageURL):
    request_id = datetime.now().strftime('%Y%m%d-%H%M%S-') + str(time.time_ns())
    logger.info(f"Starting image generation request {request_id} for URL: {image_data.url}")
    try:
        result = smart_crop_image(str(image_data.url))
        actualResult = result['smartCropsResult']['values'][0]['boundingBox']
        logger.info(f"Smart crop completed for request {request_id}")

        base64_cropped_image = crop_image_from_url(image_url=str(image_data.url), bbox=actualResult)
        logger.info(f"Image cropping completed for request {request_id}")

        base64_cropped_mask = get_image_mask(base64_cropped_image).get('image', '')
        logger.info(f"Mask generation completed for request {request_id}")

        png_image = base64_to_png_bytes(base64_cropped_image, is_mask=False)
        png_mask = base64_to_png_bytes(base64_cropped_mask, is_mask=True)
        logger.info(f"PNG conversion completed for request {request_id}")

        logger.info(f"Sending request to OpenAI API for request {request_id}")
        response = openAIClient.images.edit(
            image=png_image,
            mask=png_mask,
            prompt=prompt,
            n=1,
            size="512x512"
        )
        logger.info(f"OpenAI API response received for request {request_id}")
        logger.debug(f"OpenAI API response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error in process_smart_crop for request {request_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/process-images", response_model=InferenceResponse)
def process_images(request: URLRequest) -> InferenceResponse:
    request_id = datetime.now().strftime('%Y%m%d-%H%M%S-') + str(time.time_ns())
    logger.info(f"Starting image processing request {request_id}")
    try:
        messages = build_messages(request.urls)
        logger.info(f"Messages built for request {request_id}")
        logger.debug(f"Input messages: {messages}")

        completion = openAIClient.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=messages,
            response_format=ExpectedModelOutput
        )
        logger.info(f"Model inference completed for request {request_id}")
        logger.debug(f"Model response: {completion}")

        model_response = completion.choices[0].message.content

        return InferenceResponse(
            model_response=model_response,
            input_messages=messages
        )

    except Exception as e:
        logger.error(f"Error in process_images for request {request_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
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

    logger.info("Starting API server")
    uvicorn.run(app, host="0.0.0.0", port=10000)