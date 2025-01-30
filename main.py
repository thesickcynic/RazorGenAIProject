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
import numpy as np


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


def base64_to_png_bytes(base64_string: str, is_mask: bool = False) -> BytesIO:
    """
    Convert base64 string to PNG bytes in memory with RGBA format
    For masks, sets alpha channel to 0 for black pixels

    Args:
        base64_string (str): Base64 encoded image string
        is_mask (bool): Whether this is a mask image

    Returns:
        BytesIO: PNG image in memory in RGBA format
    """
    # Clean the base64 string
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    base64_string = base64_string.replace('\\n', '')

    # Convert to bytes and create image
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))

    if is_mask:
        # Convert to RGBA if needed
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # Get image data as numpy array
        img_array = np.array(image)

        # Calculate luminance (brightness) of RGB channels
        # Using standard RGB to grayscale conversion weights
        luminance = (img_array[..., 0] * 0.299 +
                     img_array[..., 1] * 0.587 +
                     img_array[..., 2] * 0.114)

        # Create alpha channel: 0 for black pixels (low luminance), 255 for others
        # You can adjust the threshold (here 50) as needed
        alpha = np.where(luminance > 50, 0, 255).astype(np.uint8)

        # Set the alpha channel
        img_array[..., 3] = alpha

        # Convert back to PIL Image
        image = Image.fromarray(img_array)
    else:
        # For non-mask images, just convert to RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

    # Convert to PNG in memory
    png_buffer = BytesIO()
    image.save(png_buffer, format='PNG')
    png_buffer.seek(0)
    png_buffer.name = 'image.png'

    return png_buffer

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


class ImageEditRequest(BaseModel):
    image: bytes
    mask: bytes
    prompt: str
    n: int
    size: str

prompt =  "Replace the background with Christmas themed imagery in the style of vintage greeting cards. Keep it subtle so as to not interfere with the actual product image."

def create_image_edit_request(
        base64_cropped_image: bytes,
        base64_cropped_mask: bytes,
        prompt: str = "Remove the background and replace it with Christmas themed imagery in the style of Hallmark greeting cards. Keep it subtle so as to not interfere with the actual product image.",
        n: int = 1,
        size: str = "512x512"
) -> Dict[str, Any]:
    """
    Creates a request body for OpenAI's Image Editing API endpoint.

    Args:
        base64_cropped_image (str): Base64 encoded image string (without data URI prefix)
        base64_cropped_mask (str): Base64 encoded mask string (without data URI prefix)
        prompt (str): The prompt telling the model what edits to make
        n (int): Number of images to generate
        size (str): Size of the output image (1024x1024 or 1024x1792 or 1792x1024)
        model (str): The model to use for image generation

    Returns:
        Dict[str, Any]: Formatted request body for the OpenAI API

    Raises:
        ValueError: If the input parameters are invalid
    """
    # Validate inputs
    if not base64_cropped_image or not base64_cropped_mask:
        raise ValueError("Both image and mask must be provided")

    # Remove data URI prefix if present
    def clean_base64(b64_string: str) -> str:
        if "base64," in b64_string:
            return b64_string.split("base64,")[1]
        return b64_string

    # Clean the base64 strings
    clean_image = clean_base64(base64_cropped_image)
    clean_mask = clean_base64(base64_cropped_mask)

    # Validate size
    # valid_sizes = ["1024x1024", "1024x1792", "1792x1024"]
    # if size not in valid_sizes:
    #     raise ValueError(f"Size must be one of {valid_sizes}")

    # Create request body
    request_body = ImageEditRequest(
        image=clean_image,
        mask=clean_mask,
        prompt=prompt,
        n=n,
        size=size
    )

    return request_body.model_dump()

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


@app.post("/smart-crop")
async def process_smart_crop(image_data: ImageURL):
    try:
        # Get cropped image and mask
        result = smart_crop_image(str(image_data.url))
        actualResult = result['smartCropsResult']['values'][0]['boundingBox']

        base64_cropped_image = crop_image_from_url(image_url=str(image_data.url), bbox=actualResult)
        base64_cropped_mask = get_image_mask(base64_cropped_image).get('image', '')

        # Convert image and mask to proper PNG format in RGBA
        # Pass is_mask=False for the main image
        png_image = base64_to_png_bytes(base64_cropped_image, is_mask=False)
        # Pass is_mask=True for the mask to properly handle alpha channel
        png_mask = base64_to_png_bytes(base64_cropped_mask, is_mask=True)

        # Make the API call to OpenAI
        response = openAIClient.images.edit(
            image=png_image,
            mask=png_mask,
            prompt=prompt,
            n=1,
            size="512x512"
        )

        return response

    except Exception as e:
        print(f"Detailed error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
# @app.post("/smart-crop")
# async def process_smart_crop(image_data: ImageURL):
#     """
#     Process an image through Azure's smart crop functionality
#
#     Args:
#         image_data: Pydantic model containing the image URL
#
#     Returns:
#         dict: The smart crop analysis results from Azure
#     """
#     try:
#         # Call the smart crop function with the provided URL
#         result = smart_crop_image(str(image_data.url))
#         print(result)
#         actualResult = result['smartCropsResult']['values'][0]['boundingBox']
#         print(actualResult)
#
#         base64_cropped_image = (crop_image_from_url(image_url=(str(image_data.url)), bbox = actualResult))
#         base64_cropped_mask = get_image_mask(base64_cropped_image).get('image','')
#
#         decoded_bytes_image = base64.b64decode(base64_cropped_image.replace('\\n', ''))
#         # pngImageImage = Image.open(BytesIO(decoded_bytes_image))
#         decoded_bytes_mask = base64.b64decode(base64_cropped_mask.replace('\\n', ''))
#         # pngImageMask = Image.open(BytesIO(decoded_bytes_mask))
#
#         # Convert base64 strings to bytes
#         image_bytes = base64_to_bytes(base64_cropped_image)
#         mask_bytes = base64_to_bytes(base64_cropped_mask)
#
#         # Create BytesIO objects
#         image_bytesio = BytesIO(image_bytes)
#         mask_bytesio = BytesIO(mask_bytes)
#
#         # Set the file names (required by the OpenAI API)
#         image_bytesio.name = "image.png"
#         mask_bytesio.name = "mask.png"
#
#         print("Pahonch gaye neeche tak BC")
#
#         dalle_request_body = {
#             "image": image_bytesio,
#             "mask": mask_bytesio,
#             "prompt": prompt,
#             "n": 1,
#             "size": "512x512"
#         }
#
#         print("BRRRRRRRRRRRRRR")
#
#
#
#         # Make the API call to OpenAI
#         response = openAIClient.images.edit(image=image_bytesio, mask=mask_bytesio, prompt=prompt, n=1, size="512x512")
#
#         print(base64_cropped_mask.get('image', ''))
#         print("Pahonch gaye neeche tak nacho BC")
#
#         return response
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing image: {str(e)}"
#         )
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