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