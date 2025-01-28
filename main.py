from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="URL to HuggingFace Inference",
    description="API for processing images through HuggingFace model",
    version="1.0.0"
)

# Security configurations
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"


class ImageURL(BaseModel):
    url: HttpUrl


class URLRequest(BaseModel):
    urls: List[HttpUrl]


class InferenceResponse(BaseModel):
    model_response: str
    input_messages: List[Dict[str, Any]]


async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    """Verify the API key provided in headers"""
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return api_key


def create_inference_client():
    """Create a HuggingFace inference client"""
    if not HUGGINGFACE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="HuggingFace API key not configured"
        )

    return InferenceClient(
        provider="hf-inference",
        token=HUGGINGFACE_API_KEY
    )



def build_messages(urls: List[HttpUrl]) -> List[Dict[str, Any]]:
    """Build the messages structure for the model input"""
    text_content = {
        "type": "text",
        "text": "Describe this image in one sentence."
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
async def process_images(
        request: URLRequest,
        api_key: str = Depends(verify_api_key)
) -> InferenceResponse:
    """
    Process images through HuggingFace model
    """
    try:
        # Create inference client
        client = create_inference_client()
        print(client.health_check())

        # Build messages structure
        messages = build_messages(request.urls)
        print(messages)
        # Make inference request
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=500
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
            detail=f"Error processing inference request: {str(e)}"
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