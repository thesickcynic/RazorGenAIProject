from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any

app = FastAPI(
    title="URL to JSON Builder",
    description="API for building JSON structure with multiple URLs",
    version="1.0.0"
)


class ImageURL(BaseModel):
    url: HttpUrl


class ContentItem(BaseModel):
    type: str
    text: str | None = None
    image_url: Dict[str, ImageURL] | None = None


class URLRequest(BaseModel):
    urls: List[HttpUrl]


@app.post("/build-json")
async def build_json(request: URLRequest) -> List[Dict[str, Any]]:
    """
    Build JSON structure with a static text content and multiple image URLs
    """
    # Create the text content item
    text_content = {
        "type": "text",
        "text": "Describe this image in one sentence."
    }

    # Create image URL content items
    image_contents = [
        {
            "type": "image_url",
            "image_url": {
                "url": str(url)
            }
        }
        for url in request.urls
    ]

    # Combine text and image contents
    content = [text_content] + image_contents

    # Create the final structure
    json_structure = [
        {
            "role": "user",
            "content": content
        }
    ]

    return json_structure


@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "name": "URL to JSON Builder API",
        "version": "1.0.0",
        "endpoints": {
            "/build-json": "POST - Build JSON structure with URLs",
            "/": "GET - This information"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=10000)