from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
import httpx
import urllib.parse
from pathlib import Path
import asyncio
from datetime import datetime

app = FastAPI(
    title="File URL Parser",
    description="API for parsing and processing file URLs",
    version="1.0.0"
)


class FileURLRequest(BaseModel):
    urls: List[HttpUrl]


class FileURLResponse(BaseModel):
    processed_urls: int
    valid_urls: List[str]
    invalid_urls: List[str]
    file_types: dict
    total_size: int
    processing_time: float


async def validate_file_url(url: str, client: httpx.AsyncClient) -> tuple:
    """
    Validate a file URL by checking its accessibility and headers
    Returns: (is_valid, file_size, file_type)
    """
    try:
        # Send HEAD request to get metadata without downloading the file
        response = await client.head(str(url), follow_redirects=True, timeout=10.0)
        response.raise_for_status()

        # Get content type and size
        content_type = response.headers.get('content-type', 'unknown')
        content_length = int(response.headers.get('content-length', 0))

        # Get file extension from URL
        file_extension = Path(urllib.parse.urlparse(str(url)).path).suffix

        return True, content_length, content_type, file_extension
    except Exception as e:
        return False, 0, None, None


@app.post("/parse-urls", response_model=FileURLResponse)
async def parse_urls(request: FileURLRequest) -> FileURLResponse:
    """
    Parse and validate a list of file URLs
    """
    start_time = datetime.now()

    valid_urls = []
    invalid_urls = []
    file_types = {}
    total_size = 0

    async with httpx.AsyncClient() as client:
        # Create tasks for all URLs
        tasks = [validate_file_url(url, client) for url in request.urls]
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for url, result in zip(request.urls, results):
            if isinstance(result, tuple) and result[0]:  # Valid URL
                is_valid, size, content_type, extension = result
                valid_urls.append(str(url))
                total_size += size

                # Track file types
                if extension:
                    ext = extension.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
                elif content_type:
                    file_types[content_type] = file_types.get(content_type, 0) + 1
            else:  # Invalid URL
                invalid_urls.append(str(url))

    processing_time = (datetime.now() - start_time).total_seconds()

    return FileURLResponse(
        processed_urls=len(request.urls),
        valid_urls=valid_urls,
        invalid_urls=invalid_urls,
        file_types=file_types,
        total_size=total_size,
        processing_time=processing_time
    )


@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "name": "File URL Parser API",
        "version": "1.0.0",
        "endpoints": {
            "/parse-urls": "POST - Parse and validate file URLs",
            "/": "GET - This information"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)