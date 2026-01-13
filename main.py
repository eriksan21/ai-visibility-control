"""
FastAPI backend for AI Visibility Control service.
Privacy-first: no storage, no logging of image content.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import logging
from processor import process_face_image

# Configure logging (no image content logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Visibility Control",
    description="Give users control over how AI models perceive their images",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported formats
ALLOWED_FORMATS = {'image/jpeg', 'image/jpg', 'image/png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend UI."""
    # In production, serve from static files
    # For MVP, return simple HTML
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Visibility Control</title>
        <meta charset="UTF-8">
    </head>
    <body>
        <h1>AI Visibility Control</h1>
        <p>API is running. Use POST /process to upload images.</p>
        <p>See <a href="/docs">/docs</a> for API documentation.</p>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-visibility-control"}


@app.post("/process")
async def process_image(
    image: UploadFile = File(...),
    mode: str = Form(default="genai_safe")
):
    """
    Process an image to reduce AI recognition consistency.
    
    Args:
        image: Image file (jpg, png)
        mode: Privacy mode ('social_safe', 'genai_safe', 'max_privacy')
    
    Returns:
        Processed image file
    """
    # Log request (no image content)
    logger.info(f"Processing request - mode: {mode}, filename: {image.filename}")
    
    # Validate file type
    if image.content_type not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(ALLOWED_FORMATS)}"
        )
    
    # Validate mode
    valid_modes = {'social_safe', 'genai_safe', 'max_privacy'}
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Allowed: {', '.join(valid_modes)}"
        )
    
    # Read image bytes (in memory only)
    try:
        img_bytes = await image.read()
        
        # Check file size
        if len(img_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File too large. Max size: 10MB"
            )
        
        # Process image
        result_bytes, metadata = process_face_image(img_bytes, mode)
        
        # Log success (no image content)
        logger.info(
            f"Processing successful - faces: {metadata['faces_processed']}, "
            f"zones: {metadata['zones_modified']}"
        )
        
        # Return processed image
        return StreamingResponse(
            io.BytesIO(result_bytes),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename=processed_{image.filename}",
                "X-Faces-Processed": str(metadata['faces_processed']),
                "X-Zones-Modified": str(metadata['zones_modified']),
                "X-Mode": metadata['mode']
            }
        )
        
    except ValueError as e:
        # Handle processing errors
        logger.warning(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal processing error. Please try again."
        )
    
    finally:
        # Explicit cleanup (though Python GC handles this)
        del img_bytes
        await image.close()


@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...)):
    """
    Analyze image without processing (returns face count only).
    Useful for previewing before processing.
    """
    logger.info(f"Analysis request - filename: {image.filename}")
    
    try:
        img_bytes = await image.read()
        
        if len(img_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Detect faces only
        from detector import FaceDetector
        import cv2
        import numpy as np
        
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image format")
        
        detector = FaceDetector()
        faces = detector.detect_faces(img)
        
        return {
            "faces_detected": len(faces),
            "processable": len(faces) > 0,
            "image_size": {
                "width": img.shape[1],
                "height": img.shape[0]
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")
    finally:
        del img_bytes
        await image.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
