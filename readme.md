# AI Visibility Control Service

**Give users control over how AI models perceive their images.**

A privacy-focused web service that processes face regions in photos to reduce AI-based recognition consistency while maintaining human-readable quality.

---

## 🎯 What This Does

- **Reduces AI recognition** of faces in photos through localized transformations
- **Preserves human perception** - images remain visually acceptable
- **Privacy-first design** - no storage, no logging, in-memory processing only
- **No biometric processing** - only detects face bounding boxes, no identity matching

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd ai-visibility-control

# Install dependencies
pip install -r backend/requirements.txt

# Run server
cd backend
python main.py
```

Server runs at `http://localhost:8000`

### Docker Deployment (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t ai-visibility-control .
docker run -p 8000:8000 ai-visibility-control
```

---

## 📡 API Reference

### `POST /process`

Process an image to reduce AI recognition.

**Request:**
```bash
curl -X POST http://localhost:8000/process \
  -F "image=@photo.jpg" \
  -F "mode=genai_safe" \
  --output protected.jpg
```

**Parameters:**
- `image` (file, required): Image file (JPG/PNG, max 10MB)
- `mode` (string, optional): Privacy mode
  - `social_safe`: Light protection (blur ~3px)
  - `genai_safe`: Moderate protection (blur ~5px) [default]
  - `max_privacy`: Strong protection (blur ~7px)

**Response:**
- Processed image file (JPEG)
- Headers:
  - `X-Faces-Processed`: Number of faces processed
  - `X-Zones-Modified`: Number of face zones modified
  - `X-Mode`: Applied privacy mode

**Error Responses:**
- `400`: Invalid format, no faces detected, or invalid mode
- `413`: File too large (>10MB)
- `500`: Internal processing error

### `POST /analyze`

Analyze image without processing (preview).

**Request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "image=@photo.jpg"
```

**Response:**
```json
{
  "faces_detected": 2,
  "processable": true,
  "image_size": {
    "width": 1920,
    "height": 1080
  }
}
```

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "ai-visibility-control"
}
```

---

## 🧠 How It Works

### Processing Pipeline

1. **Face Detection** (OpenCV Haar Cascade)
   - Detects face bounding boxes only
   - No recognition, no identity matching
   - Classical computer vision (non-ML approach)

2. **Sub-Zone Identification**
   - Eyes (highest priority for AI recognition)
   - Nose bridge (secondary priority)
   - Zones are estimated within detected faces

3. **Localized Transformations**
   - **Bilateral blur**: Edge-preserving blur maintains human perception
   - **Luminance noise**: Adds subtle noise to L channel (LAB color space)
   - **Asymmetric shifts**: Small pixel shifts break pattern matching
   - **Feathered blending**: Smooth transitions to avoid visible boundaries

4. **Quality Preservation**
   - No global effects (rest of image untouched)
   - No color overlays or visible masks
   - High JPEG quality (95%) for output

### Technical Approach

```python
# Simplified flow
faces = detect_face_bboxes(image)  # Haar Cascade
for face in faces:
    zones = extract_zones(face)  # eyes, nose_bridge
    for zone in zones:
        zone = apply_blur(zone)          # Bilateral filter
        zone = add_luminance_noise(zone)  # LAB color space
        zone = apply_asymmetry(zone)      # Warp transform
        blend_back(zone)                  # Feathered edges
```

---

## 🛡️ Privacy & Security

### Privacy Guarantees

✅ **No storage**: Images processed in RAM only  
✅ **No logging**: Image content never logged  
✅ **No persistence**: Auto-cleanup after response  
✅ **No face recognition**: Only bounding box detection  
✅ **No identity matching**: Zero biometric processing  

### What We DON'T Do

❌ No face recognition or identification  
❌ No database or file system storage  
❌ No cloud uploads or third-party APIs  
❌ No user accounts or tracking  
❌ No metadata extraction beyond basic image properties  

### Security Considerations

- **File size limits**: 10MB max to prevent DoS
- **Format validation**: Only JPG/PNG accepted
- **CORS**: Configure `allow_origins` for production
- **Rate limiting**: Implement at reverse proxy level
- **HTTPS**: Always deploy behind TLS in production

---

## ⚙️ Configuration

### Environment Variables

```bash
# .env file
MAX_FILE_SIZE=10485760  # 10MB in bytes
ALLOWED_ORIGINS=https://yourdomain.com
LOG_LEVEL=INFO
```

### Processing Modes

Customize modes in `processor.py`:

```python
ProcessingMode(
    name='Custom Mode',
    blur_radius=5,          # Kernel size (odd number)
    noise_strength=0.04,    # 0-1 range
    asymmetry_shift=2       # Pixel shift
)
```

---

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/

# Test with sample image
curl -X POST http://localhost:8000/process \
  -F "image=@test_image.jpg" \
  -F "mode=genai_safe" \
  --output result.jpg
```

**Sample test image requirements:**
- Contains at least one frontal face
- Good lighting, clear features
- JPG or PNG format
- < 10MB file size

---

## 📊 Performance

**Typical processing times** (on modern hardware):
- 1 face: ~200-400ms
- 2-3 faces: ~400-800ms
- 4+ faces: ~800ms-1.5s

**Memory usage**:
- Base: ~150MB (OpenCV models loaded)
- Per request: +20-50MB (depends on image size)

**Optimization tips**:
- Use async workers (uvicorn with `--workers 4`)
- Deploy behind CDN for static assets
- Implement request queuing for high traffic
- Consider GPU acceleration for batch processing

---

## 🚧 Limitations

### Current Limitations

1. **Face detection accuracy**: May miss faces in poor lighting or extreme angles
2. **Side profiles**: Works best on frontal faces (Haar Cascade limitation)
3. **Small faces**: May not detect faces < 30x30 pixels
4. **Effectiveness**: Reduces AI consistency but doesn't guarantee anonymity
5. **Visual quality**: Some minor quality loss at Max Privacy mode

### Not Supported

- Video processing (frame-by-frame would work but is resource-intensive)
- Batch upload (process one image at a time)
- Real-time webcam processing
- Advanced ML-based face detection (intentionally using classical CV)

---

## 🔮 Future Improvements

### Roadmap

**Phase 1: Core Enhancements**
- [ ] Add support for profile/angled faces
- [ ] Implement adaptive processing based on image quality
- [ ] Add metadata stripping (EXIF removal)
- [ ] Support WebP format

**Phase 2: Advanced Features**
- [ ] Batch processing API endpoint
- [ ] Customizable processing zones (user-defined regions)
- [ ] Before/after comparison metrics
- [ ] Video frame processing

**Phase 3: Enterprise Features**
- [ ] API key authentication
- [ ] Usage analytics (privacy-preserving)
- [ ] White-label deployment options
- [ ] Integration SDKs (Python, JavaScript)

### Technical Debt

- Replace Haar Cascade with lightweight DNN model for better accuracy
- Implement proper request queuing system
- Add comprehensive integration tests
- Create performance benchmarking suite

---

## 📚 Usage Examples

### Python Client

```python
import requests

def protect_image(image_path, mode='genai_safe'):
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'mode': mode}
        
        response = requests.post(
            'http://localhost:8000/process',
            files=files,
            data=data
        )
        
    if response.ok:
        with open('protected.jpg', 'wb') as f:
            f.write(response.content)
        print(f"Faces processed: {response.headers['X-Faces-Processed']}")
    else:
        print(f"Error: {response.json()['detail']}")

protect_image('selfie.jpg', mode='max_privacy')
```

### JavaScript Client

```javascript
async function protectImage(file, mode = 'genai_safe') {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('mode', mode);
    
    const response = await fetch('http://localhost:8000/process', {
        method: 'POST',
        body: formData
    });
    
    if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        console.log('Faces:', response.headers.get('X-Faces-Processed'));
        return url;
    } else {
        const error = await response.json();
        throw new Error(error.detail);
    }
}
```

### cURL Examples

```bash
# Basic processing
curl -X POST http://localhost:8000/process \
  -F "image=@photo.jpg" \
  -F "mode=social_safe" \
  -o protected.jpg

# With verbose output
curl -X POST http://localhost:8000/process \
  -F "image=@photo.jpg" \
  -F "mode=max_privacy" \
  -o protected.jpg \
  -w "\nFaces: %{header_X-Faces-Processed}\nZones: %{header_X-Zones-Modified}\n"

# Analyze before processing
curl -X POST http://localhost:8000/analyze \
  -F "image=@photo.jpg" | jq
```

---

## 🤝 Contributing

We welcome contributions! Areas of focus:

1. **Accuracy**: Better face detection for edge cases
2. **Performance**: Optimization for batch processing
3. **Quality**: Improved visual quality preservation
4. **Testing**: More comprehensive test coverage

---

## ⚖️ Ethics & Responsible Use

### Design Philosophy

This tool is designed to give users **control**, not to enable harm. We believe individuals should have agency over how their images are used by AI systems.

### Appropriate Use Cases

✅ Protecting personal photos before social media upload  
✅ Reducing data collection by AI training systems  
✅ Privacy protection for sensitive groups (activists, journalists)  
✅ Educational purposes and research  

### Inappropriate Use Cases

❌ Evading law enforcement or legal proceedings  
❌ Facilitating identity fraud  
❌ Circumventing security systems  
❌ Harassment or stalking  

### Legal Disclaimer

This service:
- Does NOT guarantee complete anonymity
- Does NOT provide legal protection
- Should NOT be used for illegal purposes
- Is provided "as-is" without warranty

Users are responsible for complying with local laws regarding image manipulation and privacy.

---

## 📄 License

MIT License - see LICENSE file

**Attribution appreciated but not required.**

---

## 📞 Support

- **Documentation**: This README
- **API Docs**: http://localhost:8000/docs (when running)
- **Issues**: [GitHub Issues]
- **Security**: Report vulnerabilities privately to security@yourcompany.com

---

## 🙏 Acknowledgments

- OpenCV community for Haar Cascade classifiers
- FastAPI team for the excellent framework
- Privacy advocates and researchers in the adversarial ML space

---

**Built with privacy and user agency in mind.**
