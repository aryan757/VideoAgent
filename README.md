## Variphi VideoAgent (FastAPI)

AI-powered video and data analysis service for the Variphi VGI stack. Exposes a FastAPI server that:

- Handles natural-language queries and routes them to tools (CSV Q&A, video analysis, graph plotting)
- Creates person embeddings from images for re-identification
- Tracks a specific person across one or more videos
- Provides health and welcome endpoints, and interactive docs

Requires Python 3.11.

---

### Table of contents
- Overview
- Requirements
- Installation (Python 3.11)
- Environment & Credentials
- Running the server
- API Reference
  - GET `/` and GET `/welcome`
  - GET `/health`
  - POST `/chat`
  - POST `/embeddings`
  - POST `/track_person`
- Models, data, and filesystem layout
- Security and .gitignore tips
- Troubleshooting

---

### Overview
The service centers around `IntelligentToolRouter`, which uses Google Gemini models via LangChain to route user questions to one of several tools:

- question_query: CSV Q&A using LangChain’s CSV agent
- video_analysis: Sends a video to the Gemini API for content understanding
- graph_plotting: Generates plots from CSV analysis and uploads the image to a GCS bucket
- greeting_agent: Simple conversational responses

Core modules/files:
- `fastapi_app.py`: FastAPI app, endpoints, startup initialization
- `agent_testing.py`: Implements `IntelligentToolRouter` and the tool logic
- `variphi_reid` (imported): Person re-identification utilities and tracker

---

### Requirements
- Python 3.11
- pip
- System packages commonly needed by OpenCV (platform-specific)
- Network access to download videos or call external APIs

Python libraries are listed in the repository’s `requirements.txt` (install steps below).

---

### Installation (Python 3.11)
```bash
# From the repo root (macOS example)
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- If OpenCV fails to install, install system dependencies for your OS (e.g., `brew install opencv` on macOS, or appropriate packages on Linux), then `pip install opencv-python`.

---

### Environment & Credentials
Copy `VideoAgent/.exampleenv` to `.env` and fill values:

```env
GOOGLE_API_KEY="<your-google-api-key>"
./generative-ai-variphi-098b98607a93.json
```

Required values:
- `GOOGLE_API_KEY`: Google Generative AI API key (Gemini).
- GCP service account JSON file: The code expects the file `VideoAgent/generative-ai-variphi-098b98607a93.json` to exist and uses it to upload graphs to the `vgidata` bucket. Ensure the service account has `storage.objects.create` permission for that bucket.

Optional (recommended to externalize):
- GCS Bucket: Currently hardcoded in code as `vgidata` inside `graph_plotting_agent`. Change it there if needed.

Keep credentials out of version control. See the .gitignore tips below.

---

### Running the server
```bash
cd VideoAgent
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

### API Reference

All responses conform to Pydantic models defined in `fastapi_app.py`.

#### GET `/` and GET `/welcome`
Simple welcome message and readiness indicator.

Response example:
```json
{
  "message": "Welcome to the Variphi VGI chat interface! 🚀",
  "status": "ready"
}
```

#### GET `/health`
Basic health check.

Response example:
```json
{
  "status": "healthy",
  "router_initialized": true,
  "message": "Variphi VGI Chat Interface is running!"
}
```

#### POST `/chat`
Routes a natural-language question to the right tool and returns the result.

Request body (JSON):
```json
{
  "question": "count the number of idle workers in the last events?",
  "video_url": "https://example.com/sample.mp4",
  "camera_id": "cam-42",
  "csv_file_path": "merged_events.csv"
}
```

Notes:
- Only `question` is required.
- If a valid `video_url` (http/https) is provided and the question is about video content, the video tool is selected.
- If the query is about plots/charts, the graph plotting tool is selected. It will save a PNG locally and upload it to GCS, returning a URL.
- CSV defaults to `merged_events.csv` relative to the working dir.

Response example (success):
```json
{
  "success": true,
  "message": "<tool result or text>",
  "selected_tool": "question_query",
  "query": "count the number of idle workers in the last events?",
  "error": null
}
```

#### POST `/embeddings`
Create a person’s embeddings from images. Uses `variphi_reid.PersonReIdentificationTrackerONNX` under the hood.

Content-Type: `multipart/form-data`

Form fields:
- `id_name` (string, required): Person ID/name. Images are saved to `./persons/<id_name>/`.
- `image_url` (string, optional): Download an image from a URL.
- `image_files` (file[], optional): One or more uploaded image files.

Example (curl, upload two files):
```bash
curl -X POST "http://localhost:8000/embeddings" \
  -F id_name=john_doe \
  -F image_files=@/path/to/image1.jpg \
  -F image_files=@/path/to/image2.png
```

Example (curl, URL download):
```bash
curl -X POST "http://localhost:8000/embeddings" \
  -F id_name=john_doe \
  -F image_url=https://example.com/john.jpg
```

Successful response:
```json
{
  "success": true,
  "message": "✅ Successfully created embeddings for john_doe",
  "id_name": "john_doe",
  "images_saved": 2,
  "error": null
}
```

Requirements for this endpoint:
- Model files present under `./models/`:
  - `yolov8s.pt`
  - `resnet50_market1501_aicity156.onnx`
- Embeddings database directory at `./database` (auto-created/used by the tracker)

#### POST `/track_person`
Track a pre-embedded person across one or more videos. Stops scanning a video as soon as the person is detected once (fast path).

Request body (JSON):
```json
{
  "video_urls": [
    "https://example.com/video1.mp4",
    "./local_video.mp4"
  ],
  "id_name": "john_doe"
}
```

Notes:
- The person folder `./persons/<id_name>` must exist with previously created embeddings (via `/embeddings`).
- For http/https URLs, videos are downloaded to `./temp_videos/` temporarily and removed after processing.

Response example:
```json
{
  "success": true,
  "message": "✅ Tracking completed for 2 videos. john_doe detected in 1/2 videos.",
  "results": [
    { "id_name": "john_doe", "video": "video1.mp4", "status": "detected", "detection_count": 1 },
    { "id_name": "john_doe", "video": "local_video.mp4", "status": "not_detected", "detection_count": 0 }
  ]
}
```

---

### Models, data, and filesystem layout
- `./models/yolov8s.pt` and `./models/resnet50_market1501_aicity156.onnx` must exist for detection and re-identification.
- `./database/` stores embeddings created for persons.
- `./persons/<id_name>/` keeps source/reference images for each person.
- `merged_events.csv` is the default CSV file for Q&A and plotting.
- Graph plotting saves a local PNG (e.g., `Variphi_generated_plot_XXXX.png`) and uploads it to the `vgidata` GCS bucket using the service account JSON at `VideoAgent/generative-ai-variphi-098b98607a93.json`.

---

### Security and .gitignore tips
Strongly consider ignoring sensitive and large artifacts. For example, in `VideoAgent/.gitignore`:
```gitignore
videoagent/
agent_testing.ipynb
generative-ai-variphi-098b98607a93.json
.env
*.mp4
*.ts
*.ipynb_checkpoints/
temp_videos/
persons/
database/
models/
```
If any of these are already tracked, untrack them:
```bash
git rm -r --cached VideoAgent/generative-ai-variphi-098b98607a93.json VideoAgent/agent_testing.ipynb
```

---

### Troubleshooting
- Router not initialized: Ensure `.env` contains a valid `GOOGLE_API_KEY` and that the app’s startup runs (see logs). Restart the server.
- Graph URL missing: Ensure the service account JSON exists at `VideoAgent/generative-ai-variphi-098b98607a93.json` and that it has permission to write to the `vgidata` bucket. Check that the bucket is accessible or public as needed.
- Video analysis issues: The `video_analysis` tool downloads the video to the current dir and uploads it to Gemini. Large files may take time; ensure network access. Check console logs for the processing state.
- Embedding creation errors: Verify required model files exist inside `./models/`. Ensure OpenCV can open image files and that images are valid.
- OpenCV backend errors: Install system packages for OpenCV or try `pip install opencv-python-headless` on headless systems.

---

Happy building! If you modify bucket names, file paths, or credential locations, update them in code (`agent_testing.py` and related modules) or externalize into environment variables for cleaner configuration.


