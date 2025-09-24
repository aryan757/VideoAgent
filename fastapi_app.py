"""
FastAPI Application for Variphi VGI Chat Interface

This application provides two main endpoints:
1. Welcome endpoint - Returns a welcome message
2. Chat endpoint - Handles questions using IntelligentToolRouter with conversation memory

Author: Aryan
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import requests
import uuid
from pathlib import Path
import shutil
import cv2
from agent_testing import IntelligentToolRouter
from variphi_reid import PersonReIdentificationTrackerONNX, track_person_in_video


# Initialize FastAPI app
app = FastAPI(
    title="Variphi VGI Chat Interface",
    description="AI-powered chat interface for surveillance data analysis",
    version="1.0.0"
)

# Allow UI origins (dev: "*" ; prod: set domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # replace with front-end origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global router instance (initialized once when app starts)
router = None

# Helper Functions
def download_image_from_url(image_url: str, save_path: str) -> bool:
    """
    Download an image from URL and save it to the specified path
    
    Args:
        image_url (str): URL of the image to download
        save_path (str): Local path where image should be saved
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        return True
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

################################################################################################################################

def download_video_from_url(video_url: str, save_path: str) -> bool:
    """
    Download a video from URL and save it to the specified path
    
    Args:
        video_url (str): URL of the video to download
        save_path (str): Local path where video should be saved
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False

#################################################################################################################################

def create_person_folder(id_name: str) -> str:
    """
    Create a folder for storing person's images
    
    Args:
        id_name (str): Name/ID of the person
        
    Returns:
        str: Path to the created folder
    """
    folder_path = f"./persons/{id_name}"
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


#########################################################################################################################

def track_person_fast_detection(person_folder_path: str, video_path: str, person_name: str) -> bool:
    """
    Fast person detection function that stops as soon as person is found once
    Uses existing embeddings from database instead of re-creating them
    
    Args:
        person_folder_path (str): Path to person's reference images folder
        video_path (str): Path to video file
        person_name (str): Name of the person being tracked
        
    Returns:
        bool: True if person was detected (even once), False otherwise
    """
    try:
        # Initialize tracker with correct model paths - this loads existing database
        tracker = PersonReIdentificationTrackerONNX(
            yolo_model_path="./models/yolov8s.pt",
            reid_onnx_path="./models/resnet50_market1501_aicity156.onnx",
            database_folder="./database"
        )
        
        # Check if reference person exists in loaded database
        reference_id = f"ref_{person_name}"
        if reference_id not in tracker.reference_persons:
            print(f"❌ Reference embedding for {person_name} not found in database")
            print(f"💡 Available references: {list(tracker.reference_names.values())}")
            return False
        
        print(f"✅ Using existing embedding for {person_name} from database")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return False
        
        frame_count = 0
        person_name_lower = person_name.lower()
        
        print(f"🔍 Scanning video for {person_name}...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Track persons in current frame
            detections = tracker.track_persons_with_reference(frame)
            
            # Check if target person is detected
            for person_id, box, similarity, detected_name, is_reference in detections:
                if is_reference and detected_name.lower() == person_name_lower:
                    print(f"🎉 FOUND {person_name.upper()} at frame {frame_count}! Stopping scan.")
                    cap.release()
                    return True  # Person detected! Stop immediately
            
            frame_count += 1
            
            # Optional: Print progress every 100 frames
            if frame_count % 5 == 0:
                print(f"📊 Processed {frame_count} frames, still searching...")
        
        cap.release()
        print(f"❌ {person_name.upper()} not detected in entire video ({frame_count} frames)")
        return False  # Person not found in entire video
        
    except Exception as e:
        print(f"❌ Error in tracking: {e}")
        return False

#############################################################################################################################

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str
    video_url: Optional[str] = None
    camera_id: Optional[str] = None
    csv_file_path: Optional[str] = "merged_events.csv"

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    success: bool
    message: Optional[str] = None
    selected_tool: Optional[str] = None
    query: Optional[str] = None
    error: Optional[str] = None
    #video_urls: Optional[List[str]] = None

class WelcomeResponse(BaseModel):
    """Response model for welcome endpoint"""
    message: str
    status: str

class EmbeddingRequest(BaseModel):
    """Request model for embedding creation endpoint"""
    image_url: Optional[str] = None
    id_name: str

class EmbeddingResponse(BaseModel):
    """Response model for embedding creation endpoint"""
    success: bool
    message: str
    id_name: str
    images_saved: Optional[int] = None
    error: Optional[str] = None

class TrackPersonRequest(BaseModel):
    """Request model for person tracking endpoint"""
    video_urls: List[str]
    id_name: str

class VideoResult(BaseModel):
    """Individual video tracking result"""
    id_name: str
    video: str
    status: str  # "detected" or "not_detected"
    detection_count: Optional[int] = None

class TrackPersonResponse(BaseModel):
    """Response model for person tracking endpoint"""
    success: bool
    message: str
    results: Optional[List[VideoResult]] = None
    error: Optional[str] = None


########################################################################################################################

@app.on_event("startup")
async def startup_event():
    """Initialize the IntelligentToolRouter when the app starts"""
    global router
    try:
        # Get API key from .env using dotenv
        from dotenv import load_dotenv
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        router = IntelligentToolRouter(google_api_key)
        print("✅ IntelligentToolRouter initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing router: {e}")
        raise e

@app.get("/", response_model=WelcomeResponse)
async def welcome():
    """
    Welcome endpoint - Returns a welcome message
    
    Returns:
        WelcomeResponse: Welcome message and status
    """
    return WelcomeResponse(
        message="Welcome to the Variphi VGI chat interface! 🚀",
        status="ready"
    )

@app.get("/welcome", response_model=WelcomeResponse)
async def welcome_alt():
    """
    Alternative welcome endpoint
    
    Returns:
        WelcomeResponse: Welcome message and status
    """
    return WelcomeResponse(
        message="Welcome to the Variphi VGI chat interface! 🚀",
        status="ready"
    )

@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    id_name: str = Form(...),
    image_url: str = Form(None),
    image_files: List[UploadFile] = File(None),
    image_file: List[UploadFile] = File(None)  # Support both names for compatibility
):
    """
    Create person embeddings from uploaded images or image URLs
    
    This endpoint allows you to:
    1. Upload an image file directly, OR
    2. Provide an image URL to download
    
    The images are saved in ./persons/{id_name}/ folder and 
    embeddings are created for person recognition.
    
    Args:
        id_name (str): Name/ID of the person (required)
        image_url (str): URL of image to download (optional)
        image_files (List[UploadFile]): Multiple uploaded image files (optional)
        
    Returns:
        EmbeddingResponse: Success status and embedding creation results
        
    Example usage:
        # With image URL (form-data)
        POST /embeddings
        Form data: id_name="john_doe", image_url="https://example.com/person.jpg"
        
        # With multiple file uploads (form-data)
        POST /embeddings
        Form data: id_name="john_doe", image_files=<file1>, image_files=<file2>, image_files=<file3>
    """
    try:
        file_names = [f.filename for f in image_files if f.filename] if image_files else []
        print(f"📥 Received request: id_name='{id_name}', image_url='{image_url}', image_files={file_names}")
        
        # id_name is now required via Form(...), so no need to check if it exists
            
        # Create person folder
        folder_path = create_person_folder(id_name)
        print(f"📁 Created folder: {folder_path}")
        
        images_saved = 0
        
        # Handle image URL download
        if image_url:
            print(f"🌐 Downloading image from URL: {image_url}")
            
            # Generate unique filename
            file_extension = image_url.split('.')[-1] if '.' in image_url else 'jpg'
            filename = f"image_{uuid.uuid4().hex[:8]}.{file_extension}"
            save_path = os.path.join(folder_path, filename)
            
            # Download image
            if download_image_from_url(image_url, save_path):
                images_saved += 1
                print(f"✅ Image saved: {filename}")
            else:
                raise HTTPException(status_code=400, detail="Failed to download image from URL")
        
        # Handle uploaded files (multiple files support)
        if image_files:
            for i, image_file in enumerate(image_files):
                if image_file.filename:  # Skip empty file slots
                    print(f"📤 Processing uploaded file {i+1}/{len(image_files)}: {image_file.filename}")
                    
                    # Generate unique filename
                    file_extension = image_file.filename.split('.')[-1] if '.' in image_file.filename else 'jpg'
                    filename = f"image_{uuid.uuid4().hex[:8]}.{file_extension}"
                    save_path = os.path.join(folder_path, filename)
                    
                    # Save uploaded file
                    with open(save_path, "wb") as buffer:
                        shutil.copyfileobj(image_file.file, buffer)
                    images_saved += 1
                    print(f"✅ File {i+1} saved: {filename}")
        
        # Check if at least one image was provided
        if images_saved == 0:
            raise HTTPException(
                status_code=400, 
                detail="Either image_url or image_files must be provided"
            )
        
        # Create embeddings using variphi_reid
        print(f"🧠 Creating embeddings for person: {id_name}")
        
        # Initialize tracker with correct model paths
        tracker = PersonReIdentificationTrackerONNX(
            yolo_model_path="./models/yolov8s.pt",
            reid_onnx_path="./models/resnet50_market1501_aicity156.onnx",
            database_folder="./database"
        )
        
        # Create base embedding from folder
        success = tracker.create_base_embedding_from_folder(folder_path)
        
        if success:
            return EmbeddingResponse(
                success=True,
                message=f"✅ Successfully created embeddings for {id_name}",
                id_name=id_name,
                images_saved=images_saved
            )
        else:
            return EmbeddingResponse(
                success=False,
                message=f"❌ Failed to create embeddings for {id_name}",
                id_name=id_name,
                images_saved=images_saved,
                error="Embedding creation failed"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        return EmbeddingResponse(
            success=False,
            message="An error occurred while creating embeddings",
            id_name=id_name if 'id_name' in locals() else "unknown",
            error=str(e)
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - Handles questions and provides AI-powered responses
    
    This endpoint:
    1. Takes a question from the user
    2. Routes it to the appropriate tool (question_query, video_analysis, or graph_plotting)
    3. Returns the response with conversation memory maintained
    
    Args:
        request (ChatRequest): Contains question and optional parameters
        
    Returns:
        ChatResponse: AI response with tool information
        
    Example requests:
        - {"question": "count the number of idle workers in the last events?"}
        - {"question": "plot the trend for last 3 days", "csv_file_path": "merged_events.csv"}
        - {"question": "is there violence in this video?", "video_url": "https://example.com/video.mp4"}
    """
    try:
        # Check if router is initialized
        if router is None:
            raise HTTPException(status_code=500, detail="Router not initialized")
        
        # Prepare parameters for the router
        kwargs = {}
        if request.video_url:
            kwargs["video_url"] = request.video_url
        if request.camera_id:
            kwargs["camera_id"] = request.camera_id
        if request.csv_file_path:
            kwargs["csv_file_path"] = request.csv_file_path
        
        # Route the query using IntelligentToolRouter
        # enhanced_query = router.enhanced_user_query(request.question)
        # print("enhanced_query=======>", enhanced_query)
       # router = IntelligentToolRouter(google_api_key)
        result = router.route_query(request.question, **kwargs)


        print("result=======>", result)

        if result["graph_url"]:
            result = result["graph_url"]

        
        # Check if routing was successful
        if result.get("success", False):
            return ChatResponse(
                success=True,
                message=str(result.get("result", "No response generated")),
                selected_tool=result.get("selected_tool"),
                query=result.get("query")
            )
  
        else:
            return ChatResponse(
                success=False,
                message="Failed to process your question",
                error=result.get("error", "Unknown error occurred")
            )
            
    except Exception as e:
        # Handle any unexpected errors
        return ChatResponse(
            success=False,
            message="An error occurred while processing your request",
            error=str(e)
        )


@app.post("/track_person", response_model=TrackPersonResponse)
async def track_person_in_videos(request: TrackPersonRequest):
    """
    Track a specific person across multiple videos
    
    This endpoint:
    1. Takes a list of video URLs/paths and a person ID
    2. Uses the person's pre-created embeddings to track them in each video
    3. Returns detection results for each video
    
    Args:
        request (TrackPersonRequest): Contains video_urls list and id_name
        
    Returns:
        TrackPersonResponse: Results showing if person was detected in each video
        
    Example request:
        {
            "video_urls": [
                "https://example.com/video1.mp4",
                "./local_video.mp4"
            ],
            "id_name": "john_doe"
        }
        
    Example response:
        {
            "success": true,
            "message": "Tracking completed for 2 videos",
            "results": [
                {
                    "id_name": "john_doe",
                    "video": "video1.mp4", 
                    "status": "detected",
                    "detection_count": 15
                },
                {
                    "id_name": "john_doe",
                    "video": "local_video.mp4",
                    "status": "not_detected",
                    "detection_count": 2
                }
            ]
        }
    """
    try:
        id_name = request.id_name
        video_urls = request.video_urls
        
        if not id_name:
            raise HTTPException(status_code=400, detail="id_name is required")
            
        if not video_urls or len(video_urls) == 0:
            raise HTTPException(status_code=400, detail="At least one video_url is required")
        
        # Check if person folder exists
        person_folder_path = f"./persons/{id_name}"
        if not os.path.exists(person_folder_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Person '{id_name}' not found. Please create embeddings first using /embeddings endpoint"
            )
        
        print(f"🔍 Starting tracking for person: {id_name}")
        print(f"📹 Processing {len(video_urls)} videos")
        
        results = []
        
        # Process each video
        for i, video_url in enumerate(video_urls):
            print(f"\n📹 Processing video {i+1}/{len(video_urls)}: {video_url}")
            
            try:
                # Prepare video path
                video_path = video_url
                
                # Download video if it's a URL
                if video_url.startswith(('http://', 'https://')):
                    print(f"🌐 Downloading video from URL...")
                    
                    # Generate temporary filename
                    video_extension = video_url.split('.')[-1] if '.' in video_url else 'mp4'
                    temp_filename = f"temp_video_{uuid.uuid4().hex[:8]}.{video_extension}"
                    video_path = f"./temp_videos/{temp_filename}"
                    
                    # Create temp directory
                    os.makedirs("./temp_videos", exist_ok=True)
                    
                    # Download video
                    if not download_video_from_url(video_url, video_path):
                        results.append(VideoResult(
                            id_name=id_name,
                            video=os.path.basename(video_url),
                            status="error - download failed",
                            detection_count=0
                        ))
                        continue
                    
                    print(f"✅ Video downloaded: {temp_filename}")
                
                # Track person in video using the existing function
                print(f"🔍 Tracking {id_name} in video...")
                
                # Custom tracking function - stops as soon as person is detected
                is_detected = track_person_fast_detection(person_folder_path, video_path, id_name)
                
                # Status based on whether person was found
                status = "detected" if is_detected else "not_detected"
                
                results.append(VideoResult(
                    id_name=id_name,
                    video=os.path.basename(video_url),
                    status=status,
                    detection_count=1 if is_detected else 0
                ))
                
                print(f"✅ {id_name} {'detected' if is_detected else 'not detected'} in {os.path.basename(video_url)}")
                
                # Clean up temporary video file
                if video_url.startswith(('http://', 'https://')) and os.path.exists(video_path):
                    os.remove(video_path)
                    print(f"🗑️ Cleaned up temporary file")
                    
            except Exception as video_error:
                print(f"❌ Error processing video {video_url}: {video_error}")
                results.append(VideoResult(
                    id_name=id_name,
                    video=os.path.basename(video_url),
                    status="error",
                    detection_count=0
                ))
        
        # Summary
        total_videos = len(results)
        detected_videos = len([r for r in results if r.status == "detected"])
        
        return TrackPersonResponse(
            success=True,
            message=f"✅ Tracking completed for {total_videos} videos. {id_name} detected in {detected_videos}/{total_videos} videos.",
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return TrackPersonResponse(
            success=False,
            message="An error occurred while tracking person in videos",
            error=str(e)
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the application is running
    
    Returns:
        dict: Health status
    """
    return {
        "status": "healthy",
        "router_initialized": router is not None,
        "message": "Variphi VGI Chat Interface is running!"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Variphi VGI Chat Interface...")
    print("📝 Available endpoints:")
    print("   GET  / - Welcome message")
    print("   GET  /welcome - Alternative welcome message")
    print("   POST /chat - Chat with AI (send questions)")
    print("   POST /embeddings - Create person embeddings from images")
    print("   POST /track_person - Track person in videos")
    print("   GET  /health - Health check")
    print("   GET  /docs - API documentation (Swagger UI)")
    print("   GET  /redoc - Alternative API documentation")
    
    # Run the application
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
