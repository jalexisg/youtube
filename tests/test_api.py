import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the heavy dependencies BEFORE importing web_server
with patch.dict(os.environ, {"FORCE_IMPORT_DEPENDENCIES": "0"}):
    # Mock AudioTranscriberSummarizer
    sys.modules["audio_transcriber_summarizer"] = MagicMock()
    from web_server import app, tasks

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

@patch("web_server.AudioTranscriberSummarizer")
def test_process_file(mock_ats):
    # Setup mock
    mock_instance = MagicMock()
    mock_ats.return_value = mock_instance
    mock_instance.process_media_file.return_value = {
        "transcription": {"full_text": "test text"},
        "analysis": {"extractive_summary": "summary", "keywords": ["test"]}
    }
    
    # Override the global transcriber getter or just patch the class used in the task
    # Since the task runs in background, TestClient might not wait for it unless we force it?
    # TestClient with BackgroundTasks: https://fastapi.tiangolo.com/tutorial/background-tasks/#testing-background-tasks
    # Actually TestClient runs background tasks synchronously by default? No, it doesn't wait?
    # Starlette TestClient runs background tasks.
    
    # Create a dummy file
    files = {'file': ('test.mp3', b'test content', 'audio/mpeg')}
    response = client.post("/api/process-file", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    task_id = data["task_id"]
    
    # Check status (might be queued or processing depending on how fast it runs)
    response = client.get(f"/api/tasks/{task_id}")
    assert response.status_code == 200
    assert response.json()["id"] == task_id

@patch("web_server.YoutubeDL")
def test_process_url(mock_ydl):
    # Setup mock
    mock_ydl_instance = MagicMock()
    mock_ydl.return_value.__enter__.return_value = mock_ydl_instance
    mock_ydl_instance.extract_info.return_value = {"title": "test video", "ext": "mp4"}
    mock_ydl_instance.prepare_filename.return_value = "test_video.mp4"
    
    # We also need to mock os.path.exists for the downloaded file check
    with patch("os.path.exists", return_value=True):
        response = client.post("/api/process-url", json={"url": "http://youtube.com/watch?v=123"})
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
