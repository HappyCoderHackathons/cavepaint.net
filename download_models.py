import urllib.request
from pathlib import Path

MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
}

for filename, url in MODELS.items():
    path = Path(filename)
    if path.exists():
        print(f"  already exists: {filename}")
        continue
    print(f"  downloading {filename}...")
    urllib.request.urlretrieve(url, path)
    print(f"  saved {path.stat().st_size // 1024}KB -> {filename}")
