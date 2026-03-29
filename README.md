# CavePaint.NET

Draw with your hands in 3D space, like a caveman, but make it tech.

## Inspiration

Cave paintings are humanity's oldest form of visual communication, made with bare hands, on raw surfaces, with no tools but the body itself. We wanted to bring that primal, tactile experience into the modern era: what if you could paint with your hands in the air, using gesture and motion instead of a mouse or stylus?

## What it does

CavePaint.NET is a gesture-controlled digital whiteboard that lets you draw in mid-air using hand tracking and stereo vision. Two cameras triangulate your fingertip position in 3D space, while custom-trained ML models recognize gestures (point to draw, open hand to stop, fist to erase) and directional swipes (to switch colors or undo). Strokes are rendered with speed-adaptive width, Catmull-Rom spline smoothing, and Z-depth layering so drawings realistically appear in front of or behind the user. Sessions are streamed live to a student/observer view and persisted to the cloud.

## How we built it

- **Python + Flask/aiohttp** backend with WebRTC streaming via `aiortc`
- **OpenCV** for stereo camera capture and frame processing
- **MediaPipe** for real-time hand and pose landmark detection
- **PyTorch**: custom MLP for gesture classification, custom 1D CNN for swipe detection; both trained on data we collected ourselves
- **Stereo triangulation** with toe-in angle correction to get accurate 3D fingertip coordinates in inches
- **MongoDB Atlas** for session persistence: every stroke, point, and erase is recorded and replayable
- **HTML5 + Tailwind CSS + vanilla JS** frontend with a WebRTC video feed, drag-to-rotate 3D whiteboard, and cave-themed UI

## Challenges we ran into

- **Stereo toe-in correction**: Our cameras are angled inward for overlap, which breaks naive triangulation. We had to derive per-camera de-rotation math to correct pixel coordinates before computing disparity.
- **Latency vs. accuracy**: Running MediaPipe, a gesture MLP, and a swipe CNN every frame while maintaining 30 FPS required careful threading: separate camera reader threads, background rendering, and canvas caching to keep rendering O(active stroke) instead of O(all strokes).
- **Natural drawing feel**: Raw fingertip position is jittery. We layered exponential moving average smoothing, Catmull-Rom splines, and speed-based stroke width variation to make it feel like an actual drawing tool.

## Accomplishments that we're proud of

- Built and trained our own gesture and swipe models from scratch, collecting the datasets ourselves
- Real 3D depth from a stereo camera rig with custom calibration scripts
- Z-depth occlusion: drawings actually go "behind" you based on pose segmentation
- A fully functional live streaming mode where observers watch in real time on a stone-textured virtual whiteboard

## What we learned

- Stereo vision geometry is harder than it looks: small calibration errors compound into big 3D position errors
- Building ML models end-to-end (data collection, training, real-time inference) taught us a lot about the gap between research accuracy and production feel
- WebRTC from Python is painful but possible
- Design matters: the cave aesthetic is not just decoration, it makes the whole experience feel intentional and cohesive

## What's next for CavePaint.NET

- Multi-user collaborative drawing (multiple people painting simultaneously)
- Depth camera support (Intel RealSense) to replace the stereo rig
- Gallery and session replay UI for browsing past paintings
- Mobile/tablet gesture support via front-facing camera

## Technologies

Python, PyTorch, OpenCV, MediaPipe, Flask, aiohttp, aiortc, WebRTC, MongoDB Atlas, HTML5, Tailwind CSS, JavaScript, NumPy, scikit-learn, PyMongo, uv
