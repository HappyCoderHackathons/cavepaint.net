import asyncio
import json
import os
from pathlib import Path

import av
import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

from stereo_drawing import StereoDrawingTracker, find_cameras

# ---------------------------------------------------------------------------
# Video track
# ---------------------------------------------------------------------------

class StereoVideoTrack(VideoStreamTrack):
    def __init__(self, tracker: StereoDrawingTracker):
        super().__init__()
        self.tracker = tracker

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = self.tracker.get_frame()
        if frame is None:
            frame = np.zeros((480, 1280, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

tracker = StereoDrawingTracker(
    cam0=int(os.getenv("CAM0", "2")),
    cam1=int(os.getenv("CAM1", "1")),
)
tracker.start()

pcs: set[RTCPeerConnection] = set()

TEMPLATE = (Path(__file__).parent / "templates" / "stream.html").read_text()


async def index(request):
    return web.Response(content_type="text/html", text=TEMPLATE)


async def offer(request):
    try:
        params = await request.json()
        sdp_offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_state_change():
            if pc.connectionState in ("failed", "closed"):
                await pc.close()
                pcs.discard(pc)

        pc.addTrack(StereoVideoTrack(tracker))

        await pc.setRemoteDescription(sdp_offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
        )
    except Exception as exc:
        raise web.HTTPInternalServerError(reason=str(exc))


async def clear(request):
    tracker.clear_canvas()
    return web.Response(status=204)


async def undo(request):
    tracker.undo()
    return web.Response(status=204)


async def cameras(request):
    found = find_cameras()
    return web.Response(content_type="application/json", text=json.dumps(found))


async def on_shutdown(app):
    tracker.stop()
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()


app = web.Application()
app.on_shutdown.append(on_shutdown)
app.router.add_get("/", index)
app.router.add_post("/offer", offer)
app.router.add_post("/clear", clear)
app.router.add_post("/undo", undo)
app.router.add_get("/cameras", cameras)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
