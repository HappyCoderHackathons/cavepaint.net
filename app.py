import asyncio
import json
import os
from pathlib import Path

import av
import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

from stereo_drawing import PALETTE, StereoDrawingTracker, find_cameras

# Convert BGR palette to CSS hex for the frontend
PALETTE_HEX = ["#{:02x}{:02x}{:02x}".format(r, g, b) for b, g, r in PALETTE]

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


async def state(request):
    s = tracker.get_state()
    events = [
        {"label": lbl, "color_idx": ci, "frames": f}
        for lbl, ci, f in s["swipe_events"]
    ]
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "color_idx": s["color_idx"],
            "palette": PALETTE_HEX,
            "swipe_events": events,
            "tracking": s.get("tracking", {}),
        }),
        headers={"Cache-Control": "no-store"},
    )


async def stream_state(request):
    loop = asyncio.get_event_loop()
    slot = tracker.subscribe(loop)
    response = web.StreamResponse(headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
    await response.prepare(request)
    try:
        while True:
            try:
                snapshot = await asyncio.wait_for(slot.get(), timeout=5.0)
            except asyncio.TimeoutError:
                await response.write(b": keepalive\n\n")
                continue
            events = [
                {"label": lbl, "color_idx": ci, "frames": f}
                for lbl, ci, f in snapshot["swipe_events"]
            ]
            data = json.dumps({
                "color_idx": snapshot["color_idx"],
                "palette": PALETTE_HEX,
                "swipe_events": events,
                "tracking": snapshot["tracking"],
            })
            await response.write(f"data: {data}\n\n".encode())
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        tracker.unsubscribe(slot)
    return response


async def set_color(request):
    try:
        data = await request.json()
        idx = int(data["idx"])
    except (KeyError, ValueError, TypeError):
        raise web.HTTPBadRequest(reason="Expected JSON body with integer 'idx'")
    tracker.set_color(idx)
    return web.Response(status=204)


async def cameras(request):
    found = find_cameras()
    return web.Response(content_type="application/json", text=json.dumps(found))


async def whiteboard(request):
    try:
        yaw = float(request.query.get("yaw", "0"))
        fov = float(request.query.get("fov", "80"))
        width = int(request.query.get("w", "960"))
        height = int(request.query.get("h", "260"))
    except ValueError:
        raise web.HTTPBadRequest(reason="Invalid whiteboard query params")

    frame = tracker.render_whiteboard(yaw_deg=yaw, fov_deg=fov, width=width, height=height)
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise web.HTTPInternalServerError(reason="Failed to encode whiteboard image")
    return web.Response(
        body=encoded.tobytes(),
        content_type="image/png",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


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
app.router.add_get("/state", state)
app.router.add_get("/stream", stream_state)
app.router.add_post("/color", set_color)
app.router.add_get("/whiteboard.png", whiteboard)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
