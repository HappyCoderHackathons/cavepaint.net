import base64
import asyncio
import json
import mimetypes
import os
import time
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError

import av
import cv2
import numpy as np
from aiohttp import ClientSession, ClientTimeout, web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

from stereo_drawing import PALETTE, StereoDrawingTracker, find_cameras
from stereo_drawing.mongo import drawings_col, points_col, erases_col
from session_store import SessionStore

# Convert BGR palette to CSS hex for the frontend
PALETTE_HEX = ["#{:02x}{:02x}{:02x}".format(r, g, b) for b, g, r in PALETTE]

# ---------------------------------------------------------------------------
# Video track
# ---------------------------------------------------------------------------

class StereoVideoTrack(VideoStreamTrack):
    def __init__(self, tracker: StereoDrawingTracker):
        super().__init__()
        self.tracker = tracker
        self._clock_rate = 90000
        requested_fps = int(os.getenv("STREAM_FPS", "30"))
        self._target_fps = max(5, min(90, requested_fps))
        self._ticks_per_frame = int(self._clock_rate / self._target_fps)
        self._pts = 0
        self._next_wall = time.monotonic()

    async def recv(self):
        now = time.monotonic()
        delay = self._next_wall - now
        if delay > 0:
            await asyncio.sleep(delay)
        self._next_wall = max(self._next_wall, time.monotonic()) + (1.0 / self._target_fps)

        frame = self.tracker.get_frame()
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Tracker frames are BGR already; avoid per-frame color conversion cost.
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        self._pts += self._ticks_per_frame
        video_frame.pts = self._pts
        video_frame.time_base = Fraction(1, self._clock_rate)
        return video_frame


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

def _parse_cam(env_var: str, default: str) -> int | str:
    val = os.getenv(env_var, default)
    return val if (val.startswith("http") or val.startswith("zmq://")) else int(val)

tracker = StereoDrawingTracker(
    cam0=_parse_cam("CAM0", "2"),
    cam1=_parse_cam("CAM1", "1"),
)
tracker.start()

pcs: set[RTCPeerConnection] = set()

TEMPLATE = (Path(__file__).parent / "templates" / "stream.html").read_text(encoding="utf-8")
STUDENT_TEMPLATE = (Path(__file__).parent / "templates" / "student.html").read_text(encoding="utf-8")
GALLERY_TEMPLATE = (Path(__file__).parent / "templates" / "gallery.html").read_text(encoding="utf-8")
REPLAY_TEMPLATE = (Path(__file__).parent / "templates" / "replay.html").read_text(encoding="utf-8")
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

session_store = (
    SessionStore(drawings_col, points_col, erases_col)
    if drawings_col is not None
    else None
)

_DEFAULT_REPLAY_PROMPT = (
    "Narrate this cave drawing frame in cinematic present tense. "
    "Keep it concise, vivid, and suitable for live playback commentary."
)


def _narration_chat_base_url() -> str:
    base = (
        os.getenv("NARRATION_API_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    )
    base = base.strip()
    if not base.startswith("http://") and not base.startswith("https://"):
        base = f"https://{base}"
    base = base.rstrip("/")
    # Accept either ".../chat/completions" or API base forms.
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    idx = base.find("/chat/completions")
    if idx != -1:
        return base[:idx]
    return base


def _narration_api_key() -> str | None:
    return (
        os.getenv("NARRATION_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_TOKEN")
    )


def _to_data_uri_from_image_url(image_url: str, max_bytes: int = 8_000_000) -> str:
    src = (image_url or "").strip()
    if not src:
        raise ValueError("image_url is required")
    if src.startswith("data:"):
        return src

    parsed = urlparse(src)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("image_url must start with http:// or https://")

    req = Request(src, headers={"User-Agent": "cavepaint-net/1.0"})
    with urlopen(req, timeout=15) as resp:
        raw = resp.read(max_bytes + 1)
        mime = ""
        if hasattr(resp.headers, "get_content_type"):
            mime = resp.headers.get_content_type() or ""
        if not mime:
            mime = (resp.headers.get("Content-Type") or "").split(";")[0].strip()

    if len(raw) > max_bytes:
        raise ValueError("image_url payload is too large (max 8MB)")

    if not mime or mime == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(src)
        mime = guessed or "image/jpeg"

    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
                continue
            if not isinstance(item, dict):
                continue
            txt = item.get("text")
            if isinstance(txt, str):
                out.append(txt)
        return "".join(out)
    return ""


def _extract_openai_delta_text(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return _extract_text_from_content(delta.get("content"))


def _extract_openai_message_text(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message") or {}
    return _extract_text_from_content(msg.get("content"))


async def _sse_write(response: web.StreamResponse, event: str, data: dict):
    try:
        await response.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8"))
    except ConnectionResetError:
        pass


async def index(request):
    return web.Response(content_type="text/html", text=TEMPLATE)


async def student(request):
    return web.Response(content_type="text/html", text=STUDENT_TEMPLATE)


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
            "canvas_version": s.get("canvas_version", 0),
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
                "canvas_version": snapshot.get("canvas_version", 0),
                "theta": _current_theta,
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


async def set_live_view(request):
    try:
        data = await request.json()
        yaw_raw = data.get("yaw")
        fov_raw = data.get("fov")
        if yaw_raw is None and fov_raw is None:
            raise web.HTTPBadRequest(reason="Expected 'yaw' and/or 'fov' in JSON body")
        yaw = float(yaw_raw) if yaw_raw is not None else None
        fov = float(fov_raw) if fov_raw is not None else None
    except (ValueError, TypeError, json.JSONDecodeError):
        raise web.HTTPBadRequest(reason="Invalid JSON body for live view")

    tracker.set_live_view(yaw_deg=yaw, fov_deg=fov)
    return web.Response(status=204)


async def replay_narrate_stream(request):
    response = web.StreamResponse(headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
    await response.prepare(request)

    try:
        data = await request.json()
    except json.JSONDecodeError:
        await _sse_write(response, "error", {"error": "Expected JSON body"})
        await response.write_eof()
        return response

    image_url = str(data.get("image_url") or "").strip()
    prompt = str(data.get("prompt") or _DEFAULT_REPLAY_PROMPT).strip()
    try:
        temperature = float(data.get("temperature", 0.8))
    except (TypeError, ValueError):
        temperature = 0.8
    temperature = max(0.0, min(2.0, temperature))

    if not image_url:
        await _sse_write(response, "error", {"error": "image_url is required"})
        await response.write_eof()
        return response

    base_url = _narration_chat_base_url()
    token = _narration_api_key()
    if not base_url or not token:
        await _sse_write(
            response,
            "error",
            {
                "error": (
                    "Set NARRATION_API_KEY (or OPENAI_API_KEY). "
                    "Optional: set NARRATION_API_BASE_URL (defaults to https://api.openai.com/v1)."
                )
            },
        )
        await response.write_eof()
        return response

    preferred_model = str(
        data.get("model")
        or os.getenv("NARRATION_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    ).strip()
    fallback_model = str(
        os.getenv("NARRATION_FALLBACK_MODEL")
        or os.getenv("OPENAI_FALLBACK_MODEL")
        or ""
    ).strip()
    models_to_try = [preferred_model]
    if fallback_model and fallback_model not in models_to_try:
        models_to_try.append(fallback_model)

    try:
        image_data_uri = await asyncio.to_thread(_to_data_uri_from_image_url, image_url)
    except Exception as exc:
        await _sse_write(response, "error", {"error": f"Failed to fetch image_url: {exc}"})
        await response.write_eof()
        return response

    endpoint = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    timeout = ClientTimeout(total=180)

    async with ClientSession(timeout=timeout) as session:
        for idx, model_name in enumerate(models_to_try):
            req_body = {
                "model": model_name,
                "stream": True,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a live cave-art narrator for replay mode. "
                            "Describe the frame as a short unfolding story."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                        ],
                    },
                ],
            }

            try:
                async with session.post(endpoint, headers=headers, json=req_body) as upstream:
                    if upstream.status >= 400:
                        detail = await upstream.text()
                        is_unknown_model = "unknown model" in detail.lower()
                        if is_unknown_model and idx < len(models_to_try) - 1:
                            continue
                        hint = None
                        if upstream.status == 404 and "<!doctype html>" in detail.lower():
                            hint = (
                                "The narration base URL looks like a web page, not an API base. "
                                "For OpenAI, use NARRATION_API_BASE_URL=https://api.openai.com/v1"
                            )
                        if upstream.status in (401, 403) and hint is None:
                            hint = (
                                "Check your narration API key and model access. "
                                "For OpenAI, verify OPENAI_API_KEY and that the model is enabled."
                            )
                        await _sse_write(
                            response,
                            "error",
                            {
                                "error": f"Narration request failed ({upstream.status})",
                                "detail": detail[:800],
                                "model": model_name,
                                "endpoint": endpoint,
                                "hint": hint,
                            },
                        )
                        await response.write_eof()
                        return response

                    ctype = (upstream.headers.get("Content-Type") or "").lower()
                    if "text/event-stream" in ctype:
                        pending = ""
                        async for chunk in upstream.content.iter_chunked(4096):
                            pending += chunk.decode("utf-8", errors="ignore").replace("\r\n", "\n")
                            boundary = pending.find("\n\n")
                            while boundary != -1:
                                block = pending[:boundary]
                                pending = pending[boundary + 2:]
                                payload_txt = None
                                for line in block.split("\n"):
                                    if line.startswith("data:"):
                                        payload_txt = line[5:].strip()
                                        break
                                if not payload_txt:
                                    boundary = pending.find("\n\n")
                                    continue
                                if payload_txt == "[DONE]":
                                    await _sse_write(response, "done", {"model": model_name, "endpoint": endpoint})
                                    await response.write_eof()
                                    return response
                                try:
                                    payload = json.loads(payload_txt)
                                except json.JSONDecodeError:
                                    boundary = pending.find("\n\n")
                                    continue
                                text = _extract_openai_delta_text(payload)
                                if text:
                                    await _sse_write(response, "delta", {"text": text, "model": model_name})
                                boundary = pending.find("\n\n")

                        if pending.strip():
                            try:
                                payload = json.loads(pending.strip().removeprefix("data:").strip())
                                text = _extract_openai_delta_text(payload)
                                if text:
                                    await _sse_write(response, "delta", {"text": text, "model": model_name})
                            except json.JSONDecodeError:
                                pass

                        await _sse_write(response, "done", {"model": model_name, "endpoint": endpoint})
                        await response.write_eof()
                        return response

                    # Fallback for non-streaming JSON responses.
                    payload = await upstream.json(content_type=None)
                    text = _extract_openai_message_text(payload)
                    if text:
                        await _sse_write(response, "delta", {"text": text, "model": model_name})
                    await _sse_write(response, "done", {"model": model_name, "endpoint": endpoint})
                    await response.write_eof()
                    return response
            except Exception as exc:
                if idx < len(models_to_try) - 1:
                    continue
                await _sse_write(response, "error", {"error": f"Narration upstream error: {exc}"})
                await response.write_eof()
                return response

    await _sse_write(response, "error", {"error": "No compatible narration model succeeded"})
    await response.write_eof()
    return response


async def mouse_stroke(request):
    try:
        data = await request.json()
        raw_pts = data["points"]
        color_idx = int(data.get("color_idx", 0))
        if not isinstance(raw_pts, list) or len(raw_pts) < 2:
            raise web.HTTPBadRequest(reason="Need at least 2 points")
        points = [(float(p["x"]), float(p["y"]), float(p.get("z", 0.0))) for p in raw_pts]
    except (KeyError, ValueError, TypeError) as exc:
        raise web.HTTPBadRequest(reason=str(exc))
    await asyncio.to_thread(tracker.add_mouse_stroke, points, color_idx)
    return web.Response(status=204)


async def cameras(request):
    found = find_cameras()
    return web.Response(content_type="application/json", text=json.dumps(found))


async def _render_whiteboard_response(request, fmt: str):
    try:
        yaw = float(request.query.get("yaw", "0"))
        fov = float(request.query.get("fov", "80"))
        width = int(request.query.get("w", "960"))
        height = int(request.query.get("h", "260"))
    except ValueError:
        raise web.HTTPBadRequest(reason="Invalid whiteboard query params")

    frame = await asyncio.to_thread(
        tracker.render_whiteboard,
        yaw_deg=yaw,
        fov_deg=fov,
        width=width,
        height=height,
    )
    if fmt == "jpg":
        ok, encoded = await asyncio.to_thread(
            cv2.imencode, ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        content_type = "image/jpeg"
    else:
        ok, encoded = await asyncio.to_thread(cv2.imencode, ".png", frame)
        content_type = "image/png"
    if not ok:
        raise web.HTTPInternalServerError(reason="Failed to encode whiteboard image")
    return web.Response(
        body=encoded.tobytes(),
        content_type=content_type,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


async def whiteboard(request):
    return await _render_whiteboard_response(request, "png")


async def whiteboard_jpg(request):
    return await _render_whiteboard_response(request, "jpg")


async def gallery(request):
    return web.Response(content_type="text/html", text=GALLERY_TEMPLATE)


async def replay_page(request):
    return web.Response(content_type="text/html", text=REPLAY_TEMPLATE)


async def api_sessions(request):
    if session_store is None:
        return web.Response(
            content_type="application/json",
            text=json.dumps({"error": "MongoDB not configured"}),
        )
    sessions = await asyncio.to_thread(session_store.list_sessions)
    return web.Response(content_type="application/json", text=json.dumps(sessions))


async def api_session_info(request):
    sid = request.match_info["session_id"]
    if session_store is None:
        raise web.HTTPServiceUnavailable(reason="MongoDB not configured")
    info = await asyncio.to_thread(session_store.get_session_info, sid)
    if info is None:
        raise web.HTTPNotFound()
    return web.Response(content_type="application/json", text=json.dumps(info))


async def api_session_frame(request):
    sid = request.match_info["session_id"]
    try:
        t_offset = float(request.query.get("t", "0"))
        yaw = float(request.query.get("yaw", "0"))
        fov = float(request.query.get("fov", "80"))
        width = int(request.query.get("w", "960"))
        height = int(request.query.get("h", "260"))
    except ValueError:
        raise web.HTTPBadRequest(reason="Invalid query params")
    if session_store is None:
        raise web.HTTPServiceUnavailable(reason="MongoDB not configured")
    ops = await asyncio.to_thread(session_store.build_ops_at, sid, t_offset)
    frame = await asyncio.to_thread(
        tracker.render_ops, ops, yaw_deg=yaw, fov_deg=fov, width=width, height=height
    )
    ok, encoded = await asyncio.to_thread(
        cv2.imencode, ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88]
    )
    if not ok:
        raise web.HTTPInternalServerError(reason="Encode failed")
    return web.Response(
        body=encoded.tobytes(),
        content_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


_POTATO_API = os.getenv("POTATO_API", "http://192.168.100.2:8080")
_current_theta = 0.0   # cumulative camera mount angle in degrees

async def api_motor_status(request):
    try:
        with urlopen(f"{_POTATO_API}/status", timeout=2) as resp:
            return web.Response(content_type="application/json", text=resp.read().decode())
    except URLError as exc:
        return web.Response(status=502, content_type="application/json",
                            text=json.dumps({"error": str(exc)}))


async def api_rotate(request):
    try:
        if request.method == "POST":
            body = await request.json()
            degrees = float(body["degrees"])
        else:
            degrees = float(request.query["degrees"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return web.Response(status=400, text='{"error":"degrees required"}', content_type="application/json")

    try:
        req = Request(
            f"{_POTATO_API}/rotate",
            data=json.dumps({"degrees": degrees}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
        global _current_theta
        _current_theta += degrees
        tracker.set_theta(_current_theta)
        result["theta"] = _current_theta
        return web.Response(content_type="application/json", text=json.dumps(result))
    except URLError as exc:
        return web.Response(status=502, content_type="application/json",
                            text=json.dumps({"error": f"potato unreachable: {exc}"}))


async def on_shutdown(app):
    tracker.stop()
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()


app = web.Application()
app.on_shutdown.append(on_shutdown)
app.router.add_get("/", index)
app.router.add_get("/student", student)
app.router.add_post("/offer", offer)
app.router.add_post("/clear", clear)
app.router.add_post("/undo", undo)
app.router.add_get("/cameras", cameras)
app.router.add_get("/state", state)
app.router.add_get("/stream", stream_state)
app.router.add_post("/color", set_color)
app.router.add_post("/live_view", set_live_view)
app.router.add_post("/api/replay/narrate_stream", replay_narrate_stream)
app.router.add_post("/stroke", mouse_stroke)
app.router.add_get("/whiteboard.png", whiteboard)
app.router.add_get("/whiteboard.jpg", whiteboard_jpg)
app.router.add_get("/gallery", gallery)
app.router.add_get("/replay/{session_id}", replay_page)
app.router.add_get("/api/sessions", api_sessions)
app.router.add_get("/api/sessions/{session_id}", api_session_info)
app.router.add_get("/api/sessions/{session_id}/frame", api_session_frame)
app.router.add_post("/api/rotate", api_rotate)
app.router.add_get("/api/rotate", api_rotate)
app.router.add_get("/api/motor_status", api_motor_status)
app.router.add_static("/static/", STATIC_DIR, show_index=False)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
