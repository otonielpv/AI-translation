"""FastAPI web server: /operator, /listen, /health, /status, /ws/audio."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from src.state import PipelineState, PipelineStatus, pipeline_state

if TYPE_CHECKING:
    from src.streaming.broadcaster import AudioBroadcaster

log = logging.getLogger(__name__)

_TEMPLATES = Path(__file__).parent / "templates"


def _read_template(name: str) -> str:
    return (_TEMPLATES / name).read_text(encoding="utf-8")


def create_app(
    state: PipelineState,
    broadcaster: "AudioBroadcaster",
    pipeline_control,
    qr_png_bytes: bytes,
    lan_url: str,
    lifespan: Optional[Callable] = None,
) -> FastAPI:
    app = FastAPI(title="AI-Translation", lifespan=lifespan)

    # ------------------------------------------------------------------ health
    @app.get("/health")
    async def health():
        return {"ok": True, "status": state.status.value}

    # ------------------------------------------------------------------ status
    @app.get("/status")
    async def status():
        return JSONResponse(state.to_dict())

    # ----------------------------------------------------------------- QR PNG
    @app.get("/qr.png")
    async def qr_png():
        return Response(content=qr_png_bytes, media_type="image/png")

    # --------------------------------------------------------------- operator
    @app.get("/operator", response_class=HTMLResponse)
    async def operator_page():
        html = _read_template("operator.html")
        html = html.replace("{{LAN_URL}}", lan_url)
        return HTMLResponse(html)

    @app.get("/operator/devices")
    async def op_devices():
        from src.audio.capture import list_devices
        return list_devices()

    @app.post("/operator/set-device")
    async def op_set_device(body: dict):
        idx = body.get("index")
        if idx is None:
            return {"ok": False, "error": "missing index"}
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, pipeline_control.set_device, int(idx))
        return {"ok": True}

    @app.post("/operator/start")
    async def op_start():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, pipeline_control.start)
        return {"ok": True}

    @app.post("/operator/pause")
    async def op_pause():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, pipeline_control.pause)
        return {"ok": True}

    @app.post("/operator/resume")
    async def op_resume():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, pipeline_control.resume)
        return {"ok": True}

    @app.post("/operator/stop")
    async def op_stop():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, pipeline_control.stop)
        return {"ok": True}

    # ----------------------------------------------------------------- listen
    @app.get("/listen", response_class=HTMLResponse)
    async def listen_page():
        html = _read_template("listener.html")
        return HTMLResponse(html)

    # ------------------------------------------------------ WebSocket audio
    @app.websocket("/ws/audio")
    async def ws_audio(ws: WebSocket):
        await ws.accept()
        client_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=broadcaster.max_queue)
        broadcaster.add_client(client_q)
        with state._lock:
            state.listeners += 1
        log.info("Listener connected. Total: %d", state.listeners)
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(client_q.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Send a keepalive ping so the TCP stack detects dead clients quickly
                    await ws.send_bytes(b"")
                    continue
                await ws.send_bytes(chunk)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as exc:
            log.debug("WS audio error: %s", exc)
        finally:
            broadcaster.remove_client(client_q)
            with state._lock:
                state.listeners = max(0, state.listeners - 1)
            log.info("Listener disconnected. Total: %d", state.listeners)

    # ------------------------------------------------- operator SSE status
    @app.get("/status/stream")
    async def status_stream():
        async def generate():
            try:
                while True:
                    data = state.to_dict()
                    yield f"data: {json.dumps(data)}\n\n"
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                pass

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app
