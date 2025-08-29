# main.py
import os
import json
import asyncio
import logging
import uuid
import random
import re
import time
import contextlib
from typing import Optional, Tuple, Dict, List

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import av
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from fractions import Fraction

# Pipecat core
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

# OpenAI Realtime (Beta)
from pipecat.services.openai_realtime_beta.openai import OpenAIRealtimeBetaLLMService
from pipecat.services.openai_realtime_beta.events import (
    SessionProperties,
    TurnDetection,
    ResponseCreateEvent,
    ResponseProperties,
)

# ------------------- Load env & bootstrap app -------------------
load_dotenv(override=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================== CONFIG =========================
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "12345")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2025-06-03")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful live voice agent.")
REALTIME_VOICE = os.getenv("REALTIME_VOICE", "alloy")

PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# If a human doesn’t pick up in time we auto-switch to the bot
AGENT_PICK_TIMEOUT_SEC = int(os.getenv("AGENT_PICK_TIMEOUT_SEC", "30"))

GREETING_TEXT = os.getenv(
    """ Hi this is sena bot. how can i help you today? speak in english and tamil answer about thing the user is asking"""
)

# Opus shaping for Meta WA side — we run MONO to WA
OPUS_PTIME_MS = 20
OPUS_MAX_AVG_BITRATE = 48000
OPUS_STEREO = False  # IMPORTANT: WhatsApp side is mono

# Control barge-in interruptions (kept conservative)
ALLOW_INTERRUPTION = os.getenv("ALLOW_INTERRUPTION", "0") not in ("0", "false", "False")
TURN_PREFIX_MS = int(os.getenv("TURN_PREFIX_MS", "120"))
TURN_SILENCE_MS = int(os.getenv("TURN_SILENCE_MS", "700"))

# ICE servers used by agent browser page
try:
    ICE_SERVERS = json.loads(os.getenv("ICE_SERVERS_JSON", '[{"urls":["stun:stun.l.google.com:19302"]}]'))
except Exception:
    ICE_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]
ICE_SERVERS_JS = json.dumps(ICE_SERVERS)

# ======================== LOGGING ========================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s - %(message)s"
)
log = logging.getLogger("wa-webrtc-pipecat")

# ==================== GRAPH CONSTANTS ====================
GRAPH = f"https://graph.facebook.com/v23.0/{PHONE_NUMBER_ID}"
GRAPH_CALLS = f"{GRAPH}/calls"
http_client = httpx.AsyncClient(timeout=30)

# ====================== SDP SHAPER =======================
def _extract_ice_and_fp(local_sdp: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    ufrag = None
    pwd = None
    fp = None
    rx = re.compile(r"^a=fingerprint:\s*([A-Za-z0-9\-]+)\s+([0-9A-Fa-f:]+)", re.MULTILINE)
    for line in local_sdp.splitlines():
        s = line.strip()
        if s.lower().startswith("a=ice-ufrag:"):
            ufrag = s.split(":", 1)[1].strip()
        elif s.lower().startswith("a=ice-pwd:"):
            pwd = s.split(":", 1)[1].strip()
    m = rx.search(local_sdp)
    if m:
        _, val = m.group(1).lower(), m.group(2).upper()
        fp = val
    return ufrag, pwd, fp

def _opus_fmtp() -> str:
    parts = [
        "minptime=10",
        "useinbandfec=1",
        f"stereo={'1' if OPUS_STEREO else '0'}",
        f"sprop-stereo={'1' if OPUS_STEREO else '0'}",
        "cbr=1",
        f"maxaveragebitrate={OPUS_MAX_AVG_BITRATE}",
        "maxplaybackrate=48000",
    ]
    return ";".join(parts)

def build_meta_style_sdp(local_sdp: str) -> Tuple[str, dict]:
    """
    Build an SDP answer WhatsApp accepts. We run mono -> opus/48000/1.
    """
    ufrag, pwd, fp_val = _extract_ice_and_fp(local_sdp)
    info = {"has_ufrag": bool(ufrag), "has_pwd": bool(pwd), "fp_val": fp_val}
    if not (ufrag and pwd and fp_val):
        return "", info

    stream_id = str(uuid.uuid4())
    track_id = str(uuid.uuid4())
    ssrc = random.randint(100_000_000, 4_000_000_000)
    channels = 2 if OPUS_STEREO else 1

    lines = [
        "v=0",
        "o=- 7669997803033704573 2 IN IP4 127.0.0.1",
        "s=-",
        "t=0 0",
        "a=group:BUNDLE 0",
        "a=extmap-allow-mixed",
        f"a=msid-semantic: WMS {stream_id}",
        "m=audio 9 UDP/TLS/RTP/SAVPF 111",
        "c=IN IP4 0.0.0.0",
        "a=rtcp:9 IN IP4 0.0.0.0",
        f"a=ice-ufrag:{ufrag}",
        f"a=ice-pwd:{pwd}",
        "a=ice-options:trickle",
        f"a=fingerprint:sha-256 {fp_val}",
        "a=setup:active",
        "a=mid:0",
        "a=extmap:1 urn:ietf:params:rtp-hdrext:ssrc-audio-level",
        "a=extmap:2 http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time",
        "a=extmap:3 http://www.ietf.org/id/draft-holmer-rmcat-transport-wide-cc-extensions-01",
        "a=extmap:4 urn:ietf:params:rtp-hdrext:sdes:mid",
        "a=sendrecv",
        f"a=msid:{stream_id} {track_id}",
        "a=rtcp-mux",
        "a=rtcp-rsize",
        f"a=rtpmap:111 opus/48000/{channels}",
        "a=rtcp-fb:111 transport-cc",
        f"a=fmtp:111 {_opus_fmtp()}",
        f"a=ptime:{OPUS_PTIME_MS}",
        f"a=maxptime:{OPUS_PTIME_MS}",
        f"a=ssrc:{ssrc} cname:webrtc",
        f"a=ssrc:{ssrc} msid:{stream_id} {track_id}",
    ]
    return "\r\n".join(lines) + "\r\n", info

def sample_meta_answer(channels: int) -> str:
    stereo_flag = "1" if channels == 2 else "0"
    return (
        "v=0\r\n"
        "o=- 7669997803033704573 2 IN IP4 127.0.0.1\r\n"
        "s=-\r\n"
        "t=0 0\r\n"
        "a=group:BUNDLE 0\r\n"
        "a=extmap-allow-mixed\r\n"
        "a=msid-semantic: WMS 3c28addc-03b7-4170-b5cd-535bfe767e75\r\n"
        "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
        "c=IN IP4 0.0.0.0\r\n"
        "a=rtcp:9 IN IP4 0.0.0.0\r\n"
        "a=ice-ufrag:6O0H\r\n"
        "a=ice-pwd:TYCbtfOrBMPpfxFRgSbYnuTI\r\n"
        "a=ice-options:trickle\r\n"
        "a=fingerprint:sha-256 9F:45:2C:A8:C3:C0:CC:9B:59:4F:D1:02:56:52:FA:36:00:BE:C0:79:87:B3:D9:9C:3E:BF:60:98:25:B4:26:FC\r\n"
        "a=setup:active\r\n"
        "a=mid:0\r\n"
        "a=extmap:1 urn:ietf:params:rtp-hdrext:ssrc-audio-level\r\n"
        "a=extmap:2 http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time\r\n"
        "a=extmap:3 http://www.ietf.org/id/draft-holmer-rmcat-transport-wide-cc-extensions-01\r\n"
        "a=extmap:4 urn:ietf:params:rtp-hdrext:sdes:mid\r\n"
        "a=sendrecv\r\n"
        "a=msid:3c28addc-03b7-4170-b5cd-535bfe767e75 38c455bc-3727-4129-b336-8cd2c6a68486\r\n"
        "a=rtcp-mux\r\n"
        "a=rtcp-rsize\r\n"
        f"a=rtpmap:111 opus/48000/{channels}\r\n"
        "a=rtcp-fb:111 transport-cc\r\n"
        f"a=fmtp:111 minptime=10;useinbandfec=1;stereo={stereo_flag};sprop-stereo={stereo_flag};cbr=1;maxaveragebitrate=48000;maxplaybackrate=48000\r\n"
        f"a=ptime:{OPUS_PTIME_MS}\r\n"
        f"a=maxptime:{OPUS_PTIME_MS}\r\n"
        "a=ssrc:2430753100 cname:MPddPt/R2ioP4vCm\r\n"
        "a=ssrc:2430753100 msid:3c28addc-03b7-4170-b5cd-535bfe767e75 38c455bc-3727-4129-b336-8cd2c6a68486\r\n"
    )

SAMPLE_META_ANSWER = sample_meta_answer(1 if not OPUS_STEREO else 2)

# =================== BRIDGE (Human Path) =================
class ResampleToStereo48k:
    def __init__(self):
        self.resampler = av.audio.resampler.AudioResampler(format="s16", layout="stereo", rate=48000)
    def convert_many(self, frame: av.AudioFrame) -> List[av.AudioFrame]:
        if frame is None:
            return []
        out = self.resampler.resample(frame) if (
            frame.format.name != "s16" or frame.sample_rate != 48000 or frame.layout.name != "stereo"
        ) else frame
        frames = out if isinstance(out, list) else [out]
        for f in frames:
            f.sample_rate = 48000
        return frames

class ResampleToMono48k:
    def __init__(self):
        self.resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=48000)
    def convert_many(self, frame: av.AudioFrame) -> List[av.AudioFrame]:
        if frame is None:
            return []
        out = self.resampler.resample(frame) if (
            frame.format.name != "s16" or frame.sample_rate != 48000 or frame.layout.name != "mono"
        ) else frame
        frames = out if isinstance(out, list) else [out]
        for f in frames:
            f.sample_rate = 48000
        return frames

class BridgeQueueTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self, q: asyncio.Queue, sample_rate: int = 48000, layout: str = "stereo"):
        super().__init__()
        self.q = q
        self.sr = sample_rate
        self.layout = layout
        self._tb = Fraction(1, self.sr)
        self._ts = 0
        self._channels = 2 if self.layout == "stereo" else 1

    def _silence(self, samples: int = 960) -> av.AudioFrame:
        f = av.AudioFrame(format="s16", layout=self.layout, samples=samples)
        f.sample_rate = self.sr
        for p in f.planes:
            p.update(b"\x00" * p.buffer_size)
        f.pts = self._ts
        f.time_base = self._tb
        self._ts += samples
        return f

    async def recv(self) -> av.AudioFrame:
        try:
            frame = await asyncio.wait_for(self.q.get(), timeout=0.2)
            samples = int(getattr(frame, "samples", 0) or 0)
            if samples <= 0:
                bytes_per_sample = 2 * self._channels
                samples = frame.planes[0].buffer_size // bytes_per_sample
            frame.pts = self._ts
            frame.time_base = self._tb
            self._ts += int(samples)
            return frame
        except asyncio.TimeoutError:
            return self._silence()

class CallBridge:
    def __init__(self):
        self.wa_to_browser: asyncio.Queue = asyncio.Queue(maxsize=64)
        self.browser_to_wa: asyncio.Queue = asyncio.Queue(maxsize=64)
        self.to_browser_resample = ResampleToStereo48k()
        self.to_wa_resample = ResampleToMono48k()

    async def push_from_wa(self, frame: av.AudioFrame):
        for f in self.to_browser_resample.convert_many(frame):
            if self.wa_to_browser.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self.wa_to_browser.get_nowait()
            await self.wa_to_browser.put(f)

    async def push_from_browser(self, frame: av.AudioFrame):
        for f in self.to_wa_resample.convert_many(frame):
            if self.browser_to_wa.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self.browser_to_wa.get_nowait()
            await self.browser_to_wa.put(f)

class CallCtx:
    def __init__(self, call_id: str, offer_sdp: str):
        self.call_id = call_id
        self.offer_sdp = offer_sdp
        self.created = time.time()
        self.accepted = False
        self.mode = "waiting"      # waiting | human | bot
        self.bridge = CallBridge()
        self.webrtc_wa: Optional[SmallWebRTCConnection] = None
        # Browser leg
        self.browser_pc: Optional[RTCPeerConnection] = None   # using aiortc for browser leg
        self.browser_joined: bool = False
        # Bot pipeline
        self.bot_runner: Optional[PipelineRunner] = None
        self.bot_task: Optional[asyncio.Task] = None
        self.bot_transport: Optional[SmallWebRTCTransport] = None
        self.auto_timer_task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()

# >>>>>>>>>> FIX: define ACTIVE_CALLS <<<<<<<<<<
ACTIVE_CALLS: Dict[str, CallCtx] = {}

# ======================== PAGES ==========================
def agent_html(timeout_s: int) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Call Console</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:20px;background:#0b1220;color:#e6edf3}}
    h1{{margin:0 0 12px 0}}
    .muted{{opacity:.7;font-size:.9rem}}
    table{{border-collapse:collapse;width:100%;margin-top:12px;background:#111a2b;border-radius:10px;overflow:hidden}}
    th,td{{padding:10px 12px;border-bottom:1px solid #1e2a44}}
    th{{text-align:left;background:#0d182b}}
    tr:hover td{{background:#0f1b30}}
    button{{background:#2563eb;color:white;border:none;padding:8px 12px;border-radius:8px;cursor:pointer}}
    button.alt{{background:#0ea5e9}}
    button.danger{{background:#ef4444}}
    .row{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
    .pill{{padding:2px 8px;border-radius:999px;background:#1f2a44;font-size:.8rem}}
    .ok{{background:#1f4d2b}}
    a{{color:#7dd3fc}}
  </style>
</head>
<body>
  <h1>Live Calls</h1>
  <div class="muted">Auto-assign to bot after {timeout_s}s if not accepted.</div>

  <div id="list"></div>

  <script>
    async function api(method, path, body) {{
      const r = await fetch(path, {{
        method, headers:{{"Content-Type":"application/json"}},
        body: body ? JSON.stringify(body) : undefined
      }});
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    }}

    async function load() {{
      try {{
        const calls = await api("GET", "/api/active_calls");
        const rows = calls.map(c => `
          <tr>
            <td><code>${{c.call_id}}</code></td>
            <td><span class="pill">${{c.status}}</span></td>
            <td>${{c.age_s}}s</td>
            <td class="row">
              <button onclick="acceptCall('${{c.call_id}}')">Accept (Human)</button>
              <button class="danger" onclick="hangup('${{c.call_id}}')">Hang up</button>
              <a href="/call.html?call_id=${{c.call_id}}" target="_blank">Open call tab</a>
            </td>
          </tr>
        `).join("");
        document.getElementById("list").innerHTML = `
          <table>
            <thead><tr><th>Call ID</th><th>Status</th><th>Age</th><th>Actions</th></tr></thead>
            <tbody>${{rows || ""}}</tbody>
          </table>
        `;
      }} catch (e) {{
        document.getElementById("list").innerHTML = `<div>Load failed: ${{e}}</div>`;
      }}
    }}

    async function acceptCall(id) {{
      try {{
        await api("POST", "/api/accept", {{call_id:id}});
        window.open("/call.html?call_id=" + id, "_blank");
        load();
      }} catch (e) {{ alert(e); }}
    }}

    async function hangup(id) {{
      try {{
        await api("POST", "/api/decline", {{call_id:id}});
        load();
      }} catch (e) {{ alert(e); }}
    }}

    load();
    setInterval(load, 1000);
  </script>
</body>
</html>"""

def call_html() -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Call Join</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:20px;background:#0b1220;color:#e6edf3}}
    .card{{background:#0f172a;border:1px solid #1e2a44;padding:16px;border-radius:12px;max-width:680px}}
    .muted{{opacity:.75}}
    button{{background:#22c55e;color:#07210f;border:none;padding:10px 14px;border-radius:8px;cursor:pointer}}
    .row{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px}}
    code{{background:#0b1220;padding:2px 6px;border-radius:6px;border:1px solid #1e2a44}}
  </style>
</head>
<body>
  <div class="card">
    <h2>Browser ↔ WhatsApp Bridge</h2>
    <div id="status" class="muted">Preparing…</div>
    <div class="row">
      <button id="btn">Start / Reconnect</button>
      <span>Call ID: <code id="cid"></code></span>
    </div>
    <p class="muted">Keep this tab open while you’re on the call.</p>
  </div>

  <audio id="remote" autoplay></audio>

  <script>
    const ICE_SERVERS = {ICE_SERVERS_JS};

    function getParam(name) {{
      return new URLSearchParams(location.search).get(name);
    }}

    const callId = getParam("call_id");
    document.getElementById("cid").textContent = callId || "(missing)";
    const statusEl = document.getElementById("status");
    const btn = document.getElementById("btn");
    const remoteEl = document.getElementById("remote");

    let pc, localStream;

    async function start() {{
      if (!callId) {{
        statusEl.textContent = "Missing call_id in URL.";
        return;
      }}
      statusEl.textContent = "Starting…";

      if (pc) try {{ pc.close(); }} catch (e) {{}}
      pc = new RTCPeerConnection({{ iceServers: ICE_SERVERS }});

      // play incoming audio
      pc.ontrack = (ev) => {{
        const stream = ev.streams && ev.streams[0] ? ev.streams[0] : new MediaStream([ev.track]);
        remoteEl.srcObject = stream;
      }};

      // mic
      try {{
        localStream = await navigator.mediaDevices.getUserMedia({{ audio: {{
          echoCancellation: true, noiseSuppression: true, autoGainControl: true
        }}}});
      }} catch (e) {{
        statusEl.textContent = "Mic permission denied: " + e;
        return;
      }}
      localStream.getAudioTracks().forEach(t => pc.addTrack(t, localStream));

      // gather full offer (no trickle)
      const offer = await pc.createOffer({{ offerToReceiveAudio: true }});
      await pc.setLocalDescription(offer);

      await new Promise((res) => {{
        if (pc.iceGatheringState === "complete") return res();
        const ch = () => {{
          if (pc.iceGatheringState === "complete") {{
            pc.removeEventListener("icegatheringstatechange", ch);
            res();
          }}
        }};
        pc.addEventListener("icegatheringstatechange", ch);
        setTimeout(res, 1500); // safety
      }});

      const payload = {{
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type
      }};

      const r = await fetch("/webrtc/offer/" + callId, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload)
      }});
      if (!r.ok) {{
        statusEl.textContent = "Offer failed: " + (await r.text());
        return;
      }}
      const ans = await r.json();
      await pc.setRemoteDescription(ans);
      statusEl.textContent = "Connected.";
    }}

    btn.addEventListener("click", start);
    // auto start
    start();
  </script>
</body>
</html>"""

# ======================= APIs ============================
@app.get("/agent")
async def agent_page():
    return HTMLResponse(agent_html(AGENT_PICK_TIMEOUT_SEC))

@app.get("/call.html")
async def call_page():
    return HTMLResponse(call_html())

@app.get("/api/active_calls")
async def list_active_calls():
    out = []
    now = time.time()
    for cid, ctx in list(ACTIVE_CALLS.items()):
        status = ctx.mode if ctx.mode in ("waiting", "human", "bot") else "unknown"
        out.append({"call_id": cid, "status": status, "age_s": int(now - ctx.created)})
    return JSONResponse(out)

@app.post("/api/accept")
async def api_accept(request: Request):
    body = await request.json()
    call_id = body.get("call_id")
    ctx = ACTIVE_CALLS.get(call_id)
    if not ctx:
        return JSONResponse({"ok": False, "error": "invalid_call_id"}, status_code=404)
    if ctx.auto_timer_task and not ctx.auto_timer_task.done():
        ctx.auto_timer_task.cancel()
    await accept_call_human(ctx)
    return JSONResponse({"ok": True, "status": "accepted"})

@app.post("/api/decline")
async def api_decline(request: Request):
    body = await request.json()
    call_id = body.get("call_id")
    if call_id not in ACTIVE_CALLS:
        return JSONResponse({"ok": False, "error": "invalid_call_id"}, status_code=404)
    await post_graph(GRAPH_CALLS, {"messaging_product": "whatsapp", "call_id": call_id, "action": "terminate"})
    await end_call(call_id)
    return JSONResponse({"ok": True, "status": "terminated"})

@app.post("/webrtc/offer/{call_id}")
async def webrtc_offer(call_id: str, request: Request):
    """
    BROWSER LEG — use raw aiortc so we attach tracks *before* creating the answer.
    Fixes the initial 20–30s mute from browser→phone.
    """
    ctx = ACTIVE_CALLS.get(call_id)
    if not ctx:
        return JSONResponse({"error": "invalid_call_id"}, status_code=404)
    if ctx.mode != "human":
        return JSONResponse({"error": "call not in human mode"}, status_code=409)
    if not ctx.webrtc_wa:
        return JSONResponse({"error": "wa leg not ready"}, status_code=409)

    data = await request.json()
    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

    pc = RTCPeerConnection()

    # WhatsApp -> Browser: feed WA audio to the browser (stereo@48k for browser)
    out_track = BridgeQueueTrack(ctx.bridge.wa_to_browser, sample_rate=48000, layout="stereo")
    pc.addTrack(out_track)

    # Browser -> WhatsApp: receive mic, push into WA queue
    @pc.on("track")
    async def on_browser_track(track):
        if track.kind != "audio":
            return
        log.info(f"[{call_id}] Browser audio track received")
        last_log = time.monotonic()
        pushed = 0
        try:
            while True:
                frame = await track.recv()
                samples = getattr(frame, "samples", 0) or 0
                pushed += samples
                await ctx.bridge.push_from_browser(frame)
                now = time.monotonic()
                if now - last_log >= 2.0:
                    secs = pushed / 48000.0
                    log.info(f"[{call_id}] Browser→WA audio {secs:.2f}s queued in last 2s")
                    pushed = 0
                    last_log = now
        except Exception:
            pass

    # NEGOTIATE *after* tracks exist
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    ctx.browser_pc = pc
    ctx.browser_joined = True
    log.info(f"[{call_id}] Browser joined (aiortc; sender/receiver negotiated in the first answer).")

    return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

# ===================== HTTP HELPER =======================
async def post_graph(url: str, payload: dict, retries: int = 2) -> httpx.Response:
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    backoffs = [0.25, 0.5, 1.0]
    last_resp: Optional[httpx.Response] = None
    for i in range(retries + 1):
        try:
            resp = await http_client.post(url, headers=headers, json=payload)
            last_resp = resp
            try:
                body = resp.json()
            except Exception:
                body = {"text": resp.text}
            log.info(f"Graph POST {url} -> {resp.status_code} {body}")
            if resp.status_code == 401:
                log.error("401 Unauthorized: Check WHATSAPP_ACCESS_TOKEN and PHONE_NUMBER_ID.")
            if resp.status_code != 500:
                return resp
        except httpx.HTTPError as e:
            log.error(f"Graph POST {url} failed: {e!r}")
        if i < retries:
            await asyncio.sleep(backoffs[min(i, len(backoffs)-1)])
    return last_resp or httpx.Response(500, request=httpx.Request("POST", url))

# ===================== CALL FLOW =========================
async def handle_connect_event(call: dict):
    call_id = call.get("id")
    offer_sdp = (call.get("session") or {}).get("sdp", "")
    if not offer_sdp:
        log.error(f"[{call_id}] Missing offer SDP; cannot proceed.")
        return
    log.info(f"[{call_id}] Incoming call; storing offer (len={len(offer_sdp)}). Waiting for Accept.")

    ctx = CallCtx(call_id, offer_sdp)
    ACTIVE_CALLS[call_id] = ctx

    async def auto_accept():
        try:
            await asyncio.sleep(AGENT_PICK_TIMEOUT_SEC)
            if call_id in ACTIVE_CALLS and ctx.mode == "waiting":
                log.info(f"[{call_id}] Auto-accept after {AGENT_PICK_TIMEOUT_SEC}s.")
                await accept_call_bot(ctx)
        except asyncio.CancelledError:
            pass

    ctx.auto_timer_task = asyncio.create_task(auto_accept())

async def accept_call_human(ctx: CallCtx):
    async with ctx.lock:
        if ctx.accepted:
            return

        wa_conn = SmallWebRTCConnection()
        await wa_conn.initialize(ctx.offer_sdp, "offer")
        wa_conn.force_transceivers_to_send_recv()

        # Add Browser→WA sender BEFORE we answer WA
        wa_out = BridgeQueueTrack(ctx.bridge.browser_to_wa, sample_rate=48000, layout="mono")
        wa_conn.pc.addTrack(wa_out)
        log.info(f"[{ctx.call_id}] Added Browser→WA outbound (mono@48k) before SDP answer.")

        async def _wa_incoming_loop(track):
            if track.kind != "audio":
                return
            log.info(f"[{ctx.call_id}] WhatsApp audio track received (bridging to browser)")
            last_log = time.monotonic()
            pushed = 0
            try:
                while True:
                    frame = await track.recv()
                    samples = getattr(frame, "samples", 0) or 0
                    pushed += samples
                    await ctx.bridge.push_from_wa(frame)
                    now = time.monotonic()
                    if now - last_log >= 2.0:
                        secs = pushed / 48000.0
                        log.info(f"[{ctx.call_id}] WA→Browser audio {secs:.2f}s queued in last 2s")
                        pushed = 0
                        last_log = now
            except Exception:
                pass

        @wa_conn.pc.on("track")
        async def on_wa_track(track):
            asyncio.create_task(_wa_incoming_loop(track))

        for r in wa_conn.pc.getReceivers():
            t = getattr(r, "track", None)
            if t and t.kind == "audio":
                asyncio.create_task(_wa_incoming_loop(t))

        local_sdp = wa_conn.pc.localDescription.sdp or ""
        shaped_sdp, _ = build_meta_style_sdp(local_sdp)
        if not shaped_sdp:
            shaped_sdp = SAMPLE_META_ANSWER

        with contextlib.suppress(Exception):
            await post_graph(
                GRAPH_CALLS,
                {
                    "messaging_product": "whatsapp",
                    "call_id": ctx.call_id,
                    "action": "pre_accept",
                    "session": {"sdp_type": "answer", "sdp": shaped_sdp},
                },
                retries=2,
            )

        resp = await post_graph(
            GRAPH_CALLS,
            {
                "messaging_product": "whatsapp",
                "call_id": ctx.call_id,
                "action": "accept",
                "session": {"sdp_type": "answer", "sdp": shaped_sdp},
            },
            retries=2,
        )
        if resp.status_code != 200:
            resp2 = await post_graph(
                GRAPH_CALLS,
                {
                    "messaging_product": "whatsapp",
                    "call_id": ctx.call_id,
                    "action": "accept",
                    "session": {"sdp_type": "answer", "sdp": SAMPLE_META_ANSWER},
                },
                retries=2,
            )
            if resp2.status_code != 200:
                await post_graph(GRAPH_CALLS, {"messaging_product":"whatsapp","call_id":ctx.call_id,"action":"terminate"})
                await end_call(ctx.call_id)
                return

        ctx.webrtc_wa = wa_conn
        ctx.accepted = True
        ctx.mode = "human"
        log.info(f"[{ctx.call_id}] Call accepted for HUMAN bridge (two-way).")

def _make_llm(model: str) -> OpenAIRealtimeBetaLLMService:
    return OpenAIRealtimeBetaLLMService(
        api_key=OPENAI_API_KEY,
        model=model,
        session_properties=SessionProperties(
            modalities=["audio", "text"],
            instructions=SYSTEM_PROMPT,
            voice=REALTIME_VOICE,
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            turn_detection=TurnDetection(
                threshold=0.5,
                prefix_padding_ms=TURN_PREFIX_MS,
                silence_duration_ms=TURN_SILENCE_MS,
            ),
            temperature=0.7,
            max_response_output_tokens=200,
        ),
        send_transcription_frames=False,
    )

async def _start_llm_pipeline(ctx: CallCtx, model: str, send_greeting: bool):
    assert ctx.webrtc_wa is not None, "webrtc must exist before starting LLM pipeline"

    transport = SmallWebRTCTransport(
        webrtc_connection=ctx.webrtc_wa,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,   # Silero VAD requirement
            audio_in_channels=1,
            audio_out_sample_rate=48000,  # WA side playback
            audio_out_channels=1,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=2,      # keeps OpenAI happy on chunk sizing
        ),
    )
    ctx.bot_transport = transport

    llm = _make_llm(model)

    @transport.event_handler("on_client_connected")
    async def on_connected(_t, _c):
        log.info(f"[{ctx.call_id}] Bot connected to WA peer.")
        if send_greeting:
            await asyncio.sleep(1.0)
            try:
                await llm.send_client_event(
                    ResponseCreateEvent(
                        response=ResponseProperties(
                            instructions=GREETING_TEXT,
                            output_audio_format="pcm16"
                        )
                    )
                )
                log.info(f"[{ctx.call_id}] Sent greeting.")
            except Exception as e:
                log.error(f"[{ctx.call_id}] Greeting failed: {e!r}")

    pipeline = Pipeline([transport.input(), llm, transport.output()])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=ALLOW_INTERRUPTION,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=48000,
        ),
    )
    runner = PipelineRunner(handle_sigint=False, handle_sigterm=False)

    async def run_and_log():
        try:
            log.info(f"[{ctx.call_id}] Starting LLM pipeline (model={model}, interruptions={ALLOW_INTERRUPTION}).")
            await runner.run(task)
        except Exception as e:
            log.error(f"[{ctx.call_id}] Pipeline crashed: {e!r}")
        finally:
            log.debug(f"[{ctx.call_id}] Pipeline finished.")

    ctx.bot_runner = runner
    ctx.bot_task = asyncio.create_task(run_and_log())

async def accept_call_bot(ctx: CallCtx):
    async with ctx.lock:
        if ctx.accepted:
            return
        if not OPENAI_API_KEY:
            log.error("OPENAI_API_KEY missing; cannot start bot.")
            return

        webrtc = SmallWebRTCConnection()
        await webrtc.initialize(ctx.offer_sdp, "offer")
        webrtc.force_transceivers_to_send_recv()

        priming_q = asyncio.Queue(maxsize=1)
        wa_out = BridgeQueueTrack(priming_q, sample_rate=48000, layout="mono")
        webrtc.pc.addTrack(wa_out)

        local_sdp = webrtc.pc.localDescription.sdp or ""
        shaped_sdp, _ = build_meta_style_sdp(local_sdp)
        if not shaped_sdp:
            shaped_sdp = SAMPLE_META_ANSWER

        with contextlib.suppress(Exception):
            await post_graph(
                GRAPH_CALLS,
                {
                    "messaging_product": "whatsapp",
                    "call_id": ctx.call_id,
                    "action": "pre_accept",
                    "session": {"sdp_type": "answer", "sdp": shaped_sdp},
                },
                retries=2,
            )

        resp = await post_graph(
            GRAPH_CALLS,
            {
                "messaging_product": "whatsapp",
                "call_id": ctx.call_id,
                "action": "accept",
                "session": {"sdp_type": "answer", "sdp": shaped_sdp},
            },
            retries=2,
        )
        if resp.status_code != 200:
            resp2 = await post_graph(
                GRAPH_CALLS,
                {
                    "messaging_product": "whatsapp",
                    "call_id": ctx.call_id,
                    "action": "accept",
                    "session": {"sdp_type": "answer", "sdp": SAMPLE_META_ANSWER},
                },
                retries=2,
            )
            if resp2.status_code != 200:
                await post_graph(GRAPH_CALLS, {"messaging_product":"whatsapp","call_id":ctx.call_id,"action":"terminate"})
                await end_call(ctx.call_id)
                return

        ctx.webrtc_wa = webrtc
        ctx.accepted = True
        ctx.mode = "bot"
        log.info(f"[{ctx.call_id}] Call accepted for BOT (OpenAI Realtime).")

        await _start_llm_pipeline(ctx, OPENAI_REALTIME_MODEL, send_greeting=True)

async def end_call(call_id: Optional[str]):
    if not call_id: return
    ctx = ACTIVE_CALLS.pop(call_id, None)
    if not ctx: return
    try:
        if ctx.auto_timer_task and not ctx.auto_timer_task.done():
            ctx.auto_timer_task.cancel()
    except Exception:
        pass
    try:
        if ctx.bot_task and not ctx.bot_task.done():
            ctx.bot_task.cancel()
            with contextlib.suppress(Exception):
                await ctx.bot_task
    except Exception:
        pass
    try:
        if ctx.webrtc_wa:
            await ctx.webrtc_wa.disconnect()
    except Exception:
        pass
    try:
        if ctx.browser_pc:
            await ctx.browser_pc.close()
    except Exception:
        pass
    log.info(f"[{call_id}] Cleaned up.")

# ====================== WEBHOOKS =========================
@app.get("/webhook")
async def verify(request: Request):
    q = dict(request.query_params)
    if q.get("hub.mode") == "subscribe" and q.get("hub.verify_token") == VERIFY_TOKEN and q.get("hub.challenge"):
        return PlainTextResponse(q["hub.challenge"])
    return PlainTextResponse("forbidden", status_code=403)

@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    try:
        entry = body["entry"][0]["changes"][0]["value"]
        if entry.get("messaging_product") == "whatsapp" and "calls" in entry:
            for c in entry["calls"]:
                ev = c.get("event")
                if ev == "connect":
                    asyncio.create_task(handle_connect_event(c))
                elif ev in ("terminate", "terminated", "hangup"):
                    call_id = c.get("id")
                    log.info(f"call event: {ev} id={call_id}")
                    await end_call(call_id)
    except Exception:
        log.exception("Webhook error")
    return PlainTextResponse("OK")

# ========================= RUN ===========================
if __name__ == "__main__":
    if not PHONE_NUMBER_ID or not PHONE_NUMBER_ID.isdigit():
        log.error("PHONE_NUMBER_ID must be the numeric Phone Number ID (not the raw phone number).")
        raise SystemExit(1)
    if not (WHATSAPP_ACCESS_TOKEN and len(WHATSAPP_ACCESS_TOKEN) > 50):
        log.error("WHATSAPP_ACCESS_TOKEN looks invalid/too short.")
        raise SystemExit(1)
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY missing.")
        raise SystemExit(1)

    # Tested with: pipecat-ai[openai,webrtc]==0.0.80, aiortc, av, fastapi, uvicorn, httpx, numpy, python-dotenv
    uvicorn.run(app, host="0.0.0.0", port=PORT)
