"""LLM Router v3 - 메타 라우팅 + 음성 파이프라인 + 실시간 처리 모니터"""
import os
import json
import time
import subprocess
import tempfile
import threading
import itertools
import requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file
from router import LLMRouter
from cache import SemanticCache

app = Flask(__name__)
_BASE = os.path.dirname(os.path.abspath(__file__))
router = LLMRouter(config_path=os.path.join(_BASE, "config.yaml"))
cache = SemanticCache(max_size=2000, ttl=3600, threshold=0.85)
router.start_keepalive()

request_log = []
MAX_LOG = 500

# ============================================================
# 실시간 처리 추적 (진행 중인 요청)
# ============================================================
_active = {}
_active_lock = threading.Lock()
_active_seq = itertools.count(1)


def _new_active(req_model):
    with _active_lock:
        rid = next(_active_seq)
        _active[rid] = {
            "id": rid,
            "model": req_model or "auto",
            "task_type": "-",
            "stage": "요청 수신",
            "started": time.time(),
        }
        return rid


def _update_active(rid, **kw):
    with _active_lock:
        if rid in _active:
            _active[rid].update(kw)


def _finish_active(rid):
    with _active_lock:
        return _active.pop(rid, None)


def _active_list():
    now = time.time()
    with _active_lock:
        return [
            {**v, "elapsed": round(now - v["started"], 1)}
            for v in _active.values()
        ]

# ============================================================
# 웹 대시보드
# ============================================================
@app.route("/")
def index():
    return render_template("dashboard_v2.html")


# ============================================================
# OpenAI 호환 엔드포인트 (v2 메타 라우팅)
# ============================================================
@app.route("/v1/models")
def list_models():
    models = []
    vram = router.vram_status()
    for name, cfg in router.config.get("models", {}).items():
        loaded = name in vram.get("loaded", {})
        models.append({
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": cfg.get("provider", "ollama"),
            "capabilities": cfg.get("capabilities", []),
            "speed": cfg.get("speed", "medium"),
            "vram_gb": cfg.get("vram_gb", 0),
            "loaded": loaded,
            "keep_warm": cfg.get("keep_warm", False),
        })
    return jsonify({"object": "list", "data": models})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request"}), 400

    messages = data.get("messages", [])
    requested_model = data.get("model", None)
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 4096)

    cache_key_model = requested_model or "auto"

    cache_cfg = router.config.get("cache", {})
    if cache_cfg.get("enabled", True):
        cached = cache.get(messages, cache_key_model)
        if cached:
            _log_request(cache_key_model, cached.get("task_type", "?"), cached=True)
            return _build_response(cached, cache_key_model, cached=True)

    rid = _new_active(requested_model)
    t0 = time.time()
    _update_active(rid, stage="추론 중")
    try:
        result = router.route(messages, requested_model, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        _finish_active(rid)
        return jsonify({"error": str(e)}), 500

    if "error" in result and not result.get("content"):
        _finish_active(rid)
        return jsonify({"error": result["error"]}), 503

    if cache_cfg.get("enabled", True) and result.get("content"):
        cache.set(messages, cache_key_model, result)

    elapsed_ms = int((time.time() - t0) * 1000)
    usage = result.get("usage", {}) or {}
    _finish_active(rid)
    _log_request(result["model"], result.get("task_type", "?"), error="error" in result,
                 elapsed_ms=elapsed_ms, tokens=usage.get("completion_tokens", 0))

    return _build_response(result, result["model"])


# ============================================================
# 음성 파이프라인
# ============================================================
@app.route("/v1/audio/transcriptions", methods=["POST"])
def audio_transcriptions():
    """STT: 음성 파일 → 텍스트 (faster-whisper)"""
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["python3", "-c", f"""
import sys
sys.path.insert(0, '/home/hoony/.hermes/hermes-agent/venv/lib/python3.11/site-packages')
from faster_whisper import WhisperModel
model = WhisperModel('large-v3', device='cuda', compute_type='float16')
segments, info = model.transcribe('{tmp_path}', beam_size=5, language=None)
text = ' '.join(s.text for s in segments)
print(text)
"""],
            capture_output=True, text=True, timeout=120,
        )
        text = result.stdout.strip()
        if not text:
            text = result.stderr.strip()[:200] or "(empty transcription)"
    except Exception as e:
        text = f"(transcription error: {e})"
    finally:
        os.unlink(tmp_path)

    return jsonify({"text": text, "model": "faster-whisper-large-v3"})


@app.route("/v1/audio/speech", methods=["POST"])
def audio_speech():
    """TTS: 텍스트 → 음성 파일 (edge-tts)"""
    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "No input text"}), 400

    input_text = data["input"][:1000]
    voice = data.get("voice", "ko-KR-SunHiNeural")
    speed = data.get("speed", 1.0)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        output_path = tmp.name

    try:
        subprocess.run(
            ["edge-tts", "--voice", voice, "--text", input_text,
             "--write-media", output_path, "--rate", f"{'+' if speed > 1.0 else ''}{int((speed-1)*100)}%"],
            capture_output=True, text=True, timeout=60,
        )
        return send_file(output_path, mimetype="audio/mpeg",
                         as_attachment=True, download_name="response.mp3")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/v1/voice/chat", methods=["POST"])
def voice_chat():
    """음성 전체 파이프라인: STT → 라우터 판단 → TTS"""
    data = request.get_json()
    if not data or "audio" not in data:
        return jsonify({"error": "No audio data (base64)"}), 400

    import base64
    audio_b64 = data["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(base64.b64decode(audio_b64))
        audio_path = tmp.name

    try:
        stt_result = subprocess.run(
            ["python3", "-c", f"""
import sys; sys.path.insert(0, '/home/hoony/.hermes/hermes-agent/venv/lib/python3.11/site-packages')
from faster_whisper import WhisperModel
model = WhisperModel('large-v3', device='cuda', compute_type='float16')
segments, info = model.transcribe('{audio_path}', beam_size=5, language='ko')
print(' '.join(s.text for s in segments))
"""],
            capture_output=True, text=True, timeout=120,
        )
        user_text = stt_result.stdout.strip()
        if not user_text:
            user_text = data.get("fallback_text", "")
    except Exception as e:
        user_text = f"(stt error)"
    finally:
        os.unlink(audio_path)

    llm_result = router.route([{"role": "user", "content": user_text}])

    tts_text = llm_result.get("content", "응답 생성 실패")[:500]
    audio_output = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    audio_output.close()

    try:
        subprocess.run(
            ["edge-tts", "--voice", "ko-KR-SunHiNeural", "--text", tts_text,
             "--write-media", audio_output.name],
            capture_output=True, text=True, timeout=60,
        )
        with open(audio_output.name, "rb") as f:
            audio_b64_response = base64.b64encode(f.read()).decode()
    except Exception as e:
        audio_b64_response = ""
    finally:
        os.unlink(audio_output.name)

    return jsonify({
        "stt_text": user_text,
        "llm_response": tts_text,
        "audio": audio_b64_response,
        "model": llm_result.get("model", ""),
        "task_type": llm_result.get("task_type", ""),
        "meta": llm_result.get("meta"),
    })


# ============================================================
# 원격 파일 복사 (다른 서버에서 scp/rsync로 가져오기)
# ============================================================
@app.route("/v1/files/fetch", methods=["POST"])
def files_fetch():
    """다른 서버에서 파일을 복사해온다.
    
    요청:
    {
        "source": "user@host:/path/to/file.xlsx",
        "dest": "/home/hoony/downloads/",
        "method": "scp"  # or "rsync"
    }
    """
    data = request.get_json()
    if not data or "source" not in data:
        return jsonify({"error": "source required (user@host:/path)"}), 400
    
    source = data["source"]
    dest = data.get("dest", "/home/hoony/downloads/")
    method = data.get("method", "scp")
    timeout = data.get("timeout", 300)
    
    # dest 디렉터리 생성
    os.makedirs(dest, exist_ok=True)
    
    # 보안: source에 shell 메타문자 차단
    import shlex
    if any(c in source for c in [';', '|', '&', '$', '`', '(', ')']):
        return jsonify({"error": "Invalid characters in source"}), 400
    
    if method == "rsync":
        cmd = ["rsync", "-avz", "--progress", source, dest]
    else:
        cmd = ["scp", "-o", "StrictHostKeyChecking=no", source, dest]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            # 복사된 파일 목록
            files = os.listdir(dest)
            return jsonify({
                "status": "success",
                "method": method,
                "source": source,
                "dest": dest,
                "files": files,
                "output": result.stdout[-500:] if result.stdout else "",
            })
        else:
            return jsonify({
                "status": "error",
                "method": method,
                "source": source,
                "error": result.stderr[-500:] if result.stderr else "unknown error",
            }), 500
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "error": f"Timeout after {timeout}s"}), 504
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/v1/files/list", methods=["GET"])
def files_list():
    """복사된 파일 목록"""
    dest = request.args.get("path", "/home/hoony/downloads/")
    if not os.path.isdir(dest):
        return jsonify({"error": f"Directory not found: {dest}"}), 404
    files = []
    for f in os.listdir(dest):
        fp = os.path.join(dest, f)
        if os.path.isfile(fp):
            st = os.stat(fp)
            files.append({
                "name": f,
                "size": st.st_size,
                "size_mb": round(st.st_size / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
            })
    files.sort(key=lambda x: x["modified"], reverse=True)
    return jsonify({"path": dest, "count": len(files), "files": files})


# ============================================================
# 모니터링 엔드포인트
# ============================================================
@app.route("/api/status")
def api_status():
    health = router.check_models_health()
    vram = router.vram_status()
    models_cfg = router.config.get("models", {})
    for name, info in vram.get("loaded", {}).items():
        info["vram_gb"] = models_cfg.get(name, {}).get("vram_gb", 0)
    vram_cfg = router.config.get("vram", {})
    return jsonify({
        "status": "running",
        "uptime": time.time(),
        "models": health,
        "vram": vram,
        "vram_total_gb": vram_cfg.get("total_gb", 96),
        "gpu_count": vram_cfg.get("gpu_count", 6),
        "session": router.session_stats(),
        "cache": cache.stats(),
        "requests_served": len(request_log),
        "active_count": len(_active),
    })


@app.route("/api/active")
def api_active():
    """진행 중인 요청 (실시간 처리 패널용)"""
    return jsonify({"active": _active_list(), "server_time": time.time()})


@app.route("/api/slots")
def api_slots():
    """llama.cpp 백엔드의 /slots 상태 프록시 (슬롯별 실시간 진행)"""
    result = {}
    seen = set()
    for pname, pcfg in router.config.get("providers", {}).items():
        base = pcfg.get("base_url")
        if not base or pcfg.get("type") != "openai_compatible" or base in seen:
            continue
        seen.add(base)
        try:
            r = requests.get(f"{base}/slots", timeout=3)
            result[base] = r.json() if r.status_code == 200 else {"error": f"http_{r.status_code}"}
        except Exception as e:
            result[base] = {"error": str(e)}
    return jsonify({"backend": result})


@app.route("/api/logs")
def api_logs():
    limit = min(int(request.args.get("limit", 100)), 500)
    return jsonify({"logs": request_log[-limit:]})


@app.route("/api/config")
def api_config():
    safe = {"models": router.config.get("models", {}),
            "meta_routing": router.config.get("meta_routing", {}),
            "vram": router.config.get("vram", {}),
            "voice": router.config.get("voice", {}),
            "session": router.config.get("session", {}),
            "cache": router.config.get("cache", {})}
    return jsonify(safe)


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    cache.clear()
    return jsonify({"status": "cleared"})


@app.route("/api/classify", methods=["POST"])
def classify_prompt():
    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"error": "No messages"}), 400
    meta = router.meta
    result = meta.classify(data["messages"], router.model_mgr)
    return jsonify({
        "meta_routing": result,
        "session": router.session_stats(),
        "vram": router.vram_status(),
    })


# ============================================================
# 내부 함수
# ============================================================
def _build_response(result, model, cached=False):
    content = result.get("content", "")
    now = int(time.time())
    return jsonify({
        "id": f"chatcmpl-{now}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                      "finish_reason": "stop"}],
        "usage": result.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        "router_meta": {
            "task_type": result.get("task_type", "general"),
            "model_used": model,
            "meta_routing": result.get("meta"),
            "fallback_used": result.get("fallback_used", False),
            "cached": cached,
        },
    })


def _log_request(model, task_type, cached=False, error=False, elapsed_ms=0, tokens=0):
    entry = {"time": datetime.now().isoformat(), "model": model,
             "task_type": task_type, "cached": cached, "error": error,
             "elapsed_ms": elapsed_ms, "tokens": tokens}
    request_log.append(entry)
    if len(request_log) > MAX_LOG:
        request_log.pop(0)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5020))
    print(f"LLM Router v3 running on http://0.0.0.0:{port}")
    print(f"  Meta-router: qwen3.8 (always-on)")
    print(f"  Models: qwen3-coder (coding), qwq (math), qwen2.5vl (vision)")
    print(f"  Voice: STT(faster-whisper) → Router → TTS(edge-tts)")
    print(f"  VRAM: 96GB managed, qwen3.8 resident")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
