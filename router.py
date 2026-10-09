"""LLM Router v3 - llama.cpp OpenAI-compatible 기반 (ollama 의존성 제거)"""
import json
import re
import time
import yaml
import requests
import threading
from datetime import datetime
from provider_hermes import call_hermes_llm, get_hermes_models


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class ModelManager:
    """모델 상태 관리 (llama.cpp는 항상 로드되어 있으므로 VRAM 관리 단순화)"""

    def __init__(self, config):
        self.cfg = config.get("vram", {})
        self.models_cfg = config.get("models", {})
        self._loaded = {}
        self._lock = threading.Lock()
        self._init_loaded()

    def _init_loaded(self):
        now = time.time()
        for name, cfg in self.models_cfg.items():
            if cfg.get("keep_warm", False):
                self._loaded[name] = {
                    "last_used": now,
                    "state": "loaded",
                    "keep_warm": True,
                }

    def touch(self, model_name):
        with self._lock:
            if model_name in self._loaded:
                self._loaded[model_name]["last_used"] = time.time()

    def ensure_loaded(self, model_name):
        with self._lock:
            if model_name in self._loaded:
                self._loaded[model_name]["last_used"] = time.time()
                return True, "already_loaded"
            cfg = self.models_cfg.get(model_name, {})
            self._loaded[model_name] = {
                "last_used": time.time(),
                "state": "loaded",
                "keep_warm": cfg.get("keep_warm", False),
            }
            return True, "loaded"

    def status(self):
        with self._lock:
            return {
                "loaded": {n: {"since": datetime.fromtimestamp(m["last_used"]).isoformat(),
                               "keep_warm": m.get("keep_warm", False)}
                          for n, m in self._loaded.items()},
                "loaded_count": len(self._loaded),
            }


class SessionTracker:
    """연속된 요청 유형 추적 → 모델 유지 결정"""

    def __init__(self, config):
        self.cfg = config.get("session", {})
        self.window_size = self.cfg.get("window_size", 5)
        self.model_hold = self.cfg.get("model_hold", 3)
        self.ttl = self.cfg.get("ttl", 600)
        self._history = []
        self._lock = threading.Lock()

    def record(self, task_type, model):
        now = time.time()
        with self._lock:
            cutoff = now - self.ttl
            self._history = [h for h in self._history if h[0] > cutoff]
            self._history.append((now, task_type, model))

    def get_suggestion(self):
        with self._lock:
            if not self._history:
                return None
            recent = [h[2] for h in self._history[-self.window_size:]]
            from collections import Counter
            counts = Counter(recent)
            most_common = counts.most_common(1)
            if most_common and most_common[0][1] >= self.model_hold:
                return most_common[0][0]
            return None

    def stats(self):
        with self._lock:
            return {
                "window_size": len(self._history),
                "recent_models": [h[2] for h in self._history[-10:]],
                "recent_tasks": [h[1] for h in self._history[-10:]],
            }


class MetaRouter:
    """qwen3.8이 요청 분석 → 특화 모델 호출 결정"""

    def __init__(self, config):
        self.cfg = config.get("meta_routing", {})
        self.fallback_cfg = self.cfg.get("fallback_classifier", {})
        self.router_model = self.cfg.get("router_model", "qwen3.8")
        self.routing_prompt = self.cfg.get("routing_prompt", "")

    def classify(self, messages, model_manager):
        text = ""
        has_image = False
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, list):
                for part in c:
                    if part.get("type") in ("image_url", "image"):
                        has_image = True
                    if part.get("type") == "text":
                        text += part.get("text", "") + " "
            else:
                text += c + " "

        if has_image:
            return {"task_type": "vision", "model": "qwen3-vl", "confidence": "HIGH",
                    "reasoning": "Image content detected", "complexity": "moderate"}

        try:
            routing_msg = [{"role": "system", "content": self.routing_prompt},
                           {"role": "user", "content": text[:2000]}]
            result = _call_openai_compatible(self.router_model, routing_msg,
                                             temperature=0.1, max_tokens=512,
                                             models_cfg=MetaRouter._get_models_cfg(),
                                             providers_cfg=MetaRouter._get_providers_cfg())
            if result["success"]:
                content = result["content"].strip()
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    required = ["task_type", "model"]
                    if all(k in parsed for k in required):
                        return parsed
        except Exception:
            pass

        return self._fallback_classify(text)

    def _fallback_classify(self, text):
        text_lower = text.lower()
        for kw in self.fallback_cfg.get("coding_keywords", []):
            if kw.lower() in text_lower:
                return {"task_type": "coding", "model": "qwen3.8", "confidence": "MEDIUM",
                        "reasoning": "Rule-based: coding keyword detected", "complexity": "moderate"}
        for kw in self.fallback_cfg.get("math_keywords", []):
            if kw in text_lower:
                return {"task_type": "math", "model": "qwen3.8", "confidence": "MEDIUM",
                        "reasoning": "Rule-based: math keyword detected", "complexity": "moderate"}
        for kw in self.fallback_cfg.get("vision_keywords", []):
            if kw in text_lower:
                return {"task_type": "vision", "model": "qwen3-vl", "confidence": "LOW",
                        "reasoning": "Rule-based: vision keyword detected", "complexity": "simple"}
        for kw in self.fallback_cfg.get("reasoning_keywords", []):
            if kw in text_lower:
                return {"task_type": "reasoning", "model": "qwen3.8", "confidence": "MEDIUM",
                        "reasoning": "Rule-based: reasoning keyword detected", "complexity": "moderate"}
        for kw in self.fallback_cfg.get("external_keywords", []):
            if kw in text_lower:
                return {"task_type": "multi_provider", "model": "hermes-config", "confidence": "MEDIUM",
                        "reasoning": "Rule-based: external API keyword detected", "complexity": "moderate"}
        return {"task_type": "general", "model": "qwen3.8", "confidence": "HIGH",
                "reasoning": "Default routing: general query", "complexity": "simple"}

    _models_cfg_cache = None
    _providers_cfg_cache = None

    @classmethod
    def _get_models_cfg(cls):
        return cls._models_cfg_cache

    @classmethod
    def _get_providers_cfg(cls):
        return cls._providers_cfg_cache


def _call_openai_compatible(model_name, messages, temperature=0.7, max_tokens=4096,
                            models_cfg=None, providers_cfg=None, timeout=180):
    """llama.cpp OpenAI-compatible /v1/chat/completions 호출"""
    mc = (models_cfg or {}).get(model_name, {})
    provider_name = mc.get("provider", "llamacpp_main")
    provider_cfg = (providers_cfg or {}).get(provider_name, {})
    base_url = provider_cfg.get("base_url", "http://127.0.0.1:8080")
    model_id = mc.get("model", provider_cfg.get("model_id", model_name))

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        if not content and reasoning:
            content = reasoning
        usage = data.get("usage", {})
        return {
            "content": content,
            "model": model_name,
            "usage": usage,
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "model": model_name}


class LLMRouter:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.model_mgr = ModelManager(self.config)
        self.session = SessionTracker(self.config)
        self.meta = MetaRouter(self.config)
        self._keepalive_running = False
        self._keepalive_thread = None
        MetaRouter._models_cfg_cache = self.config.get("models", {})
        MetaRouter._providers_cfg_cache = self.config.get("providers", {})

    def start_keepalive(self):
        if self._keepalive_running:
            return
        self._keepalive_running = True

        def _loop():
            interval = self.config.get("meta_routing", {}).get("keepalive_interval", 30)
            while self._keepalive_running:
                try:
                    _call_openai_compatible(
                        "qwen3.8",
                        [{"role": "user", "content": "ping"}],
                        temperature=0.0, max_tokens=1,
                        models_cfg=self.config.get("models", {}),
                        providers_cfg=self.config.get("providers", {}),
                        timeout=10,
                    )
                    self.model_mgr.touch("qwen3.8")
                except Exception:
                    pass
                time.sleep(interval)

        self._keepalive_thread = threading.Thread(target=_loop, daemon=True)
        self._keepalive_thread.start()

    def stop_keepalive(self):
        self._keepalive_running = False

    def route(self, messages, requested_model=None, **kwargs):
        task_type = None
        selected_model = requested_model
        meta_result = None
        errors = []

        meta_enabled = self.config.get("meta_routing", {}).get("enabled", True)
        if meta_enabled and not requested_model:
            meta_result = self.meta.classify(messages, self.model_mgr)
            task_type = meta_result.get("task_type", "general")
            selected_model = meta_result.get("model", "qwen3.8")
        else:
            task_type = self._rule_classify(messages)
            if not requested_model:
                selected_model = self._select_model(task_type)

        if not requested_model:
            session_suggestion = self.session.get_suggestion()
            if session_suggestion and session_suggestion != selected_model:
                selected_model = session_suggestion

        ok, msg = self.model_mgr.ensure_loaded(selected_model)
        if not ok:
            errors.append({"model": selected_model, "error": msg})
            fallback_models = self._get_fallback_chain(task_type)
            for fb_model in fallback_models:
                if fb_model == selected_model:
                    continue
                ok2, _ = self.model_mgr.ensure_loaded(fb_model)
                if ok2:
                    selected_model = fb_model
                    break
            else:
                return {"content": "", "model": selected_model, "task_type": task_type or "general",
                        "usage": {}, "fallback_used": True, "error": f"Load failed: {msg}"}

        try:
            result = self.call_model(selected_model, messages, **kwargs)
            if result["success"]:
                self.session.record(task_type or "general", selected_model)
                self.model_mgr.touch(selected_model)
                return {
                    "content": result["content"],
                    "model": selected_model,
                    "task_type": task_type,
                    "usage": result.get("usage", {}),
                    "meta": meta_result,
                    "fallback_used": bool(errors),
                }
            else:
                errors.append({"model": selected_model, "error": result["error"]})
        except Exception as e:
            errors.append({"model": selected_model, "error": str(e)})

        fallback_models = self._get_fallback_chain(task_type)
        for fb_model in fallback_models:
            if fb_model == selected_model:
                continue
            try:
                ok2, _ = self.model_mgr.ensure_loaded(fb_model)
                if not ok2:
                    continue
                result = self.call_model(fb_model, messages, **kwargs)
                if result["success"]:
                    self.session.record(task_type or "general", fb_model)
                    self.model_mgr.touch(fb_model)
                    return {
                        "content": result["content"],
                        "model": fb_model,
                        "task_type": task_type,
                        "usage": result.get("usage", {}),
                        "meta": meta_result,
                        "fallback_used": True,
                        "fallback_chain": errors,
                    }
            except Exception:
                continue

        return {"content": "", "model": selected_model, "task_type": task_type or "general",
                "usage": {}, "fallback_used": True, "error": f"All models failed: {errors}"}

    def call_model(self, model_name, messages, **kwargs):
        provider = self._get_provider_type(model_name)
        if provider == "hermes":
            return self._call_hermes(model_name, messages, **kwargs)
        return _call_openai_compatible(
            model_name, messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 4096),
            models_cfg=self.config.get("models", {}),
            providers_cfg=self.config.get("providers", {}),
        )

    def _call_hermes(self, model_name, messages, **kwargs):
        mc = self._get_model_cfg(model_name)
        hermes_provider = mc.get("hermes_provider")
        hermes_model = mc.get("hermes_model")
        try:
            result = call_hermes_llm(
                messages=messages,
                provider=None if hermes_provider is None else str(hermes_provider),
                model=None if hermes_model is None else str(hermes_model),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", mc.get("max_tokens", 4096)),
            )
            if result.get("success"):
                return {
                    "content": result["content"],
                    "model": model_name,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "success": True,
                }
            return result
        except Exception as e:
            return {"success": False, "error": str(e), "model": model_name}

    def _rule_classify(self, messages):
        text = ""
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if x.get("type") == "text")
            text += c + " "
        text_lower = text.lower()
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") in ("image_url", "image"):
                        return "vision"
        fbc = self.config.get("meta_routing", {}).get("fallback_classifier", {})
        for kw in fbc.get("coding_keywords", []):
            if kw.lower() in text_lower:
                return "coding"
        for kw in fbc.get("math_keywords", []):
            if kw in text_lower:
                return "math"
        for kw in fbc.get("reasoning_keywords", []):
            if kw in text_lower:
                return "reasoning"
        for kw in fbc.get("vision_keywords", []):
            if kw in text_lower:
                return "vision"
        return "general"

    def _select_model(self, task_type):
        rules = self.config.get("routing", {}).get("rules", [])
        for rule in rules:
            if rule.get("type") == task_type:
                return rule.get("priority", ["qwen3.8"])[0]
        return "qwen3.8"

    def _get_model_cfg(self, model_name):
        models = self.config.get("models", {})
        return models.get(model_name, {})

    def _get_provider_type(self, model_name):
        return self._get_model_cfg(model_name).get("provider", "llamacpp_main")

    def _get_fallback_chain(self, task_type):
        rules = self.config.get("routing", {}).get("rules", [])
        for rule in rules:
            if rule.get("type") == task_type:
                return rule.get("priority", [])
        return ["qwen3.8"]

    def check_models_health(self):
        status = {}
        providers_cfg = self.config.get("providers", {})
        models_cfg = self.config.get("models", {})
        for name, mc in models_cfg.items():
            provider_name = mc.get("provider", "")
            provider_cfg = providers_cfg.get(provider_name, {})
            base_url = provider_cfg.get("base_url", "")
            if not base_url:
                status[name] = "no_url"
                continue
            try:
                r = requests.get(f"{base_url}/v1/models", timeout=5)
                status[name] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
            except Exception:
                status[name] = "unavailable"
        return status

    def vram_status(self):
        return self.model_mgr.status()

    def session_stats(self):
        return self.session.stats()
