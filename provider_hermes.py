"""
Hermes provider adapter for LLM Router.
Reads hermes config.yaml and routes through hermes's chosen model/provider pipeline.
Supports all hermes providers: ollama, openai, anthropic, gemini, deepseek, etc.
"""
import os
import json
import subprocess
import tempfile
import yaml
from pathlib import Path

HERMES_DIR = Path.home() / ".hermes"
HERMES_CONFIG = HERMES_DIR / "config.yaml"
HERMES_VENV = HERMES_DIR / "hermes-agent" / "venv"


def get_hermes_models():
    """Discover available models from hermes config"""
    try:
        with open(HERMES_CONFIG) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return {"default_provider": "ollama", "default_model": "qwen3.8", "available": []}

    model_cfg = cfg.get("model", {})
    default_provider = model_cfg.get("provider", "ollama")
    default_model = model_cfg.get("model", "qwen3.8")

    providers_cfg = cfg.get("providers", {})
    available = []
    for pname, pcfg in providers_cfg.items():
        base_url = pcfg.get("base_url", "")
        models = pcfg.get("models", [])
        available.append({
            "provider": pname,
            "type": pcfg.get("type", "openai"),
            "base_url": base_url,
            "models": models,
        })

    return {
        "default_provider": default_provider,
        "default_model": default_model,
        "available_providers": available,
    }


def call_hermes_llm(messages, provider=None, model=None, temperature=0.7, max_tokens=4096, timeout=180):
    """
    Call LLM through hermes's plugin_llm pipeline.
    Uses hermes venv for full provider support (OpenAI, Anthropic, Gemini, etc.)
    """
    cfg = get_hermes_models()
    provider = provider or cfg["default_provider"]
    model = model or cfg["default_model"]

    params = {
        "messages": messages,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(params, f)
        param_file = f.name

    script = f"""
import sys, json
sys.path = ['{HERMES_VENV}/lib/python3.11/site-packages', '{HERMES_DIR / "hermes-agent"}'] + sys.path

with open('{param_file}') as f:
    p = json.load(f)

# Use PluginLLM for full hermes provider support
from agent.plugin_llm import PluginLLM
from hermes_state import HermesConfig

try:
    config = HermesConfig(config_path='{HERMES_CONFIG}')
    llm = PluginLLM(config=config)
    result = llm.complete(
        messages=p['messages'],
        provider=p.get('provider'),
        model=p.get('model'),
        temperature=p.get('temperature', 0.7),
        max_tokens=p.get('max_tokens', 4096),
    )
    print(json.dumps({{"content": result, "success": True}}, ensure_ascii=False))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)[:500]}}, ensure_ascii=False))
"""
    try:
        r = subprocess.run(
            [str(HERMES_VENV / "bin" / "python3"), "-c", script],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ},
        )
        out = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else ""
        if out:
            return json.loads(out)
        err = r.stderr.strip()[:300] if r.stderr.strip() else "empty response"
        return {"success": False, "error": err}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "hermes LLM timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        try:
            os.unlink(param_file)
        except Exception:
            pass
