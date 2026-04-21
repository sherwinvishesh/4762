import os
import re
import time
import requests
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

def call_model_chat_completions(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 60,
    max_retries: int = 3,
) -> Dict[str, Any]:
    if not API_KEY:
        return {"ok": False, "text": None, "status": -1,
                "error": "OPENAI_API_KEY environment variable is not set."}
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            status = resp.status_code
            if status == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return {"ok": True, "text": text, "status": status, "error": None}

            retryable = (status == 429) or (500 <= status < 600)
            if retryable and attempt < max_retries - 1:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after), 10))
                else:
                    time.sleep(2 ** attempt)
                continue
            return {"ok": False, "text": None, "status": status, "error": resp.text}
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
    return {"ok": False, "text": None, "status": -1, "error": last_error}

_ANSWER_PATTERNS = [
    r"(?:^|\n)\s*(?:\*\*)?\s*(?:final\s+)?answer\s*(?:\*\*)?\s*[:=]\s*(.+?)(?:\n|$)",
    # GSM8K-style '####'
    r"####\s*(.+?)(?:\n|$)",
    # LaTeX boxed answers
    r"\\boxed\{([^{}]+)\}",
    # "The answer is X"
    r"the\s+answer\s+is\s+(.+?)(?:[.\n]|$)",
]

def _clean(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", s)
    s = s.strip("*$`_ ").strip()
    s = s.rstrip(".,;:!?")
    return s.strip()

def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    for pat in _ANSWER_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            ans = _clean(m.group(1))
            if ans:
                return ans
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    return _clean(lines[-1]) if lines else ""

def extract_number(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"\\boxed\{([^{}]+)\}", text)
    if m:
        n = re.search(r"-?\d+(?:\.\d+)?", m.group(1))
        if n:
            return n.group(0)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""
