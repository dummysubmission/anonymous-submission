"""Model client and JSON helpers for dataset filter (MIF / m3risk).

Portkey uses httpx with ``trust_env=False`` and a robust ``verify`` SSL context so
behavior matches ``data_curation/consolidate_sample.py`` on picky OpenSSL / HPC setups.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any


def pil_image_to_data_url(img: Any) -> str:
    buf = io.BytesIO()
    try:
        img.save(buf, format="PNG")
    except Exception:
        img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def save_json_list(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=4)
        f.write("\n")


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, found {type(data).__name__}.")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected every entry in {path} to be a JSON object.")
    return data


def get_model_name(client: Any, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model
    models = client.models.list().data
    if not models:
        raise RuntimeError("No models available from the inference server.")
    return models[0].id


def extract_non_thinking_section(text: str) -> str:
    think_idx = text.find("</redacted_thinking>")
    if think_idx == -1:
        return text.strip()
    return text[think_idx + len("</redacted_thinking>") :].strip()


def parse_json_payload(text: str) -> Any:
    if not text:
        return None

    def _literal_value(blob: str) -> Any:
        try:
            return ast.literal_eval(blob)
        except Exception:
            return None

    def _parse(blob: str) -> Any:
        blob = blob.strip()
        if not blob:
            return None
        if blob.startswith("```"):
            blob = re.sub(r"^```(?:json)?\s*", "", blob.strip(), flags=re.IGNORECASE)
            blob = re.sub(r"\s*```$", "", blob.strip())

        try:
            parsed = json.loads(blob)
        except Exception:
            parsed = None

        if isinstance(parsed, str):
            try:
                return json.loads(parsed)
            except Exception:
                literal_parsed = _literal_value(parsed)
                if literal_parsed is not None:
                    return literal_parsed

        if parsed is not None:
            return parsed

        literal_parsed = _literal_value(blob)
        if literal_parsed is not None:
            return literal_parsed

        if any(token in blob for token in ('\\"', "\\n", "\\t", "\\u")):
            try:
                unescaped = bytes(blob, "utf-8").decode("unicode_escape")
            except Exception:
                unescaped = None
            if unescaped is not None:
                try:
                    return json.loads(unescaped)
                except Exception:
                    return _literal_value(unescaped)

        return None

    parsed = _parse(text)
    if parsed is not None:
        return parsed

    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if not match:
        return None
    return _parse(match.group(0))


def _repo_root_default() -> Path:
    """Repository root (parent of the ``dataset_filter`` package)."""
    return Path(__file__).resolve().parent.parent


def resolve_image_path(
    sample: dict[str, Any], *, repo_root: Path | None = None
) -> Path | None:
    root = repo_root if repo_root is not None else _repo_root_default()
    raw = (sample.get("image_path") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = root / p
    return p if p.is_file() else None


def is_null_like(value: Any) -> bool:
    return value in (None, "", "null", "None")


def _httpx_verify_ssl_context() -> Any:
    """Build ``ssl.SSLContext`` for httpx ``verify=`` on picky OpenSSL / HPC setups.

    Passing ``verify=<path>`` still ends up in ``ssl.create_default_context(cafile=...)``
    and can raise ``ssl.SSLError`` on some login-node Python+OpenSSL builds (including
    with certifi's bundle). Order of attempts:

    1. ``MSLM_HTTPS_CA_BUNDLE`` (project override), then ``REQUESTS_CA_BUNDLE``,
       ``CURL_CA_BUNDLE``, ``SSL_CERT_FILE`` if each points to an existing file.
    2. ``ssl.create_default_context()`` with no ``cafile`` (OpenSSL default store).
    3. Common distro paths (RHEL ``/etc/pki/tls/certs/ca-bundle.crt``, Debian-style
       ``/etc/ssl/certs/ca-certificates.crt``).
    4. certifi's bundle.

    Set e.g. ``export MSLM_HTTPS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt`` before
    starting the Flask app if the automatic chain still fails.
    """
    import ssl

    def _try_cafile(path: str) -> Any | None:
        if not path or not Path(path).is_file():
            return None
        try:
            return ssl.create_default_context(cafile=path)
        except ssl.SSLError:
            return None

    for key in (
        "MSLM_HTTPS_CA_BUNDLE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "SSL_CERT_FILE",
    ):
        raw = os.environ.get(key)
        if raw:
            ctx = _try_cafile(raw.strip())
            if ctx is not None:
                return ctx

    try:
        return ssl.create_default_context()
    except ssl.SSLError:
        pass

    for syspath in (
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
        "/etc/ssl/certs/ca-certificates.crt",
    ):
        ctx = _try_cafile(syspath)
        if ctx is not None:
            return ctx

    import certifi

    ctx = _try_cafile(certifi.where())
    if ctx is not None:
        return ctx

    raise RuntimeError(
        "Could not create an SSL context for Portkey (httpx). "
        "Try: export MSLM_HTTPS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt "
        "(or your site's PEM bundle), then restart the server."
    )


def make_portkey_client(api_key: str) -> Any:
    """Portkey client with httpx ``trust_env=False`` and a robust ``verify`` SSL context.

    Avoids Portkey's default client, which uses env-based CA selection that often
    breaks on Slurm login nodes; see ``_httpx_verify_ssl_context``.
    """
    import httpx
    from portkey_ai import Portkey
    from portkey_ai.api_resources.global_constants import DEFAULT_CONNECTION_LIMITS
    from portkey_ai.api_resources.utils import set_base_url

    base_url = set_base_url(None, api_key)
    http_client = httpx.Client(
        base_url=base_url,
        headers={"Accept": "application/json"},
        limits=DEFAULT_CONNECTION_LIMITS,
        verify=_httpx_verify_ssl_context(),
        trust_env=False,
    )
    return Portkey(api_key=api_key, http_client=http_client)


def chat_completions_create(
    client: Any,
    model_name: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    openai_compatible: bool,
) -> Any:
    """OpenAI-compatible chat completion; mirrors ``call_model`` in Stage 1/2 scripts."""
    if openai_compatible:
        request_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    else:
        if "gpt-5" in model_name:
            request_kwargs = {
                "model": model_name,
                "messages": messages,
                "max_completion_tokens": max_tokens,
            }
        else:
            request_kwargs = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
    try:
        return client.chat.completions.create(
            **request_kwargs,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
    except TypeError:
        return client.chat.completions.create(**request_kwargs)
    except Exception as exc:
        if "extra_body" not in str(exc) and "chat_template_kwargs" not in str(exc):
            raise
        return client.chat.completions.create(**request_kwargs)
