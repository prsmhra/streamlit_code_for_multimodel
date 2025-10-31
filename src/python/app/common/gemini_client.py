"""
Wrapper + helpers for calling different GenAI SDK shapes and extracting text.
Provides:
- _call_flexible(callable_obj, kwargs: dict): call a callable with only the parameters it accepts (best-effort).
- _extract_text_from_genai_response(response): robust extractor for many SDK response shapes.
- GeminiClientWrapper: convenience wrapper that tries a number of likely client call sites.
"""
from typing import Any, Dict, Optional
import inspect
import json


def _call_flexible(callable_obj, kwargs: Dict[str, Any]):
    """
    Call callable_obj using only the kwargs its signature accepts.
    - callable_obj: a function / bound method to call
    - kwargs: a dict of potential keyword arguments
    Returns the result of the call or raises the original exception.
    Best-effort behavior:
      1) Inspect signature and filter kwargs to accepted names.
      2) Try callable_obj(**filtered_kwargs).
      3) If that fails, try callable_obj(**kwargs) (some SDKs accept **kwargs despite signatures).
      4) As a last resort, if callable_obj is a bound method that expects positional args, try passing the whole dict as single positional arg.
    """
    # Fast path if callable is None
    if callable_obj is None:
        raise RuntimeError("Callable is None")

    # Try to inspect signature and call with matching params
    try:
        sig = inspect.signature(callable_obj)
        accepted = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        try:
            return callable_obj(**filtered)
        except TypeError:
            # try calling with all kwargs
            return callable_obj(**kwargs)
    except (ValueError, TypeError):
        # try direct call with kwargs
        try:
            return callable_obj(**kwargs)
        except Exception:
            # last: try passing the kwargs dict as a single positional arg
            try:
                return callable_obj(kwargs)
            except Exception as e:
                # re-raise the last exception for caller to handle
                raise e


def _extract_text_from_genai_response(response: Any) -> Optional[str]:
    """
    Robust extractor that attempts to return human-readable text from a response object.
    Handles many common shapes:
    - response.text
    - response.candidates -> content.parts -> text/content
    - response.output (list / dict) with content/text/payload
    - response.choices (list of dicts or objects with .text)
    - fallback to str(response)
    """
    if response is None:
        return None

    # 1) common attribute .text
    try:
        if hasattr(response, "text") and getattr(response, "text"):
            txt = getattr(response, "text")
            return txt if isinstance(txt, str) else str(txt)
    except Exception:
        pass

    # 2) candidates style (some SDKs)
    try:
        candidates = getattr(response, "candidates", None)
        if candidates:
            parts = []
            for c in candidates:
                # candidate.content.parts[*].text or candidate.content
                content = getattr(c, "content", None)
                if content is not None:
                    maybe_parts = getattr(content, "parts", None)
                    if maybe_parts:
                        for p in maybe_parts:
                            t = getattr(p, "text", None) or getattr(p, "content", None)
                            if t:
                                parts.append(str(t))
                        continue
                # fallback: the candidate itself may have a string representation
                parts.append(str(c))
            if parts:
                return "\n\n".join(parts)
    except Exception:
        pass

    # 3) .output field (list / dict)
    try:
        out = getattr(response, "output", None)
        if out is not None:
            if isinstance(out, (list, tuple)) and out:
                # find first dict with content/text/payload
                for item in out:
                    if isinstance(item, dict):
                        for key in ("content", "text", "payload"):
                            if key in item and item[key]:
                                return item[key] if isinstance(item[key], str) else str(item[key])
                # fallback: join list items that are strings
                str_items = [str(i) for i in out if isinstance(i, (str, int, float))]
                if str_items:
                    return "\n\n".join(str_items)
            elif isinstance(out, dict):
                for key in ("content", "text", "payload"):
                    if key in out and out[key]:
                        return out[key] if isinstance(out[key], str) else str(out[key])
    except Exception:
        pass

    # 4) choices style (OpenAI-like)
    try:
        choices = getattr(response, "choices", None)
        if choices:
            texts = []
            for ch in choices:
                if isinstance(ch, dict) and "text" in ch and ch["text"]:
                    texts.append(ch["text"])
                else:
                    t = getattr(ch, "text", None)
                    if t:
                        texts.append(t)
            if texts:
                return "\n\n".join(texts)
    except Exception:
        pass

    # 5) Try common dict-like shape
    try:
        if isinstance(response, dict):
            for key in ("text", "output", "content", "payload"):
                if key in response and response[key]:
                    return response[key] if isinstance(response[key], str) else str(response[key])
    except Exception:
        pass

    # 6) fallback to pretty-printed JSON for objects that are serializable
    try:
        return json.dumps(response, default=lambda o: getattr(o, "__dict__", str(o)), ensure_ascii=False)
    except Exception:
        try:
            return str(response)
        except Exception:
            return None


class GeminiClientWrapper:
    """
    Thin wrapper around a genai client object. The wrapper exposes a `generate` method
    that will attempt multiple likely call sites on the provided client instance.

    Usage:
        wrapper = GeminiClientWrapper(client)
        resp = wrapper.generate("gemini-2.5-pro", instructions, inputs=..., contents=...)
    """
    def __init__(self, client: Any = None):
        self.client = client

    def generate(self, model_name: str, instructions: str, inputs: Dict[str, Any] = None, contents: Any = None, **kwargs) -> Any:
        """
        Try multiple possible call shapes on the provided client object.
        Tries (in approximate order):
          - client.models.generate_content(model=..., instructions=..., ...)
          - client.responses.create(model=..., instructions=..., ...)
          - client.generate(model=..., contents=..., ...)
          - client.generate_text / client.models.generate etc.
        The method builds a payload dict and uses _call_flexible to adapt to different SDK signatures.
        Returns the raw SDK response object from the first successful call.
        Raises RuntimeError if all attempts fail.
        """
        if self.client is None:
            raise RuntimeError("No client provided to GeminiClientWrapper")

        # Prepare candidate callables (best-effort discovery)
        candidates = []

        # Try known attribute paths and append callables if found
        try:
            if hasattr(self.client, "models") and hasattr(self.client.models, "generate_content"):
                candidates.append(("client.models.generate_content", self.client.models.generate_content))
        except Exception:
            pass

        try:
            if hasattr(self.client, "responses") and hasattr(self.client.responses, "create"):
                candidates.append(("client.responses.create", self.client.responses.create))
        except Exception:
            pass

        # direct methods on the client object
        for attr in ("generate", "generate_text", "create", "responses", "models"):
            try:
                obj = self.client
                for part in (attr.split(".") if "." in attr else (attr,)):
                    obj = getattr(obj, part)
                # Only append callables
                if callable(obj):
                    candidates.append((f"client.{attr}", obj))
            except Exception:
                # ignore missing attributes
                pass

        # Build a payload that covers the common parameter names used across SDK variants
        payload = {}
        # prefer 'model' and 'instructions' for structured instructions-based SDKs
        payload["model"] = model_name
        if instructions is not None:
            payload["instructions"] = instructions
        if inputs is not None:
            payload["inputs"] = inputs
        if contents is not None:
            # some SDKs call it 'contents' (list) while others use 'instructions'
            payload["contents"] = contents

        # Merge any additional kwargs provided by the caller (but they can be filtered by _call_flexible)
        merged_kwargs = dict(payload)
        merged_kwargs.update(kwargs)

        last_exc = None
        for name, cand in candidates:
            try:
                # Use the flexible caller which accepts (callable_obj, kwargs_dict)
                resp = _call_flexible(cand, merged_kwargs)
                return resp
            except Exception as e:
                last_exc = e
                continue

        # As a last resort, try calling the client object itself (some SDKs expose a direct call)
        try:
            resp = _call_flexible(self.client, merged_kwargs)
            return resp
        except Exception as e:
            last_exc = e

        raise RuntimeError("All call attempts failed in GeminiClientWrapper.generate: " + (str(last_exc) if last_exc else "unknown error"))


def response_requests_tool_call(response: Any) -> Optional[Dict[str, Any]]:
    """
    Inspect response object for a tool/function call. Return dict:
    {"tool_name": str, "arguments": dict, "raw": response}
    or None if no tool call requested.
    This is best-effort: SDKs differ in how they encode tool calls.
    """
    if response is None:
        return None
    # Common shapes:
    # 1) response.tool_calls or response.output[0].content.tool_call
    # 2) response.choices[0].message.get("tool_call")
    # 3) response.candidates with a tool_call payload
    try:
        # attempt several heuristics:
        if hasattr(response, "tool_calls") and getattr(response, "tool_calls"):
            tc = response.tool_calls[0]
            return {"tool_name": getattr(tc, "name", None), "arguments": getattr(tc, "arguments", {}) , "raw": response}
        # check .choices -> message -> 'tool_call' (openai-like)
        if hasattr(response, "choices"):
            for ch in response.choices:
                # if dict-like
                if isinstance(ch, dict):
                    msg = ch.get("message") or ch.get("text")
                    if isinstance(msg, dict) and "tool_call" in msg:
                        tc = msg["tool_call"]
                        return {"tool_name": tc.get("name"), "arguments": tc.get("arguments"), "raw": response}
                else:
                    # try object attributes
                    msg = getattr(ch, "message", None)
                    if hasattr(msg, "tool_call"):
                        tc = getattr(msg, "tool_call")
                        return {"tool_name": getattr(tc,"name", None), "arguments": getattr(tc,"arguments", {}) , "raw": response}
        # fallback: try to parse a 'tool_call:' JSON chunk in text
        text = _extract_text_from_genai_response(response)
        if text and "tool_call" in text:
            try:
                # look for a JSON blob after "tool_call:" or similar
                idx = text.find("{")
                j = json.loads(text[idx:])
                if "name" in j and "arguments" in j:
                    return {"tool_name": j["name"], "arguments": j["arguments"], "raw": response}
            except Exception:
                pass
    except Exception:
        pass
    return None
