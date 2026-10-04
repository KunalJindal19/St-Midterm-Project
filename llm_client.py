"""
LLM Client utility for the AI-assisted unit testing pipeline.
Provides a shared LLM calling function with automatic model fallback
when the primary model is rate-limited or unavailable.
"""

import json
import time
import requests

from config import Config


def call_llm(config: Config, system_prompt: str, user_prompt: str) -> tuple[str, str]:
    """
    Call the OpenRouter API with automatic fallback to alternative models.

    Tries the primary model first, then falls back to other free models
    if the primary returns a 404 (not found) or 429 (rate limited).

    Args:
        config: Pipeline configuration with API key and model settings.
        system_prompt: The system prompt for the LLM.
        user_prompt: The user prompt for the LLM.

    Returns:
        Tuple of (response_text, model_used).

    Raises:
        RuntimeError: If all models fail.
    """
    # Build the list of models to try: primary first, then fallbacks
    models_to_try = [config.model]
    for fallback in (config.fallback_models or []):
        if fallback != config.model and fallback not in models_to_try:
            models_to_try.append(fallback)

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai-unit-testing-pipeline",
    }

    last_error = None

    for model in models_to_try:
        payload = {
            "model": model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            response = requests.post(
                config.api_base_url,
                headers=headers,
                json=payload,
                timeout=90,
            )

            # If rate-limited (429), wait and retry the same model once
            if response.status_code == 429:
                try:
                    retry_after = float(response.headers.get("Retry-After", "5"))
                except ValueError:
                    retry_after = 5
                retry_after = min(max(retry_after, 1), 10)  # Cap at 10 seconds
                time.sleep(retry_after)

                response = requests.post(
                    config.api_base_url,
                    headers=headers,
                    json=payload,
                    timeout=90,
                )

            # If we still get a retryable error, try the next model
            if response.status_code in (404, 429, 502, 503):
                last_error = f"Model {model}: HTTP {response.status_code}"
                try:
                    err_data = response.json()
                    err_msg = err_data.get("error", {}).get("message", "")
                    last_error = f"Model {model}: {err_msg}"
                except Exception:
                    pass
                time.sleep(1)  # Brief pause before trying next model
                continue

            response.raise_for_status()
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0]["message"]
                # Only the final answer is used. Reasoning/thinking fields are never read,
                # otherwise a reasoning model's intermediate text would be parsed as code/tests.
                content = message.get("content") or ""

                if not content.strip():
                    last_error = f"Model {model}: Empty response content"
                    continue

                # A response cut off at max_tokens is incomplete (typically a model that wrote out
                # its reasoning instead of the answer), so it is not used.
                if data["choices"][0].get("finish_reason") == "length":
                    last_error = f"Model {model}: response truncated at max_tokens ({config.max_tokens})"
                    continue

                return content, model
            elif "error" in data:
                last_error = f"Model {model}: API error: {data['error']}"
                continue
            else:
                last_error = f"Model {model}: Unexpected response: {json.dumps(data)[:200]}"
                continue

        except requests.exceptions.Timeout:
            last_error = f"Model {model}: Request timed out"
            continue
        except requests.exceptions.HTTPError as e:
            last_error = f"Model {model}: {e}"
            continue
        except Exception as e:
            last_error = f"Model {model}: {type(e).__name__}: {e}"
            continue

    raise RuntimeError(f"All models failed. Last error: {last_error}")
