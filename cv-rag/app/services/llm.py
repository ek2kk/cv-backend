import time

from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam

from app.core.config import get_settings
from app.core.logging import get_logger
from app.dependencies import get_llm_client
from app.exceptions import LlmConfigError, LlmResponseError

settings = get_settings()
logger = get_logger(__name__)


MODELS = [
    settings.openrouter.model,
    "qwen/qwen3-next-80b:free",
    "deepseek/deepseek-chat-v3:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]


def call_llm(messages: list[ChatCompletionMessageParam], max_tokens: int = 500) -> str:
    if not settings.openrouter.api_key:
        raise LlmConfigError("OpenRouter API key is not configured")

    last_error: Exception | None = None

    for model in MODELS:
        for attempt in range(2):
            try:
                response = get_llm_client().chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                if content:
                    logger.info("LLM request succeeded", extra={"extra_data": {"model": model}})
                    return content

                last_error = LlmResponseError(f"Model {model} returned an empty answer")

            except (RateLimitError, APIError, APIConnectionError, APITimeoutError) as e:
                last_error = e
                logger.warning(
                    "LLM request failed",
                    extra={"extra_data": {"model": model, "attempt": attempt + 1}},
                )
                time.sleep(1.2 * (attempt + 1))

    if last_error:
        raise last_error

    raise LlmConfigError("No LLM models were configured")
