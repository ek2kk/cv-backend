import time

from openai import APIError, OpenAI, RateLimitError

from settings import settings

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.openrouter_api_key,
)


MODELS = [
    settings.openrouter_model,
    "qwen/qwen3-next-80b:free",
    "deepseek/deepseek-chat-v3:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]


def call_llm(messages: list[dict], max_tokens: int = 500) -> str:
    last_error: Exception | None = None

    for model in MODELS:
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                if content:
                    return content

                last_error = RuntimeError(f"Model {model} returned an empty answer")

            except (RateLimitError, APIError) as e:
                last_error = e
                time.sleep(1.2 * (attempt + 1))

    if last_error:
        raise last_error

    raise RuntimeError("No LLM models were configured")
