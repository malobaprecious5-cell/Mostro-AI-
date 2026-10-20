"""DeepSeek chat adapter with bounded retries and usage logging."""

from collections.abc import Sequence
from time import perf_counter

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import AIMessage
from langchain_deepseek import ChatDeepSeek

from ai.config import Settings
from ai.core.logging import get_logger
from ai.llm.retry import retrying

logger = get_logger(__name__)


class DeepSeekChatClient:
    def __init__(
        self,
        settings: Settings,
        *,
        model_name: str | None = None,
        model: BaseChatModel | None = None,
    ) -> None:
        self.model_name = model_name or settings.llm_model_main
        self.max_attempts = settings.provider_max_attempts
        if model is None:
            api_key = settings.deepseek_api_key.get_secret_value()
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY must be configured")
            model = ChatDeepSeek(
                model=self.model_name,
                temperature=0,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=0,
            )
        self._model = model

    async def ainvoke(self, input: str | Sequence[BaseMessage]) -> AIMessage:
        started_at = perf_counter()
        try:
            async for attempt in retrying(
                attempts=self.max_attempts,
                provider="deepseek",
                model=self.model_name,
                operation="chat",
            ):
                with attempt:
                    response = await self._model.ainvoke(input)
        except Exception as error:
            logger.error(
                "llm.request.failed",
                provider="deepseek",
                model=self.model_name,
                operation="chat",
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
                error_type=type(error).__name__,
            )
            raise

        usage = response.usage_metadata
        logger.info(
            "llm.request.completed",
            provider="deepseek",
            model=self.model_name,
            operation="chat",
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
            input_tokens=usage.get("input_tokens") if usage is not None else None,
            output_tokens=usage.get("output_tokens") if usage is not None else None,
            total_tokens=usage.get("total_tokens") if usage is not None else None,
        )
        return response
