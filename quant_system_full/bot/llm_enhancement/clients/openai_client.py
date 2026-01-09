"""
OpenAI Client Wrapper

Provides a wrapper around OpenAI API with:
- Error handling and retries
- Cost tracking
- Timeout management
- Response validation
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Pricing (per 1M tokens) for GPT-5 and GPT-4o models
PRICING = {
    # GPT-5 models (2025)
    "gpt-5-nano": {
        "input": 0.05,    # $0.05 per 1M input tokens
        "output": 0.40    # $0.40 per 1M output tokens
    },
    "gpt-5-mini": {
        "input": 0.25,    # $0.25 per 1M input tokens
        "output": 2.00    # $2.00 per 1M output tokens
    },
    "gpt-5": {
        "input": 1.25,    # $1.25 per 1M input tokens
        "output": 10.00   # $10.00 per 1M output tokens
    },
    # GPT-4o models (legacy)
    "gpt-4o-mini": {
        "input": 0.150,   # $0.150 per 1M input tokens
        "output": 0.600   # $0.600 per 1M output tokens
    },
    "gpt-4o": {
        "input": 2.50,    # $2.50 per 1M input tokens
        "output": 10.00   # $10.00 per 1M output tokens
    },
    "gpt-4o-2024-08-06": {
        "input": 2.50,
        "output": 10.00
    }
}


class OpenAIClient:
    """
    Wrapper around OpenAI API for LLM enhancement system.

    Features:
    - Automatic retries with exponential backoff
    - Cost tracking per request
    - Timeout management
    - Response validation
    - Error handling
    """

    def __init__(self, config):
        """
        Initialize OpenAI client.

        Args:
            config: LLMEnhancementConfig instance
        """
        self.config = config
        self._client = None

        # Track usage
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.openai_api_key,
                    timeout=self.config.timeout
                )
                logger.debug("[LLM] OpenAI client initialized")
            except Exception as e:
                logger.error(f"[LLM] Failed to initialize OpenAI client: {e}")
                raise
        return self._client

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call OpenAI API with retry logic.

        Args:
            prompt: The prompt text
            model: Model name (defaults to config.model_triage)
            temperature: Sampling temperature (defaults to config.temperature)
            max_tokens: Maximum tokens in response
            response_format: Response format (e.g., {"type": "json_object"})
            reasoning_effort: GPT-5 reasoning effort level (minimal|low|medium|high)

        Returns:
            dict: {
                "content": str,           # Response text
                "input_tokens": int,      # Tokens in prompt
                "output_tokens": int,     # Tokens in response
                "cost": float,            # Cost in USD
                "model": str,             # Model used
                "timestamp": str          # ISO format timestamp
            }

        Raises:
            Exception: If all retries fail
        """
        model = model or self.config.model_triage
        temperature = temperature if temperature is not None else self.config.temperature

        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug(f"[LLM] Calling OpenAI (attempt {attempt + 1}/{self.config.max_retries + 1})")

                # Build request
                messages = [{"role": "user", "content": prompt}]

                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature
                }

                if max_tokens:
                    kwargs["max_tokens"] = max_tokens

                if response_format:
                    kwargs["response_format"] = response_format

                # GPT-5 specific: Add reasoning_effort parameter
                if reasoning_effort and model.startswith("gpt-5"):
                    kwargs["reasoning_effort"] = reasoning_effort
                    logger.debug(f"[LLM] Using GPT-5 reasoning_effort={reasoning_effort}")

                # Make API call
                start_time = time.time()
                response = self.client.chat.completions.create(**kwargs)
                elapsed = time.time() - start_time

                # Extract response
                content = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

                # Calculate cost
                cost = self._calculate_cost(model, input_tokens, output_tokens)

                # Update tracking
                self.total_calls += 1
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_cost += cost

                result = {
                    "content": content,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost,
                    "model": model,
                    "elapsed": elapsed,
                    "timestamp": datetime.now().isoformat()
                }

                logger.info(
                    f"[LLM_API] Call successful: "
                    f"model={model}, tokens={input_tokens}in+{output_tokens}out, "
                    f"cost=${cost:.6f}, latency={elapsed:.2f}s"
                )

                # Also log summary for easier tracking
                logger.debug(
                    f"[LLM_API] Session stats: "
                    f"total_calls={self.total_calls}, "
                    f"total_tokens={self.total_input_tokens + self.total_output_tokens}, "
                    f"cumulative_cost=${self.total_cost:.4f}"
                )

                return result

            except Exception as e:
                last_error = e
                logger.warning(f"[LLM] API call failed (attempt {attempt + 1}): {e}")

                # Don't retry on certain errors
                error_str = str(e).lower()
                if "invalid_api_key" in error_str or "authentication" in error_str:
                    logger.error(f"[LLM] Authentication error, not retrying: {e}")
                    raise

                if "context_length_exceeded" in error_str:
                    logger.error(f"[LLM] Context length exceeded, not retrying: {e}")
                    raise

                # Exponential backoff
                if attempt < self.config.max_retries:
                    sleep_time = 2 ** attempt  # 1s, 2s, 4s, ...
                    logger.debug(f"[LLM] Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

        # All retries exhausted
        logger.error(f"[LLM] All retries exhausted. Last error: {last_error}")
        raise Exception(f"OpenAI API call failed after {self.config.max_retries + 1} attempts: {last_error}")

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for API call.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        # Get pricing for model (default to gpt-4o-mini if unknown)
        pricing = PRICING.get(model, PRICING["gpt-4o-mini"])

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            dict: {
                "total_calls": int,
                "total_input_tokens": int,
                "total_output_tokens": int,
                "total_cost": float
            }
        """
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost
        }

    def reset_usage_stats(self):
        """Reset usage statistics (useful for per-run tracking)."""
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        logger.debug("[LLM] Usage stats reset")
