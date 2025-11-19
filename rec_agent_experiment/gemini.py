# gemini.py
"""
Safe Gemini LLM wrapper following the same interface as LLMBase from llm.py
Includes full safety handling for missing content, finish_reason errors,
and provides a stable interface for websocietysimulator RecAgent.
"""

from typing import Dict, List, Optional, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import google.generativeai as genai
import logging
import os

# Import LLMBase from your existing llm.py
from websocietysimulator.llm import LLMBase


logger = logging.getLogger("websocietysimulator")


class GeminiLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """
        Initialize Gemini LLM.

        Args:
            api_key: Google API key
            model: Gemini model name
        """
        super().__init__(model)
        genai.configure(api_key=api_key)

        # Base model instance
        self.client = genai.GenerativeModel(model)
        self.embedding_model = None

    # ---------------------------
    # SAFE WRAPPED CALL
    # ---------------------------
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=3, max=40),
        stop=stop_after_attempt(4),
    )
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 5000,
        stop_strs: Optional[List[str]] = None,
        n: int = 1,
    ) -> Union[str, List[str]]:
        """
        Call Gemini in an OpenAI-like interface.
        Ensures safety for finish_reason != 0 and missing content.

        Args:
            messages: list of {"role": "...", "content": "..."}
            model: optional override model name
            temperature: sampling temperature
            max_tokens: output token limit
            stop_strs: stop sequences
            n: number of generations

        Returns:
            str or list[str]
        """

        # Convert messages into a single prompt string
        prompt = ""
        for m in messages:
            prompt += f"{m['role'].upper()}: {m['content']}\n"

        responses = []

        for _ in range(n):
            try:
                # ---------------------------
                # Call Gemini
                # ---------------------------
                response = self.client.generate_content(
                    contents=prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "stop_sequences": stop_strs or [],
                    },
                )

                # ---------------------------
                # Extract candidate
                # ---------------------------
                if not response.candidates:
                    logger.error("Gemini: No candidates returned.")
                    responses.append("[ERROR: no candidates]")
                    continue

                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                finish_reason_map = {
                    0: "FINISH_REASON_UNSPECIFIED",
                    1: "STOP",
                    2: "MAX_TOKENS",
                    3: "SAFETY",
                    4: "RECITATION",
                    5: "OTHER",
                }

                # ---------------------------
                # SAFETY / ERROR HANDLING
                # ---------------------------
                if finish_reason not in (0, 1):
                    reason_name = finish_reason_map.get(finish_reason, "UNKNOWN")
                    logger.error(
                        "Gemini blocked output (finish_reason=%s [%s]). "
                        "Returning safe empty string.",
                        finish_reason,
                        reason_name,
                    )
                    responses.append(f"[ERROR: finish_reason={finish_reason}]")
                    continue

                if not candidate.content or len(candidate.content.parts) == 0:
                    logger.error("Gemini: Empty content parts.")
                    responses.append("[ERROR: empty content]")
                    continue

                # ---------------------------
                # Extract actual text from parts
                # ---------------------------
                parts = candidate.content.parts
                text_out = ""

                for p in parts:
                    if hasattr(p, "text"):
                        text_out += p.text

                if text_out.strip() == "":
                    logger.error("Gemini: Parts exist but have no text.")
                    responses.append("[ERROR: no text extracted]")
                    continue

                responses.append(text_out)

            except Exception as e:
                logger.error(f"Gemini API Exception: {e}")
                raise e

        if n == 1:
            return responses[0]
        return responses

    # ---------------------------
    # Embedding model
    # ---------------------------
    def get_embedding_model(self):
        """
        Return Gemini embedding model.
        """
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except Exception as e:
            raise ImportError(
                "Missing dependency: pip install langchain-google-genai"
            ) from e

        if self.embedding_model is None:
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model="text-embedding-004"
            )
        return self.embedding_model
