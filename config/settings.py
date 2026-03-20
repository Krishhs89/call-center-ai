"""
Configuration and environment settings for Call Center AI Assistant.
Loads API keys and model preferences from environment variables.
"""

import os
from typing import Literal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)


class Settings:
    """Application settings loaded from environment variables.

    All API key properties re-read from .env on every access so that
    key rotations take effect without restarting the Streamlit server.
    """

    @property
    def ANTHROPIC_API_KEY(self) -> str:
        load_dotenv(override=True)
        return os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def OPENAI_API_KEY(self) -> str:
        load_dotenv(override=True)
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def GOOGLE_API_KEY(self) -> str:
        load_dotenv(override=True)
        return os.getenv("GOOGLE_API_KEY", "")

    @property
    def DEFAULT_LLM(self) -> str:
        return os.getenv("DEFAULT_LLM", "claude")

    @property
    def CLAUDE_MODEL(self) -> str:
        return os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

    @property
    def GPT4_MODEL(self) -> str:
        return os.getenv("GPT4_MODEL", "gpt-4-turbo")

    @property
    def GEMINI_MODEL(self) -> str:
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    @property
    def WHISPER_MODEL(self) -> str:
        return os.getenv("WHISPER_MODEL", "whisper-1")

    @property
    def MOCK_LLM(self) -> bool:
        load_dotenv(override=True)
        return os.getenv("MOCK_LLM", "false").lower() == "true"

    @property
    def DEBUG(self) -> bool:
        return os.getenv("DEBUG", "false").lower() == "true"

    @property
    def LOG_LEVEL(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")

    @property
    def MAX_FILE_SIZE_MB(self) -> int:
        return int(os.getenv("MAX_FILE_SIZE_MB", "100"))

    @property
    def LANGCHAIN_TRACING_V2(self) -> bool:
        return os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

    @property
    def LANGCHAIN_API_KEY(self) -> str:
        return os.getenv("LANGCHAIN_API_KEY", "")

    @property
    def LANGCHAIN_PROJECT(self) -> str:
        return os.getenv("LANGCHAIN_PROJECT", "call-center-ai")

    @classmethod
    def validate(cls) -> bool:
        load_dotenv(override=True)
        has_api_key = bool(
            os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not has_api_key:
            raise ValueError(
                "No API keys found. Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY in .env"
            )
        return has_api_key

    @classmethod
    def get_llm_config(cls, llm_name: Literal["claude", "gpt4", "gemini"]) -> dict:
        load_dotenv(override=True)
        if llm_name == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            return {"model": os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"), "api_key": api_key, "temperature": 0.7}
        elif llm_name == "gpt4":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not configured")
            return {"model": os.getenv("GPT4_MODEL", "gpt-4-turbo"), "api_key": api_key, "temperature": 0.7}
        elif llm_name == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY", "")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not configured")
            return {"model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), "api_key": api_key, "temperature": 0.7}
        else:
            raise ValueError(f"Unsupported LLM: {llm_name}")


# Create a singleton instance
settings = Settings()
