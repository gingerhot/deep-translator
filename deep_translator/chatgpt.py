__copyright__ = "Copyright (C) 2020 Nidhal Baccouri"

from typing import List, Optional

from deep_translator.base import BaseTranslator
from deep_translator.constants import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)
from deep_translator.exceptions import ApiKeyException
from dotenv import dotenv_values


class ChatGptTranslator(BaseTranslator):
    """
    class that wraps functions, which use the ChatGPT
    under the hood to translate word(s)
    """
    ENV_VARS = dotenv_values(".env")

    def __init__(
        self,
        source: str = "auto",
        target: str = "english",
        api_key: Optional[str] = ENV_VARS.get(OPENAI_API_KEY, None),
        base_url: Optional[str] = ENV_VARS.get(OPENAI_BASE_URL, None),
        model: Optional[str] = ENV_VARS.get(OPENAI_MODEL, "gpt-3.5-turbo"),
        **kwargs,
    ):
        """
        @param api_key: your openai api key.
        @param base_url: your openai base URL.
        @param model: which model you choose.
        @param source: source language
        @param target: target language
        """
        if not api_key:
            raise ApiKeyException(env_var=OPENAI_API_KEY)

        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        super().__init__(source=source, target=target, **kwargs)

    def translate(self, text: str, **kwargs) -> str:
        """
        @param text: text to translate
        @return: translated text
        """
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url if self.base_url else None
        )

        prompt = "Translate below into {}, only return translated result:\n{}".format(self.target, text)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.choices[0].message.content

    def translate_file(self, path: str, **kwargs) -> str:
        return self._translate_file(path, **kwargs)

    def translate_batch(self, batch: List[str], **kwargs) -> List[str]:
        """
        @param batch: list of texts to translate
        @return: list of translations
        """
        return self._translate_batch(batch, **kwargs)
