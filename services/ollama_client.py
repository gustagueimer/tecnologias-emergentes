import json
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaClient:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name

    def warmup(self) -> None:
        self.chat("Responda apenas: ok")

    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": [],
        }

        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        payload["messages"].append({"role": "user", "content": prompt})

        data = self._post(payload)
        return data.get("message", {}).get("content", "").strip()

    def stream_chat(self, prompt: str, system_prompt: str | None = None) -> Iterable[str]:
        payload = {
            "model": self.model_name,
            "stream": True,
            "messages": [],
        }

        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        payload["messages"].append({"role": "user", "content": prompt})

        request = Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=60) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    text = chunk.get("message", {}).get("content", "")
                    if text:
                        yield text
                    if chunk.get("done"):
                        break
        except HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {err.code} no Ollama: {detail}") from err
        except URLError as err:
            raise RuntimeError(f"Nao foi possivel conectar ao Ollama em {self.base_url}") from err

    def _post(self, payload: dict) -> dict:
        request = Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw)
        except HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {err.code} no Ollama: {detail}") from err
        except URLError as err:
            raise RuntimeError(f"Nao foi possivel conectar ao Ollama em {self.base_url}") from err