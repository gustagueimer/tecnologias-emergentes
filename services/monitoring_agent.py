import json
from typing import Any

from services.ollama_client import OllamaClient


class MonitoringAgent:
    def __init__(
        self,
        ollama_client: OllamaClient,
        name: str = "Agente AgroVision",
        role: str = "triagem operacional de eventos",
        objective: str = "analisar deteccoes recentes, explicar riscos e sugerir a proxima acao",
    ):
        self.ollama_client = ollama_client
        self.name = name
        self.role = role
        self.objective = objective

    def build_system_prompt(self) -> str:
        return (
            f"Voce e {self.name}. Papel: {self.role}. Objetivo: {self.objective}. "
            "Responda em portugues do Brasil com tres secoes: Leitura, Risco e Recomendacao."
        )

    def build_context(self, events: list[dict[str, Any]], user_message: str) -> str:
        labels = [event["label"] for event in events]
        compact_events = [
            {
                "event_time": event["event_time"],
                "label": event["label"],
                "confidence": round(float(event["confidence"]), 2),
            }
            for event in events
        ]

        return (
            f"Mensagem do operador: {user_message}\n"
            f"Sequencia de labels detectados: {labels}\n"
            f"Eventos recentes: {json.dumps(compact_events, ensure_ascii=True)}"
        )

    def analyze(self, events: list[dict[str, Any]], user_message: str) -> str:
        if not events:
            return "Leitura: nao ha eventos recentes. Risco: baixo no momento. Recomendacao: manter monitoramento continuo."

        prompt = self.build_context(events, user_message)
        try:
            return self.ollama_client.chat(prompt=prompt, system_prompt=self.build_system_prompt())
        except Exception as exc:
            return (
                "Leitura: ha eventos recentes no monitoramento. "
                "Risco: variacao operacional sem consolidacao por falha de IA textual. "
                "Recomendacao: acompanhar por mais alguns minutos e validar infraestrutura do Ollama. "
                f"Detalhe: {exc}"
            )

    def analyze_stream(self, events: list[dict[str, Any]], user_message: str):
        prompt = self.build_context(events, user_message)
        try:
            yield from self.ollama_client.stream_chat(prompt=prompt, system_prompt=self.build_system_prompt())
        except Exception as exc:
            yield f"Falha ao gerar streaming da analise: {exc}"