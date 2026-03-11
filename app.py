from __future__ import annotations

from pathlib import Path

import gradio as gr

from pokemon_rag.data_loader import build_documents, load_records
from pokemon_rag.ollama_client import OllamaChatClient
from pokemon_rag.retriever import PokemonRetriever, format_context


SYSTEM_PROMPT = """You are a helpful Pokemon assistant.
Use the retrieved context to answer the user's question.
If the answer is not supported by the context, say that you are not sure based on the current Pokemon dataset.
Be clear, concise, and accurate.
"""


class PokemonRAGChatbot:
    def __init__(self, retriever: PokemonRetriever, llm_client: OllamaChatClient):
        self.retriever = retriever
        self.llm_client = llm_client

    def _normalize_history(self, history: list | None) -> list[tuple[str, str]]:
        """Normalize Gradio chat history into (user, assistant) tuples."""
        history = history or []
        normalized: list[tuple[str, str]] = []

        for item in history:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                user_text = str(item[0]) if item[0] is not None else ""
                assistant_text = str(item[1]) if item[1] is not None else ""
                normalized.append((user_text, assistant_text))
            elif isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if role == "user":
                    normalized.append((str(content), ""))
                elif role == "assistant" and normalized:
                    last_user, _ = normalized[-1]
                    normalized[-1] = (last_user, str(content))

        return normalized

    def answer(self, user_message: str, history: list[list[str]] | None = None) -> str:
        normalized_history = self._normalize_history(history)
        results = self.retriever.search(user_message, top_k=5)
        context = format_context(results)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for user_text, assistant_text in normalized_history[-3:]:
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

        prompt = (
            f"Retrieved Pokemon context:\n{context}\n\n"
            f"User question: {user_message}\n\n"
            "Answer using the context above. If the context is weak or missing, say so clearly."
        )
        messages.append({"role": "user", "content": prompt})

        return self.llm_client.chat(messages)


def build_bot(model_name: str = "llama3.2") -> PokemonRAGChatbot:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "pokedex.json"
    if not data_path.exists():
        raise FileNotFoundError("Missing data/pokedex.json. Run `py build_pokedex.py` first.")

    records = load_records(data_path)
    documents = build_documents(records)
    retriever = PokemonRetriever(documents)
    llm_client = OllamaChatClient(model=model_name)
    return PokemonRAGChatbot(retriever=retriever, llm_client=llm_client)


def main() -> None:
    bot = build_bot()

    def chat_fn(message: str, history: list[list[str]]) -> str:
        return bot.answer(message, history)

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="Pokemon RAG Chatbot",
        description="Ask questions about Pokemon using locally retrieved Pokedex data and an Ollama model.",
        examples=[
            "What type is Pikachu?",
            "Compare Bulbasaur and Charmander.",
            "Tell me about Gengar.",
            "Which retrieved Pokemon has the highest total stats?",
        ],
    )

    demo.launch()


if __name__ == "__main__":
    main()