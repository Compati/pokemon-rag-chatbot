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

    def answer(self, user_message: str, history: list[list[str]] | None = None) -> tuple[str, str | None]:
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

        answer_text = self.llm_client.chat(messages)
        
        # Extract image_url from top result if available
        image_url = None
        if results and len(results) > 0:
            top_metadata = results[0].document.get("metadata", {})
            image_url = top_metadata.get("image_url")
        
        return answer_text, image_url


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

    with gr.Blocks() as demo:
        gr.Markdown("# Pokemon RAG Chatbot")
        gr.Markdown("Ask questions about Pokemon using locally retrieved Pokedex data and an Ollama model.")
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="Ask about Pokemon...",
                    show_label=False,
                )
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear")
                
                gr.Examples(
                    examples=[
                        "What type is Pikachu?",
                        "Compare Bulbasaur and Charmander.",
                        "Tell me about Gengar.",
                        "Which retrieved Pokemon has the highest total stats?",
                    ],
                    inputs=msg,
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Top Retrieved Pokemon")
                pokemon_image = gr.Image(
                    label="",
                    show_label=False,
                )
        
        def respond(message: str, chat_history: list) -> tuple[list, str, str | None]:
            answer_text, image_url = bot.answer(message, chat_history)
            if chat_history is None:
                chat_history = []
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": answer_text})
            return chat_history, "", image_url
        
        submit.click(respond, [msg, chatbot], [chatbot, msg, pokemon_image])
        msg.submit(respond, [msg, chatbot], [chatbot, msg, pokemon_image])
        clear.click(lambda: ([], None), None, [chatbot, pokemon_image])

    demo.launch()


if __name__ == "__main__":
    main()