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


POKEMON_THEME_CSS = """
body, .gradio-container {
    background: linear-gradient(180deg, #ffefef 0%, #f7f9fc 45%, #e8f1ff 100%);
    font-family: 'Segoe UI', Tahoma, sans-serif;
    color: #1f2937;
}

.gradio-container {
    max-width: 1100px !important;
}

.gradio-container, .gradio-container * {
    color: #1f2937;
}

#pokemon-shell {
    border: 4px solid #1f1f1f;
    border-radius: 24px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
    overflow: hidden;
    background: white;
}

#pokemon-header {
    background: linear-gradient(180deg, #ef5350 0%, #d32f2f 100%);
    color: white;
    padding: 18px 24px;
    border-bottom: 6px solid #1f1f1f;
}

#pokemon-header h1 {
    margin: 0;
    font-size: 2rem;
    display: flex;
    align-items: center;
    gap: 12px;
}

#pokemon-logo {
    width: 42px;
    height: 42px;
    flex: 0 0 auto;
}

#pokemon-header p {
    margin: 6px 0 0 0;
    font-size: 1rem;
    opacity: 0.95;
}

#pokemon-badge-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 12px;
}

.pokemon-badge {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 0.9rem;
}

#pokemon-body {
    padding: 16px;
    background:
        radial-gradient(circle at top center, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.98) 35%, rgba(248,250,255,1) 100%);
}

#pokemon-body, #pokemon-body * {
    color: #1f2937;
}

#pokemon-tip {
    background: #fff8e1;
    border: 2px solid #fbc02d;
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 14px;
    color: #4e342e;
}

.gr-button-primary {
    background: linear-gradient(180deg, #42a5f5 0%, #1e88e5 100%) !important;
    border: none !important;
}

.gr-button-primary:hover {
    filter: brightness(1.05);
}

textarea, input, .gr-textbox, .gr-textbox textarea, .gr-textbox input {
    background: #ffffff !important;
    color: #111827 !important;
}

.gr-chatbot, .gr-chatbot * {
    color: #111827 !important;
}

.gr-chatbot img {
    border-radius: 50%;
}

@media (prefers-color-scheme: dark) {
    body, .gradio-container {
        background: linear-gradient(180deg, #111827 0%, #0f172a 55%, #111827 100%);
        color: #f9fafb;
    }

    .gradio-container, .gradio-container * {
        color: #f9fafb;
    }

    #pokemon-shell {
        background: #111827;
        border-color: #e5e7eb;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
    }

    #pokemon-header {
        background: linear-gradient(180deg, #ef5350 0%, #b71c1c 100%);
        border-bottom-color: #e5e7eb;
        color: #ffffff;
    }

    #pokemon-header h1,
    #pokemon-header p,
    #pokemon-header .pokemon-badge {
        color: #ffffff !important;
    }

    #pokemon-body {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
        color: #f9fafb;
    }

    #pokemon-body, #pokemon-body * {
        color: #f9fafb;
    }

    #pokemon-tip {
        background: #3b2f0b;
        border-color: #fbc02d;
        color: #fef3c7;
    }

    textarea, input, .gr-textbox, .gr-textbox textarea, .gr-textbox input {
        background: #0f172a !important;
        color: #f9fafb !important;
        border-color: #475569 !important;
    }

    .gr-chatbot {
        background: #0f172a !important;
        border: 1px solid #334155 !important;
    }

    .gr-chatbot, .gr-chatbot * {
        color: #f9fafb !important;
    }
}

footer { visibility: hidden !important; }
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

    def respond(message: str, chat_history: list) -> tuple[list, str, str | None]:
        answer_text, image_url = bot.answer(message, chat_history)
        if chat_history is None:
            chat_history = []
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": answer_text})
        return chat_history, "", image_url

    with gr.Blocks(title="Pokemon RAG Chatbot") as demo:
        gr.HTML(f"<style>{POKEMON_THEME_CSS}</style>")
        gr.HTML(
            """
            <div id="pokemon-shell">
              <div id="pokemon-header">
                <h1>
                  <svg id="pokemon-logo" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <circle cx="50" cy="50" r="47" fill="#ffffff" stroke="#1f1f1f" stroke-width="6"/>
                    <path d="M3 50 A47 47 0 0 1 97 50 L3 50 Z" fill="#ef5350" stroke="#1f1f1f" stroke-width="0"/>
                    <line x1="3" y1="50" x2="97" y2="50" stroke="#1f1f1f" stroke-width="8"/>
                    <circle cx="50" cy="50" r="16" fill="#ffffff" stroke="#1f1f1f" stroke-width="6"/>
                    <circle cx="50" cy="50" r="6" fill="#90caf9" stroke="#1f1f1f" stroke-width="3"/>
                  </svg>
                  Pokémon Trainer Chat Lab
                </h1>
                <p>Ask about Pokémon stats, generations, regions, evolutions, type matchups, and moves.</p>
                <div id="pokemon-badge-row">
                  <span class="pokemon-badge">Local Ollama</span>
                  <span class="pokemon-badge">Pokédex RAG</span>
                  <span class="pokemon-badge">Gradio UI</span>
                </div>
              </div>
              <div id="pokemon-body">
            """
        )

        gr.Markdown(
            "**Pokédex Tips:** Try specific questions first for the best results, then use follow-ups like `What moves does that Pokémon have?`",
            elem_id="pokemon-tip",
        )

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500, label="Pokémon Chat", layout="bubble")
                msg = gr.Textbox(
                    label="Trainer Question",
                    placeholder="Ask your Pokédex question here... e.g. What does Eevee evolve into?",
                )
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear")

                gr.Examples(
                    examples=[
                        "What type is Pikachu?",
                        "What generation is Rowlet from?",
                        "What does Munchlax evolve into?",
                        "What is Charizard weak to?",
                        "What moves can Pikachu learn?",
                    ],
                    inputs=msg,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Top Retrieved Pokémon")
                pokemon_image = gr.Image(label="", show_label=False)

        submit.click(respond, [msg, chatbot], [chatbot, msg, pokemon_image])
        msg.submit(respond, [msg, chatbot], [chatbot, msg, pokemon_image])
        clear.click(lambda: ([], "", None), None, [chatbot, msg, pokemon_image])

        gr.HTML("</div></div>")

    demo.launch()


if __name__ == "__main__":
    main()