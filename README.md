# Pokemon RAG Chatbot with Ollama

This starter project builds a local chatbot in Python using:

- **Ollama** for the LLM
- **RAG** for retrieval over Pokemon data
- **Pokemon Database** (`https://pokemondb.net/pokedex/all`) as the first knowledge source
- **Gradio** for a simple local chat UI

The design follows the chatbot ideas from **Class 7**, but swaps in a local Ollama-based workflow instead of hosted APIs.

## Project structure

```text
Midterm Project/
|- app.py
|- build_pokedex.py
|- requirements.txt
|- .gitignore
|- data/
|  |- .gitkeep
+- pokemon_rag/
   |- __init__.py
   |- data_loader.py
   |- retriever.py
   +- ollama_client.py
```

## What this starter app does

1. Downloads and parses the Pokemon table from Pokemon Database.
2. Infers each Pokemon's **generation** and **region** from its National Dex number.
3. Enriches each Pokemon with a simple **evolution chain** from its profile page.
4. Converts each Pokemon row into a text document.
5. Builds a lightweight **TF-IDF retriever**.
6. Sends the top retrieved Pokemon facts to **Ollama** as context.
7. Runs a local **Gradio** chatbot.

## Prerequisites

1. Install Ollama from `https://ollama.com/download`
2. Pull a model, for example:

```bash
ollama pull llama3.2
```

3. Create a virtual environment and install dependencies:

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## First run

Build the dataset:

```bash
py build_pokedex.py
```

This now rebuilds `data/pokedex.json` with:
- base stats
- Pokemon types
- inferred generations
- inferred home regions
- evolution-chain information

Run the app:

```bash
py app.py
```

## Next steps

- Inspect `data/pokedex.json`
- Test a few Pokemon questions
- Ask generation/region questions like `What generation is Pikachu from?`
- Ask region questions like `Which Pokemon in the dataset are from Kanto?`
- Ask evolution questions like `What does Munchlax evolve into?`
- Ask chain questions like `What is the evolution chain for Charmander?`
- Improve retrieval quality
- Add more Pokemon sources later