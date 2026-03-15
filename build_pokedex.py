from pathlib import Path

from pokemon_rag.data_loader import (
    enrich_records_with_evolutions,
    enrich_records_with_type_matchups,
    fetch_pokedex_html,
    parse_pokedex_html,
    save_records,
)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    output_path = project_root / "data" / "pokedex.json"

    print("Downloading Pokemon data...")
    html = fetch_pokedex_html()

    print("Parsing records...")
    records = parse_pokedex_html(html)

    print("Enriching records with evolution chains...")
    records = enrich_records_with_evolutions(records)

    print("Enriching records with type strengths and weaknesses...")
    records = enrich_records_with_type_matchups(records)

    print(f"Saving {len(records)} records to {output_path}...")
    save_records(records, output_path)

    print("Done.")


if __name__ == "__main__":
    main()