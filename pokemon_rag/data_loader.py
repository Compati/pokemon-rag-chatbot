from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


POKEDEX_URL = "https://pokemondb.net/pokedex/all"
BASE_URL = "https://pokemondb.net"

GENERATION_RANGES = [
    (1, 151, "Generation I", "Kanto"),
    (152, 251, "Generation II", "Johto"),
    (252, 386, "Generation III", "Hoenn"),
    (387, 493, "Generation IV", "Sinnoh"),
    (494, 649, "Generation V", "Unova"),
    (650, 721, "Generation VI", "Kalos"),
    (722, 809, "Generation VII", "Alola"),
    (810, 905, "Generation VIII", "Galar / Hisui"),
    (906, 1025, "Generation IX", "Paldea"),
]

TYPE_EFFECTIVENESS: dict[str, dict[str, list[str]]] = {
    "Normal": {"strong_against": [], "weak_against": ["Rock", "Steel"], "no_effect": ["Ghost"], "weak_to": ["Fighting"], "resists": [], "immune_to": ["Ghost"]},
    "Fire": {"strong_against": ["Grass", "Ice", "Bug", "Steel"], "weak_against": ["Fire", "Water", "Rock", "Dragon"], "no_effect": [], "weak_to": ["Water", "Ground", "Rock"], "resists": ["Fire", "Grass", "Ice", "Bug", "Steel", "Fairy"], "immune_to": []},
    "Water": {"strong_against": ["Fire", "Ground", "Rock"], "weak_against": ["Water", "Grass", "Dragon"], "no_effect": [], "weak_to": ["Electric", "Grass"], "resists": ["Fire", "Water", "Ice", "Steel"], "immune_to": []},
    "Electric": {"strong_against": ["Water", "Flying"], "weak_against": ["Electric", "Grass", "Dragon"], "no_effect": ["Ground"], "weak_to": ["Ground"], "resists": ["Electric", "Flying", "Steel"], "immune_to": []},
    "Grass": {"strong_against": ["Water", "Ground", "Rock"], "weak_against": ["Fire", "Grass", "Poison", "Flying", "Bug", "Dragon", "Steel"], "no_effect": [], "weak_to": ["Fire", "Ice", "Poison", "Flying", "Bug"], "resists": ["Water", "Electric", "Grass", "Ground"], "immune_to": []},
    "Ice": {"strong_against": ["Grass", "Ground", "Flying", "Dragon"], "weak_against": ["Fire", "Water", "Ice", "Steel"], "no_effect": [], "weak_to": ["Fire", "Fighting", "Rock", "Steel"], "resists": ["Ice"], "immune_to": []},
    "Fighting": {"strong_against": ["Normal", "Ice", "Rock", "Dark", "Steel"], "weak_against": ["Poison", "Flying", "Psychic", "Bug", "Fairy"], "no_effect": ["Ghost"], "weak_to": ["Flying", "Psychic", "Fairy"], "resists": ["Bug", "Rock", "Dark"], "immune_to": []},
    "Poison": {"strong_against": ["Grass", "Fairy"], "weak_against": ["Poison", "Ground", "Rock", "Ghost"], "no_effect": ["Steel"], "weak_to": ["Ground", "Psychic"], "resists": ["Grass", "Fighting", "Poison", "Bug", "Fairy"], "immune_to": []},
    "Ground": {"strong_against": ["Fire", "Electric", "Poison", "Rock", "Steel"], "weak_against": ["Grass", "Bug"], "no_effect": ["Flying"], "weak_to": ["Water", "Grass", "Ice"], "resists": ["Poison", "Rock"], "immune_to": ["Electric"]},
    "Flying": {"strong_against": ["Grass", "Fighting", "Bug"], "weak_against": ["Electric", "Rock", "Steel"], "no_effect": [], "weak_to": ["Electric", "Ice", "Rock"], "resists": ["Grass", "Fighting", "Bug"], "immune_to": ["Ground"]},
    "Psychic": {"strong_against": ["Fighting", "Poison"], "weak_against": ["Psychic", "Steel"], "no_effect": ["Dark"], "weak_to": ["Bug", "Ghost", "Dark"], "resists": ["Fighting", "Psychic"], "immune_to": []},
    "Bug": {"strong_against": ["Grass", "Psychic", "Dark"], "weak_against": ["Fire", "Fighting", "Poison", "Flying", "Ghost", "Steel", "Fairy"], "no_effect": [], "weak_to": ["Fire", "Flying", "Rock"], "resists": ["Grass", "Fighting", "Ground"], "immune_to": []},
    "Rock": {"strong_against": ["Fire", "Ice", "Flying", "Bug"], "weak_against": ["Fighting", "Ground", "Steel"], "no_effect": [], "weak_to": ["Water", "Grass", "Fighting", "Ground", "Steel"], "resists": ["Normal", "Fire", "Poison", "Flying"], "immune_to": []},
    "Ghost": {"strong_against": ["Psychic", "Ghost"], "weak_against": ["Dark"], "no_effect": ["Normal"], "weak_to": ["Ghost", "Dark"], "resists": ["Poison", "Bug"], "immune_to": ["Normal", "Fighting"]},
    "Dragon": {"strong_against": ["Dragon"], "weak_against": ["Steel"], "no_effect": ["Fairy"], "weak_to": ["Ice", "Dragon", "Fairy"], "resists": ["Fire", "Water", "Electric", "Grass"], "immune_to": []},
    "Dark": {"strong_against": ["Psychic", "Ghost"], "weak_against": ["Fighting", "Dark", "Fairy"], "no_effect": [], "weak_to": ["Fighting", "Bug", "Fairy"], "resists": ["Ghost", "Dark"], "immune_to": ["Psychic"]},
    "Steel": {"strong_against": ["Ice", "Rock", "Fairy"], "weak_against": ["Fire", "Water", "Electric", "Steel"], "no_effect": [], "weak_to": ["Fire", "Fighting", "Ground"], "resists": ["Normal", "Grass", "Ice", "Flying", "Psychic", "Bug", "Rock", "Dragon", "Steel", "Fairy"], "immune_to": ["Poison"]},
    "Fairy": {"strong_against": ["Fighting", "Dragon", "Dark"], "weak_against": ["Fire", "Poison", "Steel"], "no_effect": [], "weak_to": ["Poison", "Steel"], "resists": ["Fighting", "Bug", "Dark"], "immune_to": ["Dragon"]},
}


def fetch_pokedex_html(url: str = POKEDEX_URL, timeout: int = 30) -> str:
    """Download the Pokemon Database National Dex page."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def _clean_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def _extract_primary_dex_number(dex_number: str) -> int | None:
    digits = "".join(ch for ch in dex_number if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def infer_generation_and_region(dex_number: str) -> dict[str, str]:
    """Infer generation and region from National Dex number ranges."""
    primary_number = _extract_primary_dex_number(dex_number)
    if primary_number is None:
        return {"generation": "Unknown", "region": "Unknown"}

    for start, end, generation, region in GENERATION_RANGES:
        if start <= primary_number <= end:
            return {"generation": generation, "region": region}

    return {"generation": "Unknown", "region": "Unknown"}


def _extract_evolution_chain(profile_html: str) -> list[str]:
    """Extract a simple evolution chain from a Pokémon profile page."""
    soup = BeautifulSoup(profile_html, "lxml")

    selectors = [
        ".infocard-list-evo .ent-name",
        ".infocard-list-evo a.ent-name",
    ]

    names: list[str] = []
    for selector in selectors:
        matches = soup.select(selector)
        if matches:
            for match in matches:
                name = _clean_text(match.get_text(" ", strip=True))
                if name and name not in names:
                    names.append(name)
            if names:
                break

    return names


def enrich_records_with_evolutions(
    records: list[dict[str, Any]], timeout: int = 30
) -> list[dict[str, Any]]:
    """Fetch each Pokémon page and enrich records with evolution-chain information."""
    session = requests.Session()
    evolution_cache: dict[str, list[str]] = {}

    for record in records:
        profile_url = record.get("profile_url")
        if not profile_url:
            record["evolution_chain"] = []
            record["evolves_from"] = None
            record["evolves_to"] = []
            continue

        if profile_url not in evolution_cache:
            try:
                response = session.get(profile_url, timeout=timeout)
                response.raise_for_status()
                evolution_cache[profile_url] = _extract_evolution_chain(response.text)
            except requests.RequestException:
                evolution_cache[profile_url] = []

        chain = evolution_cache[profile_url]
        record_name = record.get("name", "")

        evolves_from = None
        evolves_to: list[str] = []
        if record_name in chain:
            idx = chain.index(record_name)
            if idx > 0:
                evolves_from = chain[idx - 1]
            if idx < len(chain) - 1:
                evolves_to = chain[idx + 1 :]

        record["evolution_chain"] = chain
        record["evolves_from"] = evolves_from
        record["evolves_to"] = evolves_to

    return records


def enrich_records_with_type_matchups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add simple type strengths/weaknesses summaries based on each Pokémon's types."""
    for record in records:
        types = record.get("types", [])
        offense_strong: set[str] = set()
        offense_weak: set[str] = set()
        offense_no_effect: set[str] = set()
        defense_weak: set[str] = set()
        defense_resist: set[str] = set()
        defense_immune: set[str] = set()

        for pokemon_type in types:
            chart = TYPE_EFFECTIVENESS.get(pokemon_type, {})
            offense_strong.update(chart.get("strong_against", []))
            offense_weak.update(chart.get("weak_against", []))
            offense_no_effect.update(chart.get("no_effect", []))
            defense_weak.update(chart.get("weak_to", []))
            defense_resist.update(chart.get("resists", []))
            defense_immune.update(chart.get("immune_to", []))

        record["type_matchups"] = {
            "offense_strong_against": sorted(offense_strong),
            "offense_weak_against": sorted(offense_weak),
            "offense_no_effect_against": sorted(offense_no_effect),
            "defense_weak_to": sorted(defense_weak),
            "defense_resists": sorted(defense_resist),
            "defense_immune_to": sorted(defense_immune),
        }

    return records


def _extract_moves(profile_html: str, limit: int = 12) -> list[str]:
    """Extract a starter set of moves from a Pokémon profile page."""
    soup = BeautifulSoup(profile_html, "lxml")
    moves: list[str] = []

    selectors = [
        'a[href^="/move/"]',
        '.resp-scroll a[href^="/move/"]',
    ]

    for selector in selectors:
        for link in soup.select(selector):
            move_name = _clean_text(link.get_text(" ", strip=True))
            if move_name and move_name not in moves:
                moves.append(move_name)
            if len(moves) >= limit:
                return moves

    return moves


def enrich_records_with_moves(
    records: list[dict[str, Any]], timeout: int = 30, limit: int = 12
) -> list[dict[str, Any]]:
    """Fetch each Pokémon profile page and extract a starter list of moves."""
    session = requests.Session()
    move_cache: dict[str, list[str]] = {}

    for record in records:
        profile_url = record.get("profile_url")
        if not profile_url:
            record["moves"] = []
            continue

        if profile_url not in move_cache:
            try:
                response = session.get(profile_url, timeout=timeout)
                response.raise_for_status()
                move_cache[profile_url] = _extract_moves(response.text, limit=limit)
            except requests.RequestException:
                move_cache[profile_url] = []

        record["moves"] = move_cache[profile_url]

    return records


def parse_pokedex_html(html: str) -> list[dict[str, Any]]:
    """Parse the main Pokedex table into structured records."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table#pokedex")
    if table is None:
        raise ValueError("Could not find the pokedex table on the page.")

    records: list[dict[str, Any]] = []

    for row in table.select("tbody tr"):
        cells = row.find_all("td")
        if len(cells) < 8:
            continue

        dex_number = _clean_text(cells[0].get_text(" ", strip=True))
        name_cell = cells[1]
        name = _clean_text(name_cell.get_text(" ", strip=True))
        profile_link = name_cell.select_one("a.ent-name") or name_cell.select_one("a")
        profile_url = urljoin(BASE_URL, profile_link.get("href", "")) if profile_link else None
        type_values = [_clean_text(a.get_text(" ", strip=True)) for a in cells[2].select("a")]
        total = _clean_text(cells[3].get_text(" ", strip=True))
        hp = _clean_text(cells[4].get_text(" ", strip=True))
        attack = _clean_text(cells[5].get_text(" ", strip=True))
        defense = _clean_text(cells[6].get_text(" ", strip=True))
        sp_atk = _clean_text(cells[7].get_text(" ", strip=True)) if len(cells) > 7 else ""
        sp_def = _clean_text(cells[8].get_text(" ", strip=True)) if len(cells) > 8 else ""
        speed = _clean_text(cells[9].get_text(" ", strip=True)) if len(cells) > 9 else ""
        generation_region = infer_generation_and_region(dex_number)

        record = {
            "dex_number": dex_number,
            "name": name,
            "profile_url": profile_url,
            "types": type_values,
            "generation": generation_region["generation"],
            "region": generation_region["region"],
            "total": total,
            "hp": hp,
            "attack": attack,
            "defense": defense,
            "special_attack": sp_atk,
            "special_defense": sp_def,
            "speed": speed,
        }
        records.append(record)

    if not records:
        raise ValueError("No Pokemon records were parsed from the source page.")

    return records


def build_documents(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create retrieval documents from parsed Pokemon records."""
    documents: list[dict[str, Any]] = []
    for record in records:
        types_text = ", ".join(record.get("types", [])) or "Unknown"
        generation_text = record.get("generation", "Unknown")
        region_text = record.get("region", "Unknown")
        evolution_chain = record.get("evolution_chain", [])
        evolves_from = record.get("evolves_from") or "None"
        evolves_to = ", ".join(record.get("evolves_to", [])) or "None"
        evolution_text = " -> ".join(evolution_chain) if evolution_chain else "Unknown"
        type_matchups = record.get("type_matchups", {})
        offense_strong = ", ".join(type_matchups.get("offense_strong_against", [])) or "Unknown"
        offense_weak = ", ".join(type_matchups.get("offense_weak_against", [])) or "Unknown"
        offense_no_effect = ", ".join(type_matchups.get("offense_no_effect_against", [])) or "None"
        defense_weak = ", ".join(type_matchups.get("defense_weak_to", [])) or "Unknown"
        defense_resists = ", ".join(type_matchups.get("defense_resists", [])) or "Unknown"
        defense_immune = ", ".join(type_matchups.get("defense_immune_to", [])) or "None"
        moves_text = ", ".join(record.get("moves", [])) or "Unknown"
        text = (
            f"Pokemon #{record['dex_number']} is {record['name']}. "
            f"Type: {types_text}. "
            f"Generation: {generation_text}. "
            f"Region: {region_text}. "
            f"Evolution chain: {evolution_text}. "
            f"Evolves from: {evolves_from}. "
            f"Evolves to: {evolves_to}. "
            f"Offensive strengths: {offense_strong}. "
            f"Offensive weaknesses: {offense_weak}. "
            f"No effect against: {offense_no_effect}. "
            f"Defensive weaknesses: {defense_weak}. "
            f"Defensive resistances: {defense_resists}. "
            f"Defensive immunities: {defense_immune}. "
            f"Sample moves: {moves_text}. "
            f"Base stat total: {record['total']}. "
            f"HP: {record['hp']}. Attack: {record['attack']}. Defense: {record['defense']}. "
            f"Special Attack: {record['special_attack']}. Special Defense: {record['special_defense']}. "
            f"Speed: {record['speed']}."
        )
        documents.append({"id": record["dex_number"], "metadata": record, "text": text})
    return documents


def save_records(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_records(input_path: Path) -> list[dict[str, Any]]:
    return json.loads(input_path.read_text(encoding="utf-8"))