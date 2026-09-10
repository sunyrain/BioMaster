"""Cross-entity browsing of the loaded evidence snapshot, without inference."""
from __future__ import annotations
import json
from .explorer_data import clean


def browse(data, section, search="", category="all", page=1, page_size=20):
    sections = {"known_diseases", "known_targets", "pathways", "structures", "txgnn_diseases", "pairs_drug", "pairs_target"}
    if section not in sections:
        raise ValueError("Unknown browse section")
    if page < 1 or not 1 <= page_size <= 100:
        raise ValueError("Invalid pagination")
    categories = {"known_diseases": {"all", "approved", "clinical"}, "structures": {"all", "experimental", "predicted"}}
    if category not in categories.get(section, {"all"}):
        raise ValueError("Unknown category")
    data.ensure_loaded()
    kind = "target" if section in {"pathways", "structures", "pairs_target"} else "drug"
    entities = data.targets if kind == "target" else data.drugs
    needle = search.strip().casefold()
    matches = []
    all_total = 0
    covered = set()
    matched_entities = set()
    for entity in sorted(entities.values(), key=lambda e: (str(e["name"]).casefold(), e["id"])):
        if section == "structures":
            if not entity.get("structure"): continue
            pockets = entity.get("pockets", [])
            experimental = sum(p.get("type") == "experimental" for p in pockets)
            predicted = sum(p.get("type") == "predicted" for p in pockets)
            records = [dict(entity["structure"], name=entity["name"], pocket_count=len(pockets), experimental_pockets=experimental, predicted_pockets=predicted)]
        elif section.startswith("pairs_"):
            if not entity.get("scored"): continue
            groups = data._drug_groups if kind == "drug" else data._target_groups
            records = [{"name": entity["name"], "pair_count": len(groups.get(entity["id"], [])), "source": "当前模型评分快照"}]
        else:
            records = entity.get(section, [])
        for record in records:
            all_total += 1
            covered.add(entity["id"])
            if section == "known_diseases":
                phase = record.get("phase")
                if category == "approved" and phase != 4: continue
                if category == "clinical" and phase not in (1, 2, 3): continue
            if section == "structures" and category != "all" and not record.get(category + "_pockets"): continue
            if needle:
                haystack = f'{entity["name"]} {entity["id"]} {entity.get("identifiers", {})} {json.dumps(clean(record), ensure_ascii=False)}'
                if needle not in haystack.casefold(): continue
            matched_entities.add(entity["id"])
            matches.append({"entity": {"id": entity["id"], "kind": kind, "name": entity["name"]}, "record": record})
    return clean({"section": section, "page": page, "page_size": page_size, "total": len(matches), "all_total": all_total,
                  "entity_total": len(covered), "matched_entities": len(matched_entities),
                  "items": matches[(page - 1) * page_size:page * page_size]})
