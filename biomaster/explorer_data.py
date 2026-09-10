"""Read-only project explorer, using complete scored catalogs and explicit provenance.

The selected release is scored once in FP32 and cached with a bundle fingerprint.
Ranks are calculated before search/pagination; missing scores never become zeros.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
BUNDLE = "outputs/biomaster_best_model_20260906/retargetmap_selected_v1"
PAIRS = "outputs/unified_pair_program_720x384_v1/UNIFIED_DTA_720_X_384_PAIR_MATRIX_V1.csv.gz"
FROZEN = "outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz"
DTIAM = "outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_720x384_deployment_v1/DTIAM_720X384_SCORES_V1.csv.gz"
TARGETS = "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
MODELS = ("biomaster", "drugclip", "dtiam", "conplex")
MODEL_INFO = {
    "biomaster": {"name": "ReTargetMap", "version": "Selected · 2026-09-06", "role": "DrugCLIP + Morgan + ESM2; FP32 双方向网络", "score_type": "logit", "source": BUNDLE, "note": "≤2025部署拟合；分数不是概率或实验亲和力。反向使用独立方向输出，仅作辅助证据。"},
    "drugclip": {"name": "DrugCLIP", "version": "统一 720 × 384 比较核心", "role": "结构表征比较证据", "score_type": "cosine similarity", "source": PAIRS, "note": "382个靶点可评分；实验与预测口袋之间未校准，跨来源总排序仅用于探索。"},
    "dtiam": {"name": "DTIAM", "version": "Public retrained deployment v1", "role": "独立比较模型", "score_type": "model probability", "source": DTIAM, "note": "模型输出并非实验亲和力。"},
    "conplex": {"name": "ConPLex", "version": "统一 720 × 384 比较核心", "role": "序列比较证据", "score_type": "model score", "source": PAIRS, "note": "模型输出并非实验亲和力。"},
}


def clean(value: Any) -> Any:
    """Produce strict JSON values, preserving absent values as null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _present(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in ("", "nan", "none", "<na>")


def _bool(value: Any) -> bool:
    return str(value).lower() in ("true", "1", "1.0")


def _read(root: Path, relative: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = root / relative
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, usecols=(lambda c: c in columns) if columns else None, low_memory=False)


def _source(root: Path, name: str, relative: str) -> dict:
    path = root / relative
    return {"name": name, "path": relative, "available": path.is_file() or path.is_dir(),
            "size_bytes": path.stat().st_size if path.is_file() else None}


def _entity(kind: str, identifier: str, name: str, **kwargs: Any) -> dict:
    return dict(id=identifier, kind=kind, name=name, subtitle="", identifiers={},
                description=None, properties={}, known_targets=[], known_diseases=[],
                txgnn_diseases=[], pathways=[], target_diseases=[], pockets=[],
                structure=None, sources=[], missing=[], scored=False, **kwargs)


class ExplorerData:
    """Immutable, shared snapshot; no large-file IO happens in request handlers."""

    def __init__(self, root: str | Path, *, infer: bool = True):
        self.root = Path(root).resolve()
        self.infer = infer
        self.drugs: dict[str, dict] = {}
        self.targets: dict[str, dict] = {}
        self.sources: list[dict] = []
        self.warnings: list[str] = []
        self.structure_files: dict[tuple[str, str], Path] = {}
        self._lock = threading.RLock()
        self._ready = False
        self._svg_cache: dict[str, str | None] = {}

    def ensure_loaded(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self._load()
            self._ready = True

    def _load(self) -> None:
        columns = ["ligand_inchikey", "drug_names", "ligand_smiles", "project_entity_ids",
                   "target_chembl_id", "gene_symbol", "conplex_score", "drugclip_cosine_mean",
                   "drugclip_sixfold_std", "drugclip_evidence_scope", "pair_pocket_evidence_source",
                   "pocket_evidence_tier", "is_any_frozen_known_relationship", "pair_novelty_class_384",
                   "chembl37_action_types", "chembl37_mechanism_of_action", "assay_lane",
                   "is_chembl37_mechanism_relationship", "cross_pocket_source_drugclip_rank_policy"]
        pairs = _read(self.root, PAIRS, columns)
        frozen = _read(self.root, FROZEN, ["ligand_inchikey", "drug_names", "target_chembl_id", "gene_symbol",
                      "biomaster_independent_borda_score", "dtiam_probability"])
        keys = ["ligand_inchikey", "target_chembl_id"]
        if pairs.empty and not frozen.empty:
            pairs = frozen.drop(columns=["biomaster_independent_borda_score", "dtiam_probability"], errors="ignore").copy()
        if pairs.empty:
            pairs = pd.DataFrame(columns=keys + ["drug_names", "gene_symbol"])
            self.warnings.append("未找到本地完整评分表；请挂载项目 outputs 数据目录。")
        if pairs.duplicated(keys).any():
            raise ValueError("Duplicate drug-target pairs in the complete catalog")
        pairs = pairs.sort_values(keys, kind="stable").reset_index(drop=True)
        self.sources.extend([_source(self.root, "统一比较模型与已知关系", PAIRS),
                             _source(self.root, "2026-09-01 冻结合同 Borda", FROZEN),
                             _source(self.root, "2026-09-06 选定独立模型", BUNDLE)])
        for identifier, group in pairs.groupby("ligand_inchikey", sort=False):
            row = group.iloc[0]
            name = str(row.get("drug_names", identifier))
            item = _entity("drug", str(identifier), name)
            item["identifiers"] = {"inchikey": identifier, "project_ids": clean(row.get("project_entity_ids"))}
            item["properties"] = {"smiles": clean(row.get("ligand_smiles"))}
            item["subtitle"] = str(identifier)
            item["scored"] = True
            item["structure_url"] = f"/api/molecule/{identifier}.svg" if _present(row.get("ligand_smiles")) else None
            self.drugs[str(identifier)] = item
        target_frame = _read(self.root, TARGETS)
        if not target_frame.empty:
            for row in target_frame.to_dict("records"):
                identifier = str(row["target_chembl_id"])
                name = str(row.get("gene_symbol") or identifier)
                item = _entity("target", identifier, name)
                item["subtitle"] = clean(row.get("target_name"))
                item["description"] = clean(row.get("protein_description"))
                item["identifiers"] = {"chembl_id": identifier, "uniprot_id": clean(row.get("uniprot_accession")), "ensembl_id": clean(row.get("ot_id"))}
                item["properties"] = {"family": clean(row.get("target_class_l1")), "subfamily": clean(row.get("target_class_leaf")),
                     "sequence_length": clean(row.get("sequence_length")), "organism": clean(row.get("organism")),
                     "assay_lane": clean(row.get("assay_lane")), "non_gpcr": _bool(row.get("set_non_gpcr_all")), "af_mean_plddt": clean(row.get("af_mean_plddt")),
                     "pocket_mean_plddt": clean(row.get("pocket_mean_plddt")), "sequence": clean(row.get("sequence"))}
                self.targets[identifier] = item
        self.sources.append(_source(self.root, "ChEMBL 37 / OpenTargets 靶点登记", TARGETS))
        for identifier, group in pairs.groupby("target_chembl_id", sort=False):
            if identifier not in self.targets:
                item = _entity("target", str(identifier), str(group.iloc[0].get("gene_symbol", identifier)))
                item["identifiers"] = {"chembl_id": identifier, "uniprot_id": None, "ensembl_id": None}
                self.targets[str(identifier)] = item
            self.targets[str(identifier)]["scored"] = True
        # Use packaged identifiers to recover mappings even if the full registry is absent.
        bundle_targets = _read(self.root, BUNDLE + "/targets.csv.gz")
        for row in bundle_targets.to_dict("records"):
            if str(row.get("chembl_id")) in self.targets:
                self.targets[str(row["chembl_id"])]["identifiers"]["uniprot_id"] = str(row["target_id"])
        if not frozen.empty:
            if frozen.duplicated(keys).any():
                raise ValueError("Duplicate keys in frozen score table")
            pairs = pairs.merge(frozen[keys + [c for c in ["biomaster_independent_borda_score", "dtiam_probability"] if c in frozen]], on=keys, how="left", validate="one_to_one")
        dtiam = _read(self.root, DTIAM, keys + ["dtiam_probability"])
        if not dtiam.empty:
            pairs = pairs.drop(columns=["dtiam_probability"], errors="ignore").merge(dtiam, on=keys, how="left", validate="one_to_one")
        self.sources.append(_source(self.root, "DTIAM 全目录评分", DTIAM))
        pairs["drugclip"] = pd.to_numeric(pairs.get("drugclip_cosine_mean", np.nan), errors="coerce")
        pairs["conplex"] = pd.to_numeric(pairs.get("conplex_score", np.nan), errors="coerce")
        pairs["dtiam"] = pd.to_numeric(pairs.get("dtiam_probability", np.nan), errors="coerce")
        pairs["frozen"] = pd.to_numeric(pairs.get("biomaster_independent_borda_score", np.nan), errors="coerce")
        pairs["biomaster"] = np.nan
        pairs["biomaster_reverse"] = np.nan
        if len(pairs) and (self.root / BUNDLE / "MANIFEST.json").is_file():
            try:
                selected = self._selected_scores(pairs)
                if selected is not None:
                    pairs["biomaster"] = selected[:, 0]
                    pairs["biomaster_reverse"] = selected[:, 1]
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                LOG.exception("Selected model scores are unavailable")
                self.warnings.append(f"选定模型分数暂不可用：{type(exc).__name__}: {exc}")
        else:
            self.warnings.append("未找到2026-09-06模型包；默认模型分数保留为空，冻结Borda单列展示。")
        for model in MODELS + ("frozen",):
            for kind, group_key in (("drug", keys[0]), ("target", keys[1])):
                column = "biomaster_reverse" if model == "biomaster" and kind == "target" else model
                pairs[f"{kind}_{model}_rank"] = pairs.groupby(group_key)[column].rank(ascending=False, method="first", na_option="keep")
                pairs[f"{kind}_{model}_denominator"] = pairs.groupby(group_key)[column].transform("count")
        self.pairs = pairs
        self._drug_groups = pairs.groupby(keys[0]).indices
        self._target_groups = pairs.groupby(keys[1]).indices
        self._load_evidence()
        target_aliases = {str(value): key for key, entity in self.targets.items()
                          for value in [key, entity["identifiers"].get("uniprot_id")]
                          if _present(value)}
        for entity in self.drugs.values():
            normalized = {}
            for relation in entity["known_targets"]:
                relation = dict(relation)
                original_id = str(relation.get("id", ""))
                target_id = target_aliases.get(original_id)
                relation["explorable"] = target_id is not None
                relation["in_catalog"] = target_id is not None
                if target_id:
                    relation["source_id"] = original_id
                    relation["id"] = target_id
                    relation["identifiers"] = self.targets[target_id]["identifiers"]
                if relation["id"] in normalized:
                    normalized[relation["id"]].update({key: value for key, value in relation.items() if value is not None})
                else:
                    normalized[relation["id"]] = relation
            entity["known_targets"] = list(normalized.values())
        for kind, entities in (("drug", self.drugs), ("target", self.targets)):
            for identifier, entity in entities.items():
                entity["models"] = self._model_coverage(kind, identifier)
                fields = ("known_diseases", "txgnn_diseases", "known_targets") if kind == "drug" else ("pathways", "target_diseases", "pockets", "structure")
                entity["missing"] = [field for field in fields if not entity.get(field)]
        self._aliases = {}
        for kind, entities in (("drug", self.drugs), ("target", self.targets)):
            aliases: dict[str, list[str]] = {}
            for identifier, entity in entities.items():
                for value in [identifier, entity["name"], *entity["identifiers"].values()]:
                    if isinstance(value, str) and _present(value):
                        aliases.setdefault(value.lower(), []).append(identifier)
            self._aliases[kind] = {key: sorted(set(ids)) for key, ids in aliases.items()}
        LOG.info("Explorer loaded %d drugs, %d registered targets, %d pairs", len(self.drugs), len(self.targets), len(pairs))

    def _selected_scores(self, pairs: pd.DataFrame) -> np.ndarray | None:
        bundle = self.root / BUNDLE
        manifest = (bundle / "MANIFEST.json").read_bytes()
        # A cache may only describe an intact release, even if files changed in place.
        for name, record in json.loads(manifest).get("files", {}).items():
            path = (bundle / name).resolve()
            if not path.is_relative_to(bundle.resolve()):
                raise ValueError("Invalid path in selected release manifest")
            if not path.is_file() and record.get("optional_geometry"):
                continue
            if not path.is_file():
                raise ValueError(f"Missing selected release file: {name}")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b""):
                    digest.update(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Selected release integrity check failed: {name}")
        key_bytes = "\n".join(pairs.ligand_inchikey + "__" + pairs.target_chembl_id).encode()
        fingerprint = hashlib.sha256(b"explorer-fp32-v1\0" + manifest + key_bytes).hexdigest()
        cache = self.root / "outputs/biomaster_explorer/cache" / f"selected_{fingerprint}.npz"
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as data:
                scores = data["scores"]
            if scores.shape == (len(pairs), 2) and np.isfinite(scores).all():
                self.selected_provenance = {"fingerprint": fingerprint, "precision": "FP32", "device": "cpu", "cache": str(cache.relative_to(self.root))}
                return scores
        if not self.infer:
            self.warnings.append("选定模型缓存不存在，--no-infer 已启用；默认模型分数为空。")
            return None
        LOG.info("Computing selected model FP32 catalog scores once (%d pairs)", len(pairs))
        import torch
        # The release package is imported under its own name, avoiding research-code drift.
        package_name = "_biomaster_explorer_selected"
        spec = importlib.util.spec_from_file_location(package_name, bundle / "retargetmap/__init__.py", submodule_search_locations=[str(bundle / "retargetmap")])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        ranker = module.CatalogRanker(bundle, device="cpu", verify=True)
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(min(4, old_threads))
            ids = [self.targets[str(v)]["identifiers"]["uniprot_id"] for v in pairs.target_chembl_id]
            frame = ranker.score_pairs(pairs.ligand_inchikey.astype(str).tolist(), ids, batch_size=1024)
            scores = frame[["drug_to_target_score", "target_to_drug_score"]].to_numpy(dtype=np.float32)
        finally:
            torch.set_num_threads(old_threads)
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, scores=scores)
        temporary.replace(cache)
        self.selected_provenance = {"fingerprint": fingerprint, "precision": "FP32", "device": "cpu", "cache": str(cache.relative_to(self.root))}
        return scores

    def _load_evidence(self) -> None:
        pairs = self.pairs
        from .explorer_disease_reviews import load_disease_reviews
        disease_reviews = load_disease_reviews(self.root, self.drugs, self.targets)
        for kind, entities in (("drugs", self.drugs), ("targets", self.targets)):
            for identifier, values in disease_reviews[kind].items():
                entities[identifier].update(values)
        self.sources.extend(disease_reviews["sources"])
        if "is_any_frozen_known_relationship" in pairs:
            for row in pairs[pairs.is_any_frozen_known_relationship.map(_bool)].to_dict("records"):
                drug = self.drugs[str(row["ligand_inchikey"])]
                drug["known_targets"].append({"id": str(row["target_chembl_id"]), "name": str(row["gene_symbol"]),
                    "source": "ChEMBL 37 / 项目冻结已知关系", "action": clean(row.get("chembl37_action_types")),
                    "evidence": clean(row.get("chembl37_mechanism_of_action")), "type": "known_relation"})
        try:
            from .explorer_annotations import load_annotations
            annotations = load_annotations(self.root, self.drugs, self.targets)
            for key, entities in (("drugs", self.drugs), ("targets", self.targets)):
                for identifier, values in annotations.get(key, {}).items():
                    if identifier not in entities:
                        continue
                    for field, value in values.items():
                        if isinstance(value, dict) and isinstance(entities[identifier].get(field), dict):
                            entities[identifier][field].update(value)
                        elif field == "known_targets":
                            existing = entities[identifier][field]
                            known_rows = {row["id"]: row for row in existing}
                            for row in value:
                                if row["id"] in known_rows:
                                    known_rows[row["id"]].update({key: val for key, val in row.items() if val is not None})
                                else:
                                    existing.append(row)
                                    known_rows[row["id"]] = row
                        else:
                            entities[identifier][field] = value
            self.sources.extend(annotations.get("sources", []))
            self.annotation_counts = annotations.get("counts", {})
        except ImportError:
            self.warnings.append("疾病与通路适配器不可用。")
        try:
            from .explorer_structures import load_structures
            structures = load_structures(self.root, self.targets)
            for identifier, values in structures.get("targets", {}).items():
                if identifier in self.targets:
                    self.targets[identifier].update(values)
            self.structure_files = structures.get("files", {})
            self.sources.extend(structures.get("sources", []))
        except ImportError:
            self.warnings.append("结构适配器不可用。")
        try:
            from .explorer_experiments import load_experiments
            experiments = load_experiments(self.root, self.drugs, self.targets)
            for key, entities in (("drugs", self.drugs), ("targets", self.targets)):
                for identifier, values in experiments.get(key, {}).items():
                    if identifier in entities:
                        entities[identifier].update(values)
            self.sources.extend(experiments.get("sources", []))
            self.experiment_counts = experiments.get("counts", {})
            self.experiment_summary = experiments.get("summary", {})
        except ImportError:
            self.warnings.append("实验设计适配器不可用。")

    def resolve(self, kind: str, identifier: str) -> str:
        self.ensure_loaded()
        if kind not in ("drug", "target"):
            raise ValueError("kind must be drug or target")
        entities = self.drugs if kind == "drug" else self.targets
        if identifier in entities:
            return identifier
        ids = self._aliases[kind].get(identifier.lower(), [])
        if len(ids) == 1:
            return ids[0]
        raise KeyError(f"Unknown or ambiguous {kind}: {identifier}")

    def _rows(self, kind: str, identifier: str) -> pd.DataFrame:
        groups = self._drug_groups if kind == "drug" else self._target_groups
        indices = groups.get(identifier, [])
        return self.pairs.iloc[indices]

    def _model_coverage(self, kind: str, identifier: str) -> list[dict]:
        rows = self._rows(kind, identifier)
        return [dict(MODEL_INFO[m], id=m, coverage=int(rows[f"{kind}_{m}_denominator"].iloc[0]) if len(rows) else 0,
                     available=bool(len(rows) and rows[f"{kind}_{m}_denominator"].iloc[0] > 0)) for m in MODELS]

    @staticmethod
    def brief(entity: dict) -> dict:
        result = {key: entity.get(key) for key in ("id", "kind", "name", "subtitle", "identifiers", "scored")}
        result["annotations"] = {key: len(entity.get(key, [])) for key in (
            "known_targets", "known_diseases", "txgnn_diseases", "pathways", "target_diseases", "pockets", "experiments")}
        result["classification"] = entity.get("properties", {}).get("family" if entity["kind"] == "target" else "molecular_formula")
        return result

    def search(self, q: str = "", kind: str = "all", limit: int = 20) -> dict:
        self.ensure_loaded()
        if kind not in ("all", "drug", "target"):
            raise ValueError("kind must be all, drug or target")
        limit = max(1, min(2000, int(limit)))
        query = q.strip().lower()
        entities = ([] if kind == "target" else list(self.drugs.values())) + ([] if kind == "drug" else list(self.targets.values()))
        found = []
        for item in entities:
            haystack = " ".join([item["id"], str(item["name"]), str(item.get("subtitle", "")), str(item["identifiers"])]).lower()
            if query in haystack:
                priority = (0 if query and (query == item["name"].lower() or query == item["id"].lower()) else 1,
                            0 if item["name"].lower().startswith(query) else 1, 0 if item["scored"] else 1, item["name"].lower())
                found.append((priority, item))
        found.sort(key=lambda entry: entry[0])
        return clean({"items": [self.brief(item) for _, item in found[:limit]], "total": len(found), "query": q})

    def entity(self, kind: str, identifier: str) -> dict:
        identifier = self.resolve(kind, identifier)
        original = (self.drugs if kind == "drug" else self.targets)[identifier]
        result = {key: copy.deepcopy(value[:100] if isinstance(value, list) else value) for key, value in original.items()}
        for key, value in original.items():
            if isinstance(value, list):
                result[f"{key}_total"] = len(value)
        result["rank_scope"] = "逐药物靶点检索" if kind == "drug" else "逐靶点老药检索 · 辅助证据"
        result["sources"] = self.sources
        return clean(result)

    def evidence(self, kind: str, identifier: str, section: str, page: int = 1, page_size: int = 50, search: str = "") -> dict:
        identifier = self.resolve(kind, identifier)
        if section not in ("known_targets", "known_diseases", "txgnn_diseases", "pathways", "target_diseases", "pockets", "experiments"):
            raise ValueError("Unsupported evidence section")
        if page < 1 or not 1 <= page_size <= 200:
            raise ValueError("page must be positive; page_size must be between 1 and 200")
        original = (self.drugs if kind == "drug" else self.targets)[identifier].get(section, [])
        if section == "experiments":
            from .explorer_spr_reviews import live_spr_items
            original = live_spr_items(self.root, original)
        rows = original
        if search.strip():
            needle = search.strip().lower()
            rows = [row for row in rows if needle in json.dumps(clean(row), ensure_ascii=False).lower()]
        return clean({"kind": kind, "id": identifier, "section": section, "page": page, "page_size": page_size,
                      "total": len(rows), "all_total": len(original), "items": rows[(page - 1) * page_size:page * page_size]})

    def rankings(self, kind: str, identifier: str, model: str = "biomaster", search: str = "", page: int = 1, page_size: int = 20, *, export: bool = False, relationship: str = "all", order: str = "asc") -> dict:
        identifier = self.resolve(kind, identifier)
        if model not in MODELS:
            raise ValueError("Unknown model")
        if order not in ("asc", "desc"):
            raise ValueError("Unknown sort order")
        if relationship not in ("all", "known", "unannotated"):
            raise ValueError("Unknown relationship filter")
        if page < 1 or not 1 <= page_size <= 200:
            raise ValueError("page must be positive; page_size must be between 1 and 200")
        rows = self._rows(kind, identifier)
        denominator = int(rows[f"{kind}_{model}_denominator"].iloc[0]) if len(rows) else 0
        entity_key = "target_chembl_id" if kind == "drug" else "ligand_inchikey"
        name_key = "gene_symbol" if kind == "drug" else "drug_names"
        other_kind = "target" if kind == "drug" else "drug"
        if relationship != "all":
            known = rows.get("is_any_frozen_known_relationship", pd.Series(False, index=rows.index)).map(_bool)
            rows = rows[known if relationship == "known" else ~known]
        if search.strip():
            needle = search.strip().lower()
            mask = rows[entity_key].str.lower().str.contains(needle, regex=False) | rows[name_key].fillna("").str.lower().str.contains(needle, regex=False)
            rows = rows[mask]
        rows = rows.sort_values([f"{kind}_{model}_rank", entity_key], ascending=[order == "asc", True], na_position="last", kind="stable")
        total = len(rows)
        if not export:
            rows = rows.iloc[(page - 1) * page_size:page * page_size]
        items = []
        for row in rows.to_dict("records"):
            scores = {m: row["biomaster_reverse"] if m == "biomaster" and kind == "target" else row[m] for m in MODELS}
            ranks = {m: row[f"{kind}_{m}_rank"] for m in MODELS}
            denominators = {m: row[f"{kind}_{m}_denominator"] for m in MODELS}
            counterpart = (self.targets if kind == "drug" else self.drugs)[str(row[entity_key])]
            items.append({"id": str(row[entity_key]), "kind": other_kind, "name": str(row[name_key]), "subtitle": counterpart.get("subtitle"),
                  "identifiers": counterpart["identifiers"], "rank": ranks[model], "score": scores[model], "scores": scores, "ranks": ranks,
                  "denominators": denominators, "known_relation": _bool(row.get("is_any_frozen_known_relationship")) if "is_any_frozen_known_relationship" in row else None,
                  "action": row.get("chembl37_action_types"), "mechanism": row.get("chembl37_mechanism_of_action"),
                  "novelty": row.get("pair_novelty_class_384"), "pocket_source": row.get("pair_pocket_evidence_source"),
                  "drugclip_scope": row.get("drugclip_evidence_scope"), "drugclip_std": row.get("drugclip_sixfold_std"),
                  "frozen_score": row.get("frozen"), "frozen_rank": row.get(f"{kind}_frozen_rank"),
                  "frozen_denominator": row.get(f"{kind}_frozen_denominator")})
        return clean({"kind": kind, "id": identifier, "model": model, "page": page, "page_size": page_size,
                      "total": total, "denominator": denominator, "catalog_count": len(self._rows(kind, identifier)), "relationship_filter": relationship,
                      "scope": "完整可评分核心；过滤与分页不改变排名" if kind == "drug" else "完整720老药范围；反向检索仅为辅助证据",
                      "auxiliary": kind == "target", "items": items, "source": MODEL_INFO[model],
                      "rank_policy": "descending score, deterministic entity-ID tie break; missing scores have null rank"})

    def summary(self) -> dict:
        self.ensure_loaded()
        counts = {"drugs": len(self.drugs), "targets": len(self.targets), "scored_targets": int(self.pairs.target_chembl_id.nunique()),
                  "pairs": len(self.pairs), "pockets": sum(len(x.get("pockets", [])) for x in self.targets.values()),
                  "structures": sum(bool(x.get("structure")) for x in self.targets.values()),
                  "pathways": sum(len(x["pathways"]) for x in self.targets.values()),
                  "target_diseases": sum(len(x["target_diseases"]) for x in self.targets.values()),
                  "txgnn_associations": sum(len(x["txgnn_diseases"]) for x in self.drugs.values()),
                  "known_diseases": sum(len(x["known_diseases"]) for x in self.drugs.values())}
        counts["non_gpcr_targets"] = sum(bool(item["properties"].get("non_gpcr")) for item in self.targets.values())
        counts.update(getattr(self, "annotation_counts", {}))
        counts.update(getattr(self, "experiment_counts", {}))
        models = [dict(MODEL_INFO[m], id=m, coverage=int(self.pairs[m].notna().sum()),
                       targets=int(self.pairs.loc[self.pairs[m].notna(), "target_chembl_id"].nunique()),
                       available=bool(self.pairs[m].notna().any())) for m in MODELS]
        def featured(kind, names):
            result = []
            for name in names:
                matches = self.search(name, kind, 1)["items"]
                if matches:
                    result.extend(matches)
            return result
        return clean({"project": "Palinova", "version": "2026-09-06 Selected", "counts": counts, "models": models,
                      "sources": self.sources, "warnings": self.warnings,
                      "selected_provenance": getattr(self, "selected_provenance", None), "experiments": getattr(self, "experiment_summary", {}),
                      "featured": {"drugs": featured("drug", ["imatinib", "erlotinib", "dasatinib", "gefitinib", "metformin", "ruxolitinib"]),
                                   "targets": featured("target", ["EGFR", "JAK2", "RET", "MGLL", "ABL1", "BRAF"])},
                      "boundaries": ["720种老药 × 384个已评分核心靶点；888官方登记（含745非GPCR）不扩大排序分母。",
                          "ReTargetMap采用2026-09-06选定模型的≤2025部署权重；双方向logit不是概率或实验亲和力。",
                          "给定靶点的老药排名使用反向输出，仅作为辅助证据。",
                          "DrugCLIP实验与预测口袋跨来源未统一校准；总排名为探索性比较。",
                          "TxGNN仅展示本地缓存覆盖的疾病预测；未收录关系代表未知，不等同于阴性。",
                          "AlphaFold/P2Rank为预测结构和预测口袋；实验口袋单独标注证据来源。"]})

    def structure_path(self, target_id: str, pocket: str = "") -> Path:
        target_id = self.resolve("target", target_id)
        path = self.structure_files.get((target_id, pocket))
        if path is None:
            raise KeyError("Structure or pocket not available")
        path = Path(path).resolve()
        allowed = [self.root / "data/processed/alphafold_receptors_v6",
                   self.root / "outputs/strict_receptor_protocol_338_v1",
                   self.root / "outputs/recovered_no_experimental_pocket_targets_ch37_v1"]
        if not any(path.is_relative_to(directory.resolve()) for directory in allowed) or not path.is_file():
            raise KeyError("Structure path outside the registered local structure roots")
        if path.suffix.lower() not in (".pdb", ".cif", ".gz", ".sdf"):
            raise KeyError("Unsupported structure format")
        return path

    def molecule_svg(self, drug_id: str) -> str | None:
        drug_id = self.resolve("drug", drug_id)
        with self._lock:
            if drug_id in self._svg_cache:
                return self._svg_cache[drug_id]
            smiles = self.drugs[drug_id]["properties"].get("smiles")
            if not _present(smiles):
                return None
            try:
                from rdkit import Chem
                from rdkit.Chem.Draw import rdMolDraw2D
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    return None
                drawer = rdMolDraw2D.MolDraw2DSVG(520, 300)
                drawer.drawOptions().clearBackground = False
                rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
                drawer.FinishDrawing()
                svg = drawer.GetDrawingText()
            except ImportError:
                svg = None
            self._svg_cache[drug_id] = svg
            return svg
