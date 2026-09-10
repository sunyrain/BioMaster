"""Portable, frozen display snapshot. No training, inference or remote data access."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser

import pandas as pd

from .explorer_data import ExplorerData
from .explorer_disease import DiseaseEvidenceStore
from .explorer_server import ExplorerHTTPServer, ExplorerHandler


class OfflineData(ExplorerData):
    def _load(self):
        with gzip.open(self.root / "snapshot/entities.json.gz", "rt", encoding="utf-8") as stream:
            snapshot = json.load(stream)
        for key in ("drugs", "targets", "sources", "warnings", "annotation_counts",
                    "experiment_counts", "experiment_summary", "selected_provenance"):
            setattr(self, key, snapshot[key])
        self._aliases = snapshot["aliases"]
        self.pairs = pd.read_parquet(self.root / "snapshot/pairs.parquet")
        self._drug_groups = self.pairs.groupby("ligand_inchikey").indices
        self._target_groups = self.pairs.groupby("target_chembl_id").indices
        self.structure_files = {(key, pocket): self.root / relative
                                for key, pocket, relative in snapshot["structure_files"]}
        self.diseases = DiseaseEvidenceStore(self.root, self.drugs, self.targets)
        if not self.diseases.enabled:
            raise ValueError("Offline disease snapshot is missing")

    def entity(self, kind, identifier):
        result = super().entity(kind, identifier)
        result.update(self.diseases.metadata(kind, result["id"]))
        for section in (("txgnn_diseases", "eckg") if kind == "drug" else ("target_diseases", "eckg")):
            result[section] = self.diseases.preview(kind, result["id"], section)
            result[section + "_total"] = self.diseases.total(kind, result["id"], section)
        return result

    def summary(self):
        result = super().summary()
        result["offline"] = True
        result["disease_evidence"] = self.diseases.summary
        return result

    def evidence(self, kind, identifier, section, page=1, page_size=50, search="", direction="", scope="eligible"):
        identifier = self.resolve(kind, identifier)
        if section in ("txgnn_diseases", "target_diseases", "eckg"):
            return self.diseases.query(kind, identifier, section, page, page_size, search, direction, scope)
        return super().evidence(kind, identifier, section, page, page_size, search)

    def structure_path(self, target_id, pocket=""):
        target_id = self.resolve("target", target_id)
        path = self.structure_files.get((target_id, pocket))
        if path is None:
            raise KeyError("Structure or pocket not available")
        path = path.resolve()
        if not path.is_relative_to((self.root / "structures").resolve()) or not path.is_file():
            raise KeyError("Structure is outside the offline package")
        return path

    def molecule_svg(self, drug_id):
        drug_id = self.resolve("drug", drug_id)
        path = self.root / "molecules" / (drug_id + ".svg")
        return path.read_text(encoding="utf-8") if path.is_file() else None


class OfflineHandler(ExplorerHandler):
    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path not in ("/api/disease-summary", "/api/evidence", "/api/evidence.csv"):
            return super().do_GET()
        try:
            data = self.server.data
            if parsed.path == "/api/disease-summary":
                self._json({"disease_evidence": data.diseases.summary})
                return
            query = parse_qs(parsed.query, keep_blank_values=True)
            arg = lambda name, default="": query.get(name, [default])[0]
            exporting = parsed.path.endswith(".csv")
            args = dict(kind=arg("kind", "target"), identifier=arg("id"),
                        section=arg("section", "target_diseases"),
                        page=1 if exporting else int(arg("page", "1")),
                        page_size=200 if exporting else int(arg("page_size", "50")),
                        search=arg("search"), direction=arg("direction"), scope=arg("scope", "eligible"))
            result = data.evidence(**args)
            if not exporting:
                self._json(result)
                return
            rows = list(result["items"])
            while len(rows) < result["total"]:
                args["page"] += 1
                batch = data.evidence(**args)["items"]
                if not batch:
                    raise ValueError("Incomplete evidence export")
                rows.extend(batch)
            fields = list(dict.fromkeys(key for row in rows for key in row)) or ["id"]
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                record = {}
                for key, value in row.items():
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, ensure_ascii=False)
                    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                        value = "'" + value
                    record[key] = value
                writer.writerow(record)
            self._send(stream.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8", filename="biomaster_evidence.csv")
        except (ValueError, TypeError) as exc:
            self._json({"error": str(exc), "status": 400}, 400)
        except (KeyError, FileNotFoundError) as exc:
            self._json({"error": str(exc), "status": 404}, 404)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            logging.exception("Offline evidence request failed")
            self._json({"error": "Offline evidence read failed", "status": 500}, 500)


def main():
    parser = argparse.ArgumentParser(description="BioMaster offline display")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = Path(__file__).resolve().parents[1]
    print("Loading BioMaster offline data. Please keep this window open.", flush=True)
    data = OfflineData(root, infer=False)
    data.ensure_loaded()
    server = ExplorerHTTPServer(("127.0.0.1", args.port), data, root / "web/dist")
    server.RequestHandlerClass = OfflineHandler
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"BioMaster ready: {url}\nPress Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
