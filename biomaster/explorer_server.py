"""Small local HTTP application for the Palinova research explorer."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import mimetypes
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .explorer_data import ExplorerData, MODELS, clean


class ExplorerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, data: ExplorerData, static_dir: Path):
        self.auth = None
        self.data = data
        from .explorer_spr_results import SPRResults
        self.spr_results = SPRResults(data.root)
        self.static_dir = Path(static_dir).resolve()
        super().__init__(address, ExplorerHandler)


class ExplorerHandler(BaseHTTPRequestHandler):
    server: ExplorerHTTPServer
    server_version = "PalinovaExplorer/1"

    def log_message(self, fmt, *args):
        logging.getLogger(__name__).info("%s %s", self.address_string(), fmt % args)

    def _send(self, body: bytes, content_type: str, status: int = 200, *, filename: str | None = None):
        path = urlsplit(self.path).path
        is_api = path.startswith("/api/")
        cache_control = "no-store" if is_api or status != 200 else "no-cache"
        if not is_api and status == 200 and not content_type.startswith("text/html"):
            cache_control = ("public, max-age=31536000, immutable"
                             if re.fullmatch(r"/assets/[^/]+-[A-Za-z0-9_-]{8,}\.(?:js|css)", path)
                             else "public, max-age=3600")
        if self.server.auth is not None:
            cache_control = "private, no-store"
        compressible = content_type.startswith(("text/", "application/json", "application/javascript", "image/svg+xml", "chemical/"))
        encodings = {}
        for part in self.headers.get("Accept-Encoding", "").lower().split(","):
            name, *params = part.strip().split(";")
            quality = 1.0
            for param in params:
                if param.strip().startswith("q="):
                    try:
                        quality = float(param.strip()[2:])
                    except ValueError:
                        quality = 0.0
            encodings[name.strip()] = quality
        compressed = compressible and len(body) >= 1024 and encodings.get("gzip", encodings.get("*", 0)) > 0
        if compressed:
            body = gzip.compress(body, compresslevel=5, mtime=0)
        etag = '"' + hashlib.sha256(body).hexdigest()[:32] + '"'
        not_modified = status == 200 and not is_api and etag in self.headers.get("If-None-Match", "").split(", ")
        self.send_response(304 if not_modified else status)
        self.send_header("Content-Type", content_type)
        if not not_modified:
            self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Cache-Control", cache_control)
        self.send_header("Vary", "Accept-Encoding")
        if not is_api and status == 200:
            self.send_header("ETag", etag)
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        if self.command != "HEAD" and not not_modified:
            self.wfile.write(body)

    def _json(self, payload, status=200):
        self._send(json.dumps(clean(payload), ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8", status)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in ("/api/spr-results/preview", "/api/spr-results/commit"):
            if self.server.auth is not None:
                self.server.auth.post(self)
            else:
                self._json({"error": "Not found"}, 404)
            return
        try:
            origin = self.headers.get("Origin")
            if (origin and urlsplit(origin).netloc != self.headers.get("Host")) or self.headers.get("Sec-Fetch-Site") == "cross-site":
                self.close_connection = True
                self._json({"error": "请求来源不匹配。"}, 403)
                return
            owner = self.server.auth.username(self) if self.server.auth else "local"
            if not owner:
                self.close_connection = True
                self._json({"error": "请先登录后上传。"}, 401)
                return
            size = int(self.headers.get("Content-Length", "0"))
            if self.headers.get("Transfer-Encoding") or not 0 < size <= 3 * 1024 * 1024:
                self.close_connection = True
                self._json({"error": "请求过大或长度无效；CSV 最大 2 MB。"}, 413)
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self.close_connection = True
                self._json({"error": "需要 application/json 请求。"}, 415)
                return
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("无效的上传请求。")
            store = self.server.spr_results
            if path.endswith("/preview"):
                self._json(store.preview(payload, store.registry(self.server.data), owner))
            else:
                token = payload.get("token")
                if not isinstance(token, str): raise ValueError("缺少预览编号。")
                self._json(store.commit(token, owner))
        except (ValueError, TypeError) as exc:
            self._json({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            logging.getLogger(__name__).exception("SPR result upload failed")
            self._json({"error": "结果保存失败，请查看服务日志后重试。"}, 500)

    def do_GET(self):
        try:
            if self.server.auth is not None and self.server.auth.get(self):
                return
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            arg = lambda name, default="": query.get(name, [default])[0]
            data = self.server.data
            if path == "/api/health":
                self._json({"status": "ok", "loaded": data._ready})
            elif path == "/api/summary":
                self._json(data.summary())
            elif path == "/api/browse":
                from .explorer_browse import browse
                self._json(browse(data, arg("section", "known_diseases"), arg("search"), arg("category", "all"), int(arg("page", "1")), int(arg("page_size", "20"))))
            elif path == "/api/affinity-matrix":
                from .explorer_affinity import AffinityAtlas
                self._json(AffinityAtlas(data.root).matrix(data, self.server.spr_results, arg("endpoint", "Kd"), float(arg("threshold", "1000")), arg("source", "all")))
            elif path == "/api/affinity-pair":
                from .explorer_affinity import AffinityAtlas
                self._json(AffinityAtlas(data.root).pair(arg("drug"), arg("target"), self.server.spr_results))
            elif path == "/api/spr-results/catalog":
                store = self.server.spr_results
                registry = store.registry(data)
                rows = store.catalog(registry, arg("scope", "baseline"), arg("order", "priority"), arg("target"), arg("search"))
                fields = ("experiment_id", "pair_id", "drug_name", "drug_id", "target_id", "target_name", "group", "priority_rank", "priority", "template_order")
                self._json({"items": [{k: row.get(k) for k in fields} for row in rows], "total": len(rows)})
            elif path == "/api/spr-results/template.csv":
                store = self.server.spr_results
                self._send(store.template(store.registry(data), arg("scope", "baseline"), arg("order", "priority"), arg("target"), arg("search")), "text/csv; charset=utf-8", filename="biomaster_spr_results_template.csv")
            elif path in ("/api/spr-results", "/api/spr-results.csv"):
                exporting = path.endswith(".csv")
                result = self.server.spr_results.results(arg("search"), arg("pair"), int(arg("page", "1")), int(arg("page_size", "20")), exporting)
                if exporting:
                    from .explorer_spr_results import csv_bytes, FIELDS
                    fields = FIELDS + ["KD_nM", "batch", "drug_id", "target_id", "drug_name", "target_name", "design_id", "is_control", "review_status", "uploaded_by", "uploaded_at", "filename", "upload_id", "source_line", "entry_method"]
                    self._send(csv_bytes(result["items"], fields), "text/csv; charset=utf-8", filename="biomaster_spr_results.csv")
                else:
                    self._json(result)
            elif path == "/api/spr-expanded":
                from .explorer_spr_expansion import expanded_items
                self._json({"items": expanded_items(data.root)})
            elif path in ("/api/spr-final.csv", "/api/spr-final-four-columns.csv", "/api/spr-final-controls.csv", "/api/spr-final.xlsx"):
                from .explorer_spr_final import DIRECTORY, load_final
                load_final(data.root)
                name = {"/api/spr-final.csv": "SPR384_FINAL_EXPERIMENT_TABLE.csv", "/api/spr-final-four-columns.csv": "SPR384_FINAL_FOUR_COLUMNS.csv", "/api/spr-final-controls.csv": "SPR112_REFERENCE_CONTROLS.csv", "/api/spr-final.xlsx": "SPR384_FINAL_EXPERIMENT_TABLE.xlsx"}[path]
                self._send((data.root / DIRECTORY / name).read_bytes(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if name.endswith(".xlsx") else "text/csv; charset=utf-8", filename=name)
            elif path == "/api/spr-design":
                data.ensure_loaded()
                items = [row for target in data.targets.values() for row in target.get("experiments", [])]
                from .explorer_spr_reviews import live_spr_items
                items = live_spr_items(data.root, items)
                self._json({"items": items})
            elif path == "/api/search":
                self._json(data.search(arg("q"), arg("kind", "all"), int(arg("limit", "20"))))
            elif path.startswith("/api/entity/"):
                parts = path.split("/")
                if len(parts) != 5:
                    raise KeyError("Entity not found")
                self._json(data.entity(parts[3], parts[4]))
            elif path in ("/api/rankings", "/api/rankings.csv", "/api/export/rankings.csv"):
                exporting = path.endswith(".csv") or arg("format") == "csv"
                result = data.rankings(arg("kind", "drug"), arg("id"), arg("model", "biomaster"), arg("search"),
                                       int(arg("page", "1")), int(arg("page_size", "20")), export=exporting, relationship=arg("relationship", "all"), order=arg("order", "asc"))
                if not exporting:
                    self._json(result)
                else:
                    stream = io.StringIO(newline="")
                    fields = ["query_kind", "query_id", "rank", "denominator", "entity_id", "name", "score", "known_relation"]
                    fields += [f"{model}_{metric}" for model in MODELS for metric in ("score", "rank", "denominator")]
                    fields += ["frozen_score", "frozen_rank", "frozen_denominator", "source", "auxiliary", "scope"]
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for item in result["items"]:
                        row = {"query_kind": result["kind"], "query_id": result["id"], "rank": item["rank"], "denominator": result["denominator"],
                               "entity_id": item["id"], "name": item["name"], "score": item["score"], "known_relation": item["known_relation"],
                               "frozen_score": item["frozen_score"], "frozen_rank": item["frozen_rank"], "frozen_denominator": item["frozen_denominator"],
                               "source": result["source"]["source"], "auxiliary": result["auxiliary"], "scope": result["scope"]}
                        for model in MODELS:
                            row.update({f"{model}_score": item["scores"][model], f"{model}_rank": item["ranks"][model], f"{model}_denominator": item["denominators"][model]})
                        # Neutralize spreadsheet formula prefixes in textual evidence.
                        row = {key: ("'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")) else value) for key, value in row.items()}
                        writer.writerow(row)
                    self._send(stream.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8", filename=f"biomaster_{result['kind']}_{result['model']}_rankings.csv")
            elif path == "/api/evidence":
                self._json(data.evidence(arg("kind", "target"), arg("id"), arg("section", "target_diseases"),
                                        int(arg("page", "1")), int(arg("page_size", "50")), arg("search")))
            elif path.startswith("/api/structure/"):
                identifier = path.removeprefix("/api/structure/")
                source = data.structure_path(identifier, arg("pocket"))
                body = gzip.decompress(source.read_bytes()) if source.suffix == ".gz" else source.read_bytes()
                self._send(body, "chemical/x-mdl-sdfile" if source.suffix == ".sdf" else "text/plain; charset=utf-8")
            elif path.startswith("/api/molecule/") and path.endswith(".svg"):
                identifier = path.removeprefix("/api/molecule/").removesuffix(".svg")
                svg = data.molecule_svg(identifier)
                if svg is None:
                    raise KeyError("Molecular structure is unavailable")
                self._send(svg.encode(), "image/svg+xml; charset=utf-8")
            elif path.startswith("/api/"):
                raise KeyError("API route not found")
            else:
                self._static(path)
        except (KeyError, FileNotFoundError) as exc:
            self._json({"error": str(exc).strip("'"), "status": 404}, 404)
        except (ValueError, TypeError) as exc:
            self._json({"error": str(exc), "status": 400}, 400)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            logging.getLogger(__name__).exception("Explorer request failed")
            self._json({"error": "本地数据读取失败，请查看服务日志。", "status": 500}, 500)

    def _static(self, path: str):
        root = self.server.static_dir
        relative = path.lstrip("/") or "index.html"
        requested = (root / relative).resolve()
        if not requested.is_relative_to(root):
            raise KeyError("File not found")
        if not requested.is_file():
            # Browser routes have no extension; missing JS/assets should stay 404.
            if not Path(relative).suffix and (root / "index.html").is_file():
                requested = root / "index.html"
            elif relative == "index.html":
                self._send("<!doctype html><meta charset=utf-8><title>Palinova</title><h1>Palinova API 已启动</h1><p>请先构建前端：cd web &amp;&amp; npm install &amp;&amp; npm run build</p><p><a href=/api/summary>查看真实项目数据</a></p>".encode(), "text/html; charset=utf-8")
                return
            else:
                raise KeyError("File not found")
        mime = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        self._send(requested.read_bytes(), mime + ("; charset=utf-8" if mime.startswith("text/") or mime in ("application/javascript", "application/json") else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the local Palinova visualization explorer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project data root")
    parser.add_argument("--static-dir", type=Path, default=None)
    parser.add_argument("--no-infer", action="store_true", help="Only read an existing selected-model cache; missing scores remain null")
    parser.add_argument("--lazy", action="store_true", help="Load data on the first API request")
    default_auth = Path.home() / ".config/biomaster/users.json"
    parser.add_argument("--auth-file", type=Path, default=default_auth if default_auth.is_file() else None, help="Enable password login using a hashed user file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = ExplorerData(args.root, infer=not args.no_infer)
    if not args.lazy:
        data.ensure_loaded()
    server = ExplorerHTTPServer((args.host, args.port), data, args.static_dir or args.root / "web/dist")
    if args.auth_file:
        from .explorer_auth import Auth
        server.auth = Auth(args.auth_file)
    logging.info("Palinova Explorer: http://%s:%d", args.host, server.server_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
