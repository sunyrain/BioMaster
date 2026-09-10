#!/usr/bin/env python3
"""Export a model-free display snapshot; source data is never modified."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.explorer_data import ExplorerData, clean
from biomaster.explorer_disease import DiseaseEvidenceStore, REQUIRED, SNAPSHOT


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def build_frontend(dest):
    # Keep the live site's source and build unchanged while correcting the
    # legacy single-disease label for the complete offline evidence snapshot.
    with tempfile.TemporaryDirectory(prefix="offline-web-", dir=dest.parent) as temp:
        web = Path(temp)
        for name in ("src", "public"):
            shutil.copytree(ROOT / "web" / name, web / name)
        for name in ("package.json", "tsconfig.json", "vite.config.ts", "index.html"):
            shutil.copyfile(ROOT / "web" / name, web / name)
        (web / "node_modules").symlink_to(ROOT / "web/node_modules", target_is_directory=True)
        app = web / "src/App.tsx"
        text = app.read_text()
        old = "当前本地快照只覆盖\n              cancer（MONDO:0004992）。知识图谱预测分数不等于治疗有效概率。"
        if old not in text:
            old = "当前快照展示已计算的疾病关联。知识图谱预测分数不等于治疗有效概率。"
        assert old in text
        text = text.replace(old, "当前离线快照包含全疾病已计算结果，默认排除已知关系及身份暂扣记录。知识图谱预测分数不等于治疗有效概率。")
        app.write_text(text)
        home = web / "src/ResearchHome.tsx"
        home.write_text(home.read_text().replace("本地覆盖 cancer · MONDO_0004992", "全疾病固定结果 · 默认排除已知及暂扣记录").replace("已有疾病关联快照 · 预测非临床结论", "全疾病固定结果 · 默认排除已知及暂扣记录"))
        subprocess.run(["./node_modules/.bin/tsc", "--noEmit"], cwd=web, check=True)
        subprocess.run(["./node_modules/.bin/vite", "build", "--emptyOutDir", "--outDir", str(dest / "web/dist")], cwd=web, check=True)


def build(dest, runtime):
    if dest.exists():
        raise SystemExit(f"Destination already exists: {dest}")
    dest.mkdir(parents=True)
    for name in ("snapshot", "structures", "molecules", "biomaster", "web/dist"):
        (dest / name).mkdir(parents=True, exist_ok=True)
    print("Loading validated existing results (inference disabled)", flush=True)
    data = ExplorerData(ROOT, infer=False)
    data.ensure_loaded()
    if data.warnings or data.pairs.biomaster.isna().any() or data.pairs.biomaster_reverse.isna().any():
        raise ValueError(f"Incomplete frozen scores: {data.warnings}")
    diseases = DiseaseEvidenceStore(ROOT, data.drugs, data.targets)
    print("Exporting annotations, scores, structures and molecules", flush=True)
    snapshot = {key: getattr(data, key, {}) for key in (
        "drugs", "targets", "sources", "warnings", "annotation_counts", "experiment_counts",
        "experiment_summary", "selected_provenance")}
    snapshot["aliases"] = data._aliases
    snapshot["structure_files"] = []
    paths = {}
    for key, path in data.structure_files.items():
        if path not in paths:
            relative = "structures/" + f"{len(paths):04d}" + path.suffix
            shutil.copyfile(path, dest / relative)
            paths[path] = relative
        snapshot["structure_files"].append([*key, paths[path]])
    with gzip.open(dest / "snapshot/entities.json.gz", "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(clean(snapshot), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    data.pairs.to_parquet(dest / "snapshot/pairs.parquet", index=False, compression="zstd")
    svg_count = 0
    for key in data.drugs:
        svg = data.molecule_svg(key)
        if svg:
            (dest / "molecules" / (key + ".svg")).write_text(svg, encoding="utf-8")
            svg_count += 1
    if svg_count != len(data.drugs):
        raise ValueError(f"Incomplete molecular images: {svg_count}")
    for name in REQUIRED:
        source = ROOT / SNAPSHOT / name
        target = dest / SNAPSHOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    target_cache = dest / diseases.cache_path.relative_to(ROOT)
    target_cache.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(diseases.cache_path.as_uri() + "?mode=ro", uri=True) as src:
        with sqlite3.connect(target_cache) as dst:
            src.backup(dst)
    for name in ("__init__.py", "explorer_data.py", "explorer_server.py", "explorer_disease.py", "explorer_offline.py"):
        text = (ROOT / "biomaster" / name).read_text()
        if name == "explorer_disease.py":
            # An offline release ships a validated, immutable SQLite cache.
            # No rebuild (and therefore no POSIX-only fcntl lock) is required.
            text = text.replace("import fcntl\n", "")
            old = '''        with self.cache_path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self._cache_valid():
                self._build_cache()'''
            assert old in text
            text = text.replace(old, '''        if not self._cache_valid():
            raise ValueError("Offline disease cache missing or invalid; extract the full package again")''')
        (dest / "biomaster" / name).write_text(text, encoding="utf-8")
    (dest / "run_offline.py").write_text("from biomaster.explorer_offline import main\nif __name__ == '__main__':\n    main()\n")
    print("Building current frontend", flush=True)
    build_frontend(dest)
    print("Adding Windows portable runtime", flush=True)
    python_dir = dest / "runtime"
    python_dir.mkdir()
    embed = runtime / "python-3.11.9-embed-amd64.zip"
    with zipfile.ZipFile(embed) as z:
        z.extractall(python_dir)
    site = python_dir / "Lib/site-packages"
    site.mkdir(parents=True)
    wheels = sorted(runtime.glob("*.whl"))
    required = {"numpy", "pandas", "pyarrow", "python_dateutil", "pytz", "tzdata", "six"}
    if {p.name.split('-')[0] for p in wheels} != required:
        raise ValueError("Missing or unexpected runtime wheel")
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as z:
            for member in z.infolist():
                parts = Path(member.filename).parts
                if any(part.endswith(".data") for part in parts):
                    raise ValueError(f"Wheel requires installer layout handling: {wheel.name}: {member.filename}")
            z.extractall(site)
    (python_dir / "python311._pth").write_text("python311.zip\n.\n..\nLib/site-packages\nimport site\n")
    (dest / "START_BIOMASTER.bat").write_bytes(b'@echo off\r\ncd /d "%~dp0"\r\n"runtime\\python.exe" -B run_offline.py %*\r\nif errorlevel 1 pause\r\n')
    (dest / "VERIFY_PACKAGE.bat").write_bytes(b'@echo off\r\ncd /d "%~dp0"\r\n"runtime\\python.exe" -B verify_package.py\r\npause\r\n')
    (dest / "verify_package.py").write_text('''import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
for index, item in enumerate(manifest["files"], 1):
    path = root / item["path"]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    if path.stat().st_size != item["bytes"] or h.hexdigest() != item["sha256"]:
        raise SystemExit("FAILED: " + item["path"])
    if index % 500 == 0:
        print("Verified", index, "files", flush=True)
print("PASS: all", len(manifest["files"]), "files verified")
''')
    (dest / "requirements-linux.txt").write_text("numpy==2.3.5\npandas==2.3.3\npyarrow==21.0.0\n")
    (dest / "README_ZH.md").write_text('''# BioMaster 离线展示包

## Windows 10/11，64 位 x86 电脑

1. 将 ZIP 完整解压到可写目录，建议 `C:\\BioMasterDemo`。不要在 ZIP 内直接运行。
2. 双击 `START_BIOMASTER.bat`。等待显示 `BioMaster ready`，浏览器会自动打开。
3. 如果浏览器没有打开，访问 http://127.0.0.1:18765 。启动窗口保持打开，Ctrl+C 停止。
4. 若端口已占用，在此目录的终端执行 `START_BIOMASTER.bat --port 18888`。
5. 可双击 `VERIFY_PACKAGE.bat` 校验下载、解压是否完整。

已附带 Python 3.11.9 Windows x64、NumPy、pandas、PyArrow 及其依赖。
首次启动和后续展示均不需要联网，不需要另装 Python、Node.js、GPU、模型权重或完整 ChEMBL。
网页的外部来源链接仍指向原站，断网时这些外链不能打开；网站内置的查询、图表、结构和导出可离线使用。
包内保留研究结果的来源与解释边界。是固定结果快照，不支持新药物推理或训练。
内存建议 8 GB 起；模型和训练环境未包含。默认仅供本机浏览器访问。

## 包内容

- snapshot：已计算评分、实体信息、ChEMBL 适应症与机制、SPR 实验设计等展示快照。
- structures / molecules：实际引用的三维结构、预先生成的药物二维 SVG。
- outputs：全疾病评分、来源校验材料和查询 SQLite；不包含研究训练目录。
- web/dist：构建好的网页；runtime：Windows 便携运行环境。
- PACKAGE_MANIFEST.json：每个交付文件的 SHA-256、字节大小及运行组件来源。
- VALIDATION.json：服务器端隔离验证结果（附在发行包中）。

Linux/macOS 可使用 Python 3.11，安装 requirements-linux.txt 后运行 `python run_offline.py`。
Windows 便携二进制在 Linux 上不能原生运行；交付前在 Linux 隔离目录验证应用和数据，
Windows 双击启动仍需在实际 Windows 上确认。

Python 来源：https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
Python 便携包说明：https://docs.python.org/3.11/using/windows.html#the-embeddable-package
第三方 Python 组件许可证保留在 runtime/Lib/site-packages 各 dist-info 目录中。
''', encoding="utf-8")
    (dest / "BUILD_BASELINE.json").write_text(json.dumps(clean({
        "summary": data.summary(), "disease_counts": diseases.counts,
        "structure_count": len(paths), "molecule_count": svg_count,
        "rankings": [{"kind": kind, "id": identifier, "result": data.rankings(kind, identifier)}
                     for kind, identifier in [("drug", list(data.drugs)[0]), ("target", list(data.targets)[0])]]
    }), ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"format": "BioMaster-offline-v1", "no_inference": True,
                "runtime_sources": [{"file": p.name, "sha256": digest(p)} for p in [embed, *wheels]],
                "files": []}
    for p in sorted(dest.rglob("*")):
        if p.is_file():
            manifest["files"].append({"path": p.relative_to(dest).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)})
    (dest / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"package": str(dest), "files": len(manifest["files"]),
                      "bytes": sum(r["bytes"] for r in manifest["files"])}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    build(args.dest.resolve(), args.runtime.resolve())
