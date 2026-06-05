from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import shutil
import stat
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2.cuda12.8"
DEFAULT_OUTPUT = "/root/autodl-tmp/tools/gnina/gnina.1.3.2"
DEFAULT_EXPECTED_SIZE = 2_052_029_472


def request_size(url: str, timeout: int) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        size = response.headers.get("Content-Length")
        return int(size) if size and size.isdigit() else None


def seed_first_chunk(output: Path, chunk_dir: Path, chunk_size: int) -> None:
    if not output.exists() or output.stat().st_size <= 0:
        return
    chunk_dir.mkdir(parents=True, exist_ok=True)
    first_chunk = chunk_dir / "chunk_00000.part"
    if first_chunk.exists() and first_chunk.stat().st_size > 0:
        return
    with output.open("rb") as source, first_chunk.open("wb") as target:
        shutil.copyfileobj(source, target, length=min(output.stat().st_size, chunk_size))


def download_chunk(
    url: str,
    chunk_path: Path,
    start: int,
    end: int,
    retries: int,
    timeout: int,
    throttle_sleep: float,
) -> tuple[Path, int]:
    expected = end - start + 1
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    existing = chunk_path.stat().st_size if chunk_path.exists() else 0
    if existing == expected:
        return chunk_path, expected
    if existing > expected:
        chunk_path.unlink()
        existing = 0

    for attempt in range(1, retries + 1):
        range_start = start + existing
        request = urllib.request.Request(url, headers={"Range": f"bytes={range_start}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status not in {206, 200}:
                    raise RuntimeError(f"unexpected HTTP status {response.status}")
                if response.status == 200 and range_start != 0:
                    raise RuntimeError("server ignored nonzero Range request")
                mode = "ab" if existing else "wb"
                with chunk_path.open(mode) as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            if attempt == retries:
                raise RuntimeError(f"failed chunk {chunk_path.name} after {retries} attempts: {exc}") from exc
            time.sleep(min(30, 2 * attempt))
        current = chunk_path.stat().st_size if chunk_path.exists() else 0
        if current == expected:
            return chunk_path, expected
        existing = current
        if throttle_sleep:
            time.sleep(throttle_sleep)
    raise RuntimeError(f"incomplete chunk {chunk_path.name}: {existing}/{expected}")


def assemble(output: Path, chunks: list[Path], expected_size: int) -> None:
    tmp = output.with_suffix(output.suffix + ".assembled")
    with tmp.open("wb") as target:
        for chunk in chunks:
            with chunk.open("rb") as source:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    actual = tmp.stat().st_size
    if actual != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"assembled size mismatch: {actual}/{expected_size}")
    tmp.replace(output)
    mode = output.stat().st_mode
    output.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the GNINA release binary with resumable HTTP ranges.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-size", type=int, default=DEFAULT_EXPECTED_SIZE)
    parser.add_argument("--chunk-size-mb", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--throttle-sleep", type=float, default=0.0)
    parser.add_argument("--no-seed", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = output.parent / "chunks" / output.name
    chunk_size = args.chunk_size_mb * 1024 * 1024
    expected_size = args.expected_size or request_size(args.url, args.timeout)
    if not expected_size:
        raise RuntimeError("could not determine expected size")

    if output.exists() and output.stat().st_size == expected_size:
        mode = output.stat().st_mode
        output.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"already complete: {output} ({expected_size} bytes)")
        return 0

    if not args.no_seed:
        seed_first_chunk(output, chunk_dir, chunk_size)

    chunk_count = math.ceil(expected_size / chunk_size)
    ranges = []
    for idx in range(chunk_count):
        start = idx * chunk_size
        end = min(expected_size - 1, start + chunk_size - 1)
        ranges.append((idx, start, end, chunk_dir / f"chunk_{idx:05d}.part"))

    print(f"downloading {expected_size} bytes in {chunk_count} chunks to {output}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_chunk,
                args.url,
                chunk_path,
                start,
                end,
                args.retries,
                args.timeout,
                args.throttle_sleep,
            )
            for _, start, end, chunk_path in ranges
        ]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            chunk_path, size = future.result()
            completed += size
            current = sum(path.stat().st_size for _, _, _, path in ranges if path.exists())
            print(f"chunk done: {chunk_path.name}; staged {current}/{expected_size} bytes")

    chunks = [chunk_path for _, _, _, chunk_path in ranges]
    missing = [path for path in chunks if not path.exists()]
    if missing:
        raise RuntimeError(f"missing chunks: {[path.name for path in missing[:10]]}")
    assemble(output, chunks, expected_size)
    print(f"complete: {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
