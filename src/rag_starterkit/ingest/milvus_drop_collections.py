from __future__ import annotations

import os
import re
import sys
from typing import List, Optional

from pymilvus import connections, utility


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v and v.strip() else default


def list_collections() -> List[str]:
    return list(utility.list_collections())


def drop_collections(
    host: str,
    port: str,
    prefix: Optional[str] = None,
    regex: Optional[str] = None,
    drop_all: bool = False,
    dry_run: bool = True,
    require_confirm: bool = True,
):
    connections.connect(alias="default", host=host, port=port)

    cols = list_collections()

    if not cols:
        print("No collections found.")
        return

    # Decide targets
    if drop_all:
        targets = cols
    elif prefix:
        targets = [c for c in cols if c.startswith(prefix)]
    elif regex:
        pat = re.compile(regex)
        targets = [c for c in cols if pat.search(c)]
    else:
        raise ValueError("Provide one of: --all OR --prefix PREFIX OR --regex REGEX")

    targets = sorted(set(targets))

    if not targets:
        print("No matching collections to drop.")
        return

    print("Milvus target:", f"{host}:{port}")
    print("Collections found:", len(cols))
    print("Collections to drop:", len(targets))
    for c in targets:
        print(" -", c)

    if dry_run:
        print("\nDry run enabled. Nothing was dropped.")
        print("Run again with --execute to actually drop collections.")
        return

    if require_confirm:
        print("\nWARNING: This will permanently delete the collections above.")
        ans = input('Type "DROP" to confirm: ').strip()
        if ans != "DROP":
            print("Confirmation not received. Aborting.")
            return

    # Drop
    dropped = 0
    for c in targets:
        try:
            utility.drop_collection(c)
            dropped += 1
            print(f"DROPPED: {c}")
        except Exception as e:
            print(f"FAILED: {c} -> {e}")

    print(f"\nDone. Dropped {dropped}/{len(targets)} collections.")


def main():
    host = _env("MILVUS_HOST", "127.0.0.1")
    port = _env("MILVUS_PORT", "19530")

    args = sys.argv[1:]

    drop_all = "--all" in args
    dry_run = "--execute" not in args
    no_confirm = "--no-confirm" in args

    prefix = None
    regex = None

    if "--prefix" in args:
        i = args.index("--prefix")
        if i + 1 >= len(args):
            raise SystemExit("Missing value for --prefix")
        prefix = args[i + 1]

    if "--regex" in args:
        i = args.index("--regex")
        if i + 1 >= len(args):
            raise SystemExit("Missing value for --regex")
        regex = args[i + 1]

    drop_collections(
        host=host,
        port=port,
        prefix=prefix,
        regex=regex,
        drop_all=drop_all,
        dry_run=dry_run,
        require_confirm=(not no_confirm),
    )


if __name__ == "__main__":
    main()
