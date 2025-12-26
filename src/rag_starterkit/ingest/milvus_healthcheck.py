from __future__ import annotations

import os
import sys
from typing import List, Optional

from pymilvus import connections, utility, Collection


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v and v.strip() else default


def milvus_healthcheck(
    host: str,
    port: str,
    show_collections: bool = True,
    load_collections: bool = False,
    limit: Optional[int] = None,
):
    print("=== Milvus Healthcheck ===")
    print(f"Target: {host}:{port}")
    print(f"pymilvus: OK (imported)")
    print()

    # Connect
    connections.connect(alias="default", host=host, port=port)
    print("Connection: OK")

    # Collections
    if not show_collections:
        return

    cols: List[str] = utility.list_collections()
    print(f"Collections found: {len(cols)}")
    if limit is not None:
        cols = cols[: max(0, int(limit))]

    if not cols:
        print("No collections exist yet.")
        return

    print()
    print("=== Collections Summary ===")
    for name in cols:
        try:
            col = Collection(name)

            if load_collections:
                try:
                    col.load()
                except Exception as e:
                    print(f"- {name}: load failed: {e}")
                    continue

            # num_entities is available without load in many setups, but load makes it reliable
            try:
                count = int(col.num_entities)
            except Exception as e:
                count = None
                print(f"- {name}: cannot read num_entities: {e}")

            # Index info
            idx_info = "no_index"
            try:
                indexes = col.indexes
                if indexes:
                    # show basic info for first index
                    idx = indexes[0]
                    params = getattr(idx, "params", None)
                    idx_info = f"index({idx.index_name}) params={params}" if params else f"index({idx.index_name})"
            except Exception:
                pass

            print(f"- {name}: entities={count} {idx_info}")

        except Exception as e:
            print(f"- {name}: ERROR: {e}")


def main():
    host = _env("MILVUS_HOST", "127.0.0.1")
    port = _env("MILVUS_PORT", "19530")

    # flags
    load = "--load" in sys.argv
    no_cols = "--no-cols" in sys.argv

    limit = None
    for i, a in enumerate(sys.argv):
        if a == "--limit" and i + 1 < len(sys.argv):
            try:
                limit = int(sys.argv[i + 1])
            except Exception:
                limit = None

    milvus_healthcheck(
        host=host,
        port=port,
        show_collections=(not no_cols),
        load_collections=load,
        limit=limit,
    )


if __name__ == "__main__":
    main()
