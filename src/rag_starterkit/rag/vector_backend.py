from typing import List, Dict

class VectorBackend:
    def upsert(self, docs: List[Dict]) -> int:
        raise NotImplementedError

    def query(self, query: str, top_k: int) -> List[Dict]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError
