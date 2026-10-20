"""Small, reusable TF-IDF index over the supplied policy documents."""

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"


class PolicyIndex:
    def __init__(self, policy_dir=POLICY_DIR, chunk_size=40, overlap=10):
        if not 0 <= overlap < chunk_size:
            raise ValueError("Overlap must be non-negative and smaller than the chunk.")
        self.chunks = []
        for path in sorted(Path(policy_dir).glob("*.txt")):
            words = path.read_text(encoding="utf-8").split()
            for start in range(0, len(words), chunk_size - overlap):
                self.chunks.append({
                    "source": path.name,
                    "text": " ".join(words[start:start + chunk_size]),
                })
                # Do not add a redundant tail after a chunk covers the last word.
                if start + chunk_size >= len(words):
                    break
        if not self.chunks:
            raise ValueError("No policies found; run generate_data.py first.")
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(c["text"] for c in self.chunks)

    def retrieve(self, query, k=3):
        if k < 1:
            raise ValueError("k must be positive.")
        scores = cosine_similarity(
            self.vectorizer.transform([query]), self.matrix
        ).ravel()
        # Source-file and chunk order break ties reproducibly.
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [
            {**self.chunks[i], "score": float(scores[i])}
            for i in ranked
        ]


_index = None


def retrieve(query, k=3):
    global _index
    if _index is None:
        _index = PolicyIndex()
    return _index.retrieve(query, k)
