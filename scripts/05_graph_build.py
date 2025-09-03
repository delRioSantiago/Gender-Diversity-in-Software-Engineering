from itertools import combinations
import pandas as pd

AUTHORSHIP_FILTERED_CSV = "./data/processed/authorship_filtered.csv"   # expects: paper_id, author_id
OUT_EDGES_CSV = "./data/processed/edges.csv"                           # writes: u, v

# Load authorships and restrict to valid authors
authorship = pd.read_csv(AUTHORSHIP_FILTERED_CSV, dtype=str)[["paper_id", "author_id"]]

# Create full-counting edges per paper
edges = []
for _, grp in authorship.groupby("paper_id"):
    ids = list(dict.fromkeys(grp["author_id"].tolist()))  # Remove duplicates in the same paper
    for u, v in combinations(ids, 2):
        a, b = (u, v) if u <= v else (v, u)               # Canonical sorting for undirected edge
        edges.append((a, b))

edges_df = pd.DataFrame(edges, columns=["u", "v"]).drop_duplicates()
edges_df.to_csv(OUT_EDGES_CSV, index=False)
print(f"edges.csv geschrieben: {len(edges_df)} Kanten")