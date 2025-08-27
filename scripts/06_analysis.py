import pandas as pd
import networkx as nx

AUTHORS_WITH_GENDER_CSV = "./data/processed/authors_with_gender.csv"  # author_id, gender
EDGES_CSV = "./data/processed/edges.csv"                              # u, v

# Daten laden
nodes_df = pd.read_csv(AUTHORS_WITH_GENDER_CSV, dtype=str)[["author_id","gender"]]
edges_df = pd.read_csv(EDGES_CSV, dtype=str)[["u","v"]]

# Graph aufbauen
G = nx.Graph()
G.add_nodes_from(nodes_df["author_id"])
G.add_edges_from(edges_df.itertuples(index=False, name=None))
nx.set_node_attributes(G, dict(zip(nodes_df["author_id"], nodes_df["gender"])), "gender")

# Basisstats
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
n_comp = nx.number_connected_components(G)
components = sorted(nx.connected_components(G), key=len, reverse=True)
lcc_size = len(components[0]) if components else 0
lcc_frac = lcc_size / n_nodes if n_nodes else 0.0

H = G.subgraph(components[0]).copy() if components else nx.Graph()
lcc_density = nx.density(H) if lcc_size > 1 else 0.0

unknown_total = sum(1 for n in G if G.nodes[n].get("gender","unknown")=="unknown")
unknown_lcc = sum(1 for n in H if H.nodes[n].get("gender","unknown")=="unknown")
unknown_rate_total = unknown_total / n_nodes if n_nodes else 0.0
unknown_rate_lcc = unknown_lcc / lcc_size if lcc_size else 0.0

print(f"Knoten: {n_nodes} | Kanten: {n_edges} | Komponenten: {n_comp}")
print(f"LCC: {lcc_size} Knoten ({lcc_frac:.2%}) | Dichte LCC: {lcc_density:.4f}")
print(f"Unknown gesamt: {unknown_total} ({unknown_rate_total:.2%}) | Unknown in LCC: {unknown_lcc} ({unknown_rate_lcc:.2%})")