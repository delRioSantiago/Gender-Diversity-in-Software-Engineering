import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg
from pathlib import Path
from math import ceil

# Pfade
DATA_PROCESSED = Path("./data/processed")
AUTHORS_WITH_GENDER_CSV = DATA_PROCESSED / "authors_with_gender.csv"
EDGES_CSV = DATA_PROCESSED / "edges.csv"
RESULTS = Path("./data/results")
R_METRICS   = RESULTS / "metrics";    R_METRICS.mkdir(parents=True, exist_ok=True)
R_CENTRAL   = RESULTS / "centrality"; R_CENTRAL.mkdir(parents=True, exist_ok=True)
R_REMOVAL   = RESULTS / "removal";    R_REMOVAL.mkdir(parents=True, exist_ok=True)
R_EGO       = RESULTS / "ego";        R_EGO.mkdir(parents=True, exist_ok=True)

# Parameter
TOP_PCT = 0.05
BTWN_K_CAP = 2000
SEED = 42
N_RANDOM = 100
MF = {"male", "female"}


# Hilfsfunktionen 

def load_graph():
    nodes_df = pd.read_csv(AUTHORS_WITH_GENDER_CSV, dtype=str)[["author_id","gender"]]
    edges_df = pd.read_csv(EDGES_CSV, dtype=str)[["u","v"]]

    G = nx.Graph()
    G.add_nodes_from(nodes_df["author_id"])
    G.add_edges_from(edges_df.itertuples(index=False, name=None))
    nx.set_node_attributes(G, dict(zip(nodes_df["author_id"], nodes_df["gender"])), "gender")
    return G, nodes_df, edges_df


def mixed_edge_share(Gx):
    m = t = 0
    for u, v in Gx.edges():
        gu, gv = Gx.nodes[u].get("gender"), Gx.nodes[v].get("gender")
        if gu in MF and gv in MF:
            t += 1
            if gu != gv:
                m += 1
    return (m, t, (m / t) if t else np.nan)


def assortativity_known(Gx):
    nodes_known = [n for n, d in Gx.nodes(data=True) if d.get("gender") in MF]
    if not nodes_known:
        return np.nan
    Hk = Gx.subgraph(nodes_known).copy()
    if Hk.number_of_edges() == 0:
        return np.nan
    return nx.attribute_assortativity_coefficient(Hk, "gender")


# Analysis modules

def compute_global_metrics(G):
    """Globale und LCC-Kennzahlen"""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc_nodes = components[0] if components else set()
    H = G.subgraph(lcc_nodes).copy()

    male_total = sum(1 for n in G if G.nodes[n].get("gender")=="male")
    female_total = sum(1 for n in G if G.nodes[n].get("gender")=="female")
    unknown_total = sum(1 for n in G if G.nodes[n].get("gender")=="unknown")

    m_mixed, t_mixed, share_mixed = mixed_edge_share(G)
    r_global = assortativity_known(G)

    male_lcc = sum(1 for n in H if H.nodes[n].get("gender")=="male")
    female_lcc = sum(1 for n in H if H.nodes[n].get("gender")=="female")
    unknown_lcc = sum(1 for n in H if H.nodes[n].get("gender")=="unknown")
    m_mixed_lcc, t_mixed_lcc, share_mixed_lcc = mixed_edge_share(H)
    r_lcc = assortativity_known(H)

    metrics = pd.DataFrame([
        {"scope":"global","nodes":n_nodes,"edges":n_edges,
         "male":male_total,"female":female_total,
         "share_female":female_total/(n_nodes-unknown_total) if (n_nodes-unknown_total)>0 else np.nan,
         "unknown":unknown_total,"share_unknown":unknown_total/n_nodes if n_nodes>0 else np.nan,
         "share_mixed_known_edges":share_mixed,"assortativity_r":r_global},
        {"scope":"lcc","nodes":len(lcc_nodes),"edges":H.number_of_edges(),
         "male":male_lcc,"female":female_lcc,
         "share_female":female_lcc/(len(lcc_nodes)-unknown_lcc) if (len(lcc_nodes)-unknown_lcc)>0 else np.nan,
         "unknown":unknown_lcc,"share_unknown":unknown_lcc/len(lcc_nodes) if len(lcc_nodes)>0 else np.nan,
         "share_mixed_known_edges":share_mixed_lcc,"assortativity_r":r_lcc},
    ])
    metrics.to_csv(R_METRICS / "metrics_global.csv", index=False)
    return H, metrics


def compute_betweenness(H):
    """Approximierte Betweenness und Top-5%"""
    k = min(BTWN_K_CAP, H.number_of_nodes())
    btw = nx.betweenness_centrality(H, k=k, seed=SEED, normalized=False)
    btw_s = pd.Series(btw, name="betweenness").sort_values(ascending=False)
    top_n = max(1, ceil(TOP_PCT * H.number_of_nodes()))
    top_ids = set(btw_s.index[:top_n])
    df_btw = pd.DataFrame({
        "author_id": btw_s.index,
        "betweenness": btw_s.values,
        "is_top5_betweenness": [aid in top_ids for aid in btw_s.index],
    })
    df_btw.to_csv(R_CENTRAL / "betweenness_lcc.csv", index=False)
    return top_ids, df_btw


def removal_experiment(H, top_ids):
    """Top-5%-Entfernung vs. Random-Gruppen + Placebo Degree und Closeness"""
    nodes_known = [n for n,d in H.nodes(data=True) if d.get("gender") in MF]
    Hk = H.subgraph(nodes_known).copy()
    r_before = assortativity_known(Hk)

    top_known = [n for n in top_ids if n in Hk]
    remove_k = len(top_known)

    H_top_removed = Hk.copy()
    H_top_removed.remove_nodes_from(top_known)
    r_after_top = assortativity_known(H_top_removed)

    delta_r = r_after_top - r_before
    rel_change = delta_r / r_before if r_before!=0 else np.nan
    print(f"Δr = {delta_r:.5f}, relative Änderung = {rel_change:.2%}")

    # Random groups
    rng = np.random.default_rng(SEED)
    rand_rs = []
    for _ in range(N_RANDOM):
        sample = rng.choice(list(Hk.nodes()), size=remove_k, replace=False)
        Hr = Hk.copy(); Hr.remove_nodes_from(sample.tolist())
        rand_rs.append(assortativity_known(Hr))
    q025, q975 = np.nanpercentile(rand_rs, [2.5,97.5])
    direction = np.sign(delta_r) or 1
    if direction>=0:
        pval = np.nanmean(np.array(rand_rs) >= r_after_top)
    else:
        pval = np.nanmean(np.array(rand_rs) <= r_after_top)

    # Placebo Degree
    degree = dict(H.degree())
    top_deg = sorted(degree, key=degree.get, reverse=True)[:remove_k]
    Hd = H.copy(); Hd.remove_nodes_from(top_deg)
    r_after_degree = assortativity_known(Hd)
    
    # Placebo Closeness
    closeness = nx.closeness_centrality(H)
    top_clo = sorted(closeness, key=closeness.get, reverse=True)[:remove_k]
    Hc = H.copy(); Hc.remove_nodes_from(top_clo)
    r_after_closeness = assortativity_known(Hc)

    df_removal = pd.DataFrame({
        "r_before":[r_before],"r_after_top":[r_after_top],
        "rand_mean":[np.nanmean(rand_rs)],"rand_std":[np.nanstd(rand_rs, ddof=1)],
        "rand_q025":[q025],"rand_q975":[q975],"p_value":[pval],
        "removed_k":[remove_k],"n_random":[N_RANDOM],
        "r_after_degree":[r_after_degree],
        "r_after_closeness":[r_after_closeness],
    })
    df_removal.to_csv(R_REMOVAL / "removal_experiment.csv", index=False,float_format="%.6f")
    pd.DataFrame({"r_random":rand_rs}).to_csv(R_REMOVAL / "removal_experiment_samples.csv",index=False,float_format="%.6f")
    return df_removal


def ego_analysis(H, top_ids):
    """Vergleich Ego-Netze Top-5% vs. Referenz"""
    def ego_stats(Gx,nodes):
        rows=[]
        for n in nodes:
            neigh=list(Gx.neighbors(n))
            k=len(neigh)
            if k==0:
                rows.append({"author_id":n,"ego_size":0,"female_share":np.nan,"known_neighbors":0})
                continue
            genders=[Gx.nodes[v].get("gender","unknown") for v in neigh]
            known=[g for g in genders if g in MF]
            fem_share=(known.count("female")/len(known)) if known else np.nan
            rows.append({"author_id":n,"ego_size":k,"female_share":fem_share,"known_neighbors":len(known)})
        return pd.DataFrame(rows)

    non_top=[n for n in H.nodes() if n not in top_ids]
    ref_n=min(len(non_top),len(top_ids))
    rng=np.random.default_rng(SEED)
    ref_sample=rng.choice(non_top,size=ref_n,replace=False).tolist()

    df_top=ego_stats(H,list(top_ids))
    df_ref=ego_stats(H,ref_sample)

    df_top.to_csv(R_EGO/"ego_top.csv",index=False,float_format="%.6f")
    df_ref.to_csv(R_EGO/"ego_ref.csv",index=False,float_format="%.6f")

    def summarize(df,label):
        s={}
        s["group"]=label; s["n_nodes"]=len(df)
        s["ego_size_mean"]=df["ego_size"].mean()
        s["female_share_mean"]=df["female_share"].mean()
        return pd.Series(s)

    summary=pd.concat([summarize(df_top,"top5"),summarize(df_ref,"ref")],axis=1).T
    summary.to_csv(R_EGO/"ego_summary.csv",index=False,float_format="%.6f")
    return df_top, df_ref, summary

def community_analysis(H):
    """Führt Leiden-Community-Detection auf H durch und misst Diversität pro Community
       + erstellt zusätzlich eine Zusammenfassung über alle Communities
    """
    import igraph as ig
    import leidenalg

    # NetworkX -> iGraph conversion
    mapping = {n: i for i, n in enumerate(H.nodes())}
    rev_mapping = {i: n for n, i in mapping.items()}
    edges = [(mapping[u], mapping[v]) for u, v in H.edges()]
    g = ig.Graph(edges=edges, directed=False)
    
    # Adopt gender attribute
    genders = [H.nodes[rev_mapping[i]].get("gender","unknown") for i in range(g.vcount())]
    g.vs["gender"] = genders

    # Leiden-Clustering
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition)
    labels = partition.membership

    # Collect results
    rows=[]
    for comm_id in set(labels):
        members=[rev_mapping[i] for i,l in enumerate(labels) if l==comm_id]
        Gc=H.subgraph(members).copy()
        male=sum(1 for n in Gc if Gc.nodes[n].get("gender")=="male")
        female=sum(1 for n in Gc if Gc.nodes[n].get("gender")=="female")
        unknown=sum(1 for n in Gc if Gc.nodes[n].get("gender")=="unknown")
        m_mixed,t_mixed,share_mixed=mixed_edge_share(Gc)
        r=assortativity_known(Gc)
        rows.append({
            "community":comm_id,"size":len(members),
            "male":male,"female":female,"unknown":unknown,
            "share_female": female/(male+female) if (male+female)>0 else np.nan,
            "share_mixed_known_edges":share_mixed,
            "assortativity_r":r
        })

    df=pd.DataFrame(rows).sort_values("size",ascending=False)
    df.to_csv(R_METRICS/"community_metrics.csv",index=False,float_format="%.6f")
    print("geschrieben:", R_METRICS/"community_metrics.csv")

    # Summary of all communities
    if not df.empty:
        summary = pd.DataFrame([{
            "n_communities": len(df),
            "mean_size": df["size"].mean(),
            "median_size": df["size"].median(),
            "var_size": df["size"].var(ddof=1),
            "mean_share_female": df["share_female"].mean(),
            "median_share_female": df["share_female"].median(),
            "var_share_female": df["share_female"].var(ddof=1),
            "mean_share_mixed": df["share_mixed_known_edges"].mean(),
            "median_share_mixed": df["share_mixed_known_edges"].median(),
            "var_share_mixed": df["share_mixed_known_edges"].var(ddof=1),
            "mean_r": df["assortativity_r"].mean(),
            "median_r": df["assortativity_r"].median(),
            "var_r": df["assortativity_r"].var(ddof=1),
        }])
    else:
        summary = pd.DataFrame([{
            "n_communities":0,"mean_size":np.nan,"median_size":np.nan,"var_size":np.nan,
            "mean_share_female":np.nan,"median_share_female":np.nan,"var_share_female":np.nan,
            "mean_share_mixed":np.nan,"median_share_mixed":np.nan,"var_share_mixed":np.nan,
            "mean_r":np.nan,"median_r":np.nan,"var_r":np.nan
        }])

    summary.to_csv(R_METRICS/"community_metrics_summary.csv",index=False,float_format="%.6f")
    print("geschrieben:", R_METRICS/"community_metrics_summary.csv")
    return df, labels



def community_bridging(H, labels, top_ids):
    """
    Prüft, ob Top-Betweenness-Knoten Communitys mit unterschiedlichen
    Frauenquoten verbinden. Gibt Detail-Tabelle + Summaries zurück,
    inkl. Auswertung für "homogene Communities".
    """
    # Mapping Node -> Community
    mapping = {n: labels[i] for i, n in enumerate(H.nodes())}
    
    # Percentage of women per community
    comm_stats = {}
    for comm_id in set(labels):
        members=[n for n,l in mapping.items() if l==comm_id]
        male=sum(1 for n in members if H.nodes[n].get("gender")=="male")
        female=sum(1 for n in members if H.nodes[n].get("gender")=="female")
        share_female = female/(male+female) if (male+female)>0 else np.nan
        comm_stats[comm_id]=share_female
    
    # Detailed table: which communities are connected?
    rows=[]
    for n in top_ids:
        neigh=list(H.neighbors(n))
        comms={mapping[v] for v in neigh if v in mapping}
        if len(comms)<=1: 
            continue
        comms=list(comms)
        for i in range(len(comms)):
            for j in range(i+1,len(comms)):
                c1,c2=comms[i],comms[j]
                sf1,sf2=comm_stats.get(c1,np.nan),comm_stats.get(c2,np.nan)
                diff=abs(sf1-sf2) if (not np.isnan(sf1) and not np.isnan(sf2)) else np.nan
                rows.append({
                    "node":n,
                    "community1":c1,"community2":c2,
                    "share_female_c1":sf1,"share_female_c2":sf2,
                    "diff_share_female":diff
                })
    df=pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(R_METRICS/"community_bridging.csv",index=False,float_format="%.6f")
        print("geschrieben:", R_METRICS/"community_bridging.csv")
    else:
        print("Keine Brücken-Communitys gefunden")
    
    # Summary of all bridging pairs
    if df.empty:
        summary=pd.DataFrame([{
            "n_bridging_nodes":0,"n_bridging_pairs":0,
            "mean_diff":np.nan,"median_diff":np.nan,
            "q25_diff":np.nan,"q75_diff":np.nan,"max_diff":np.nan
        }])
    else:
        summary=pd.DataFrame([{
            "n_bridging_nodes":df["node"].nunique(),
            "n_bridging_pairs":len(df),
            "mean_diff":df["diff_share_female"].mean(),
            "median_diff":df["diff_share_female"].median(),
            "q25_diff":df["diff_share_female"].quantile(0.25),
            "q75_diff":df["diff_share_female"].quantile(0.75),
            "max_diff":df["diff_share_female"].max()
        }])
    summary.to_csv(R_METRICS/"community_bridging_summary.csv",index=False,float_format="%.6f")
    print("geschrieben:", R_METRICS/"community_bridging_summary.csv")

    # Only consider "homogeneous" communities 
    if not df.empty:
        # Homogeneous = proportion of women <0.3 or >0.7
        df_hom = df.dropna(subset=["share_female_c1","share_female_c2"]).copy()
        df_hom = df_hom[((df_hom["share_female_c1"]<0.3)|(df_hom["share_female_c1"]>0.7)) &
                        ((df_hom["share_female_c2"]<0.3)|(df_hom["share_female_c2"]>0.7))]
        if not df_hom.empty:
            summary_hom = pd.DataFrame([{
                "n_bridging_nodes":df_hom["node"].nunique(),
                "n_bridging_pairs":len(df_hom),
                "mean_diff":df_hom["diff_share_female"].mean(),
                "median_diff":df_hom["diff_share_female"].median(),
                "q25_diff":df_hom["diff_share_female"].quantile(0.25),
                "q75_diff":df_hom["diff_share_female"].quantile(0.75),
                "max_diff":df_hom["diff_share_female"].max()
            }])
        else:
            summary_hom = pd.DataFrame([{
                "n_bridging_nodes":0,"n_bridging_pairs":0,
                "mean_diff":np.nan,"median_diff":np.nan,
                "q25_diff":np.nan,"q75_diff":np.nan,"max_diff":np.nan
            }])
        summary_hom.to_csv(R_METRICS/"community_bridging_homogeneous_summary.csv",index=False,float_format="%.6f")
        print("geschrieben:", R_METRICS/"community_bridging_homogeneous_summary.csv")
    else:
        summary_hom=pd.DataFrame([{
            "n_bridging_nodes":0,"n_bridging_pairs":0,
            "mean_diff":np.nan,"median_diff":np.nan,
            "q25_diff":np.nan,"q75_diff":np.nan,"max_diff":np.nan
        }])
    return df, summary, summary_hom


def main():
    G,_,_ = load_graph()
    H,metrics = compute_global_metrics(G)
    if H.number_of_nodes()==0:
        print("Keine LCC – Analyse übersprungen."); return

    top_ids,df_btw = compute_betweenness(H)
    df_removal = removal_experiment(H, top_ids)
    df_ego_top,df_ego_ref,summary = ego_analysis(H, top_ids)
    df_comm, labels = community_analysis(H)
    df_bridge, summary_bridge = community_bridging(H, labels, top_ids)
    print("Analyse abgeschlossen.")

if __name__=="__main__":
    main()
