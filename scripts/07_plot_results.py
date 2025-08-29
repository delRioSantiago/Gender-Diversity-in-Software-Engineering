import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path("./data/results")
FIGDIR = Path("./outputs/figures")

def _ensure_dirs():
    FIGDIR.mkdir(parents=True, exist_ok=True)

def plot_ego_size_boxplot():
    top = pd.read_csv(RESULTS / "ego" / "ego_top.csv")
    ref = pd.read_csv(RESULTS / "ego" / "ego_ref.csv")
    data = [ref["ego_size"].dropna(), top["ego_size"].dropna()]
    plt.figure()
    plt.boxplot(data, tick_labels=["ref", "top5"], showfliers=False)
    plt.ylabel("Ego-Größe (Anzahl Nachbar:innen)")
    plt.title("Ego-Größen: Top-5% vs. Referenz")
    plt.tight_layout()
    plt.savefig(FIGDIR / "ego_size_boxplot.png", dpi=200)
    plt.close()

def plot_ego_female_share_boxplot():
    top = pd.read_csv(RESULTS / "ego" / "ego_top.csv")
    ref = pd.read_csv(RESULTS / "ego" / "ego_ref.csv")
    # nur bekannte Nachbar:innen bereits in female_share berücksichtigt
    data = [ref["female_share"].dropna(), top["female_share"].dropna()]
    plt.figure()
    plt.boxplot(data, tick_labels=["ref", "top5"], showfliers=False)
    plt.ylabel("Frauenanteil im Ego-Netz")
    plt.title("Frauenanteil im Ego: Top-5% vs. Referenz")
    plt.tight_layout()
    plt.savefig(FIGDIR / "ego_female_share_boxplot.png", dpi=200)
    plt.close()

def plot_removal_assortativity_with_table():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv(RESULTS / "removal" / "removal_experiment.csv")
    row = df.iloc[0]
    r_before = row["r_before"]
    r_after_top = row["r_after_top"]
    r_rand_mean = row["rand_mean"]
    q025 = row["rand_q025"]
    q975 = row["rand_q975"]
    delta_r = r_after_top - r_before

    labels = ["vorher", "Top-Removal", "Random-Ø"]
    vals = [r_before, r_after_top, r_rand_mean]

    fig, ax = plt.subplots()
    bars = ax.bar(range(3), vals, tick_label=labels)
    # Fehlerbalken Random
    yerr = np.array([[r_rand_mean - q025], [q975 - r_rand_mean]])
    ax.errorbar(2, r_rand_mean, yerr=yerr, fmt="none", capsize=5, color="black")

    ax.set_ylabel("Assortativität r")
    ax.set_title("Assortativität: Top-Removal vs. Random")

    # Tabelle mit Kennzahlen einfügen
    cell_text = [[f"{r_before:.3f}", f"{r_after_top:.3f}", f"{delta_r:+.3f}", f"p={row['p_value']:.4f}"]]
    col_labels = ["r_before", "r_after_top", "Δr", "Permutation p"]
    table = plt.table(cellText=cell_text,
                      colLabels=col_labels,
                      loc="bottom",
                      cellLoc="center",
                      bbox=[0.0, -0.3, 1, 0.2])  # [left, bottom, width, height]

    plt.subplots_adjust(bottom=0.25)  # Platz für Tabelle
    plt.tight_layout()
    plt.savefig(FIGDIR / "removal_assortativity_with_table.png", dpi=200)
    plt.close()


def plot_community_share_female_boxplot():
    df = pd.read_csv(RESULTS / "metrics" / "community_metrics.csv")
    vals = df["share_female"].dropna()
    plt.figure()
    plt.boxplot([vals], tick_labels=["Communities"], showfliers=False)
    plt.ylabel("Frauenanteil je Community")
    plt.title("Verteilung der Frauenanteile über Communities")
    plt.tight_layout()
    plt.savefig(FIGDIR / "community_share_female_boxplot.png", dpi=200)
    plt.close()

def plot_community_assortativity_hist():
    df = pd.read_csv(RESULTS / "metrics" / "community_metrics.csv")
    vals = df["assortativity_r"].dropna()
    plt.figure()
    plt.hist(vals, bins=20)
    plt.xlabel("Assortativität r (Community-intern)")
    plt.ylabel("Häufigkeit")
    plt.title("Assortativitätsverteilung über Communities")
    plt.tight_layout()
    plt.savefig(FIGDIR / "community_assortativity_hist.png", dpi=200)
    plt.close()

def plot_bridging_diff_hist():
    df = pd.read_csv(RESULTS / "metrics" / "community_bridging.csv")
    vals = df["diff_share_female"].dropna()
    plt.figure()
    plt.hist(vals, bins=20)
    plt.xlabel("Differenz der Frauenanteile zwischen verbundenen Communities")
    plt.ylabel("Häufigkeit")
    plt.title("Bridging-Differenzen (Top-Betweenness-Knoten)")
    plt.tight_layout()
    plt.savefig(FIGDIR / "bridging_diff_hist.png", dpi=200)
    plt.close()

def main():
    _ensure_dirs()
    plot_ego_size_boxplot()
    plot_ego_female_share_boxplot()
    plot_removal_assortativity_with_table()
    plot_community_share_female_boxplot()
    plot_community_assortativity_hist()
    plot_bridging_diff_hist()
    print(f"Fertige Plots unter {FIGDIR.resolve()}")

if __name__ == "__main__":
    main()
