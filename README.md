# Gender Diversity in Software Engineering Co-Authorship Networks

Dieses Projekt analysiert Co-Autor:innen-Netzwerke der Konferenzen **ICSE, ASE und FSE (2015–2024)**.  
Ziel ist es, den Zusammenhang zwischen **Betweenness Centrality** und **Geschlechterdiversität** zu untersuchen.

Die Pipeline besteht aus 6 Schritten (`01_load-dblp.py` bis `06_analysis.py`) und erzeugt strukturierte Zwischendateien sowie Ergebnis-Tabellen.

---

## Setup

### 1. Conda Environment
conda env create -f environment.yml
conda activate gender-diversity

### 2. Daten vorbereiten

Lade das aktuelle DBLP-XML-Dump herunter:
https://dblp.org/xml/
 → dblp.xml.gz

Lege die Datei in ./data/raw/ ab.

### 3. Gender API Zwischenschritt

Die Gender-Inferenz erfolgt mit externen APIs (gender-api.io, gender-api.com).
Das heißt:

Aus 02_preprocess_authors.py wird eine firstnames.csv erzeugt.

Diese Datei musst du manuell bei den Gender-APIs hochladen.

Speichere die Ergebnisse als:

./data/raw/results_io.csv

./data/raw/results_com.csv

Danach geht es mit 03_gender_api.py weiter.

# Pipline  

### Schritt 1: DBLP-Daten parsen
python 01_load-dblp.py

### Schritt 2: Autoren vorverarbeiten
python 02_preprocess_authors.py --data-dir ./data/raw
--> jetzt "firstnames.csv" hochladen bei gender-api.io & gender-api.com
--> Ergebnisse speichern als results_io.csv und results_com.csv

### Schritt 3: Gender-Ergebnisse mergen
python 03_gender_api.py

### Schritt 4: Gender in Autorenliste einfügen
python 04_merge_gender.py

### Schritt 5: Graph aufbauen
python 05_graph_build.py

### Schritt 6: Analysen (global, ego, community, bridging)
python 06_analysis.py

## Outputs & Spalten (Dictionary)

Alle Ergebnisse liegen in `./data/results/` in Unterordnern.

### Global / LCC (metrics/)
- **metrics_global.csv**
  - `scope`: global oder lcc  
  - `nodes`, `edges`: Größe  
  - `male`, `female`, `unknown`: absolute Zahlen  
  - `share_female`, `share_unknown`: Anteile  
  - `share_mixed_known_edges`: Anteil gemischtgeschlechtlicher Kanten  
  - `assortativity_r`: Gender-Assortativität (Newman’s r)

- **community_metrics.csv**
  - `community`: Community-ID (Leiden)  
  - `size`: Anzahl Knoten  
  - `male`, `female`, `unknown`: absolute Zahlen  
  - `share_female`: Anteil Frauen an bekannten Geschlechtern  
  - `share_mixed_known_edges`: Anteil gemischtgeschlechtlicher Kanten  
  - `assortativity_r`: Gender-Assortativität innerhalb der Community  

- **community_metrics_summary.csv**
  - `n_communities`: Gesamtzahl Communities  
  - `mean_size`, `median_size`, `var_size`  
  - `mean_share_female`, `median_share_female`, `var_share_female`  
  - `mean_share_mixed`, `median_share_mixed`, `var_share_mixed`  
  - `mean_r`, `median_r`, `var_r`

### Centrality (centrality/)
- **betweenness_lcc.csv**
  - `author_id`: eindeutige ID  
  - `betweenness`: approximierte Betweenness Centrality  
  - `is_top5_betweenness`: True/False, ob in Top-5%  

### Removal-Experimente (removal/)
- **removal_experiment.csv**
  - `r_before`: Assortativität vor Entfernen  
  - `r_after_top`: nach Entfernen Top-5% Betweenness  
  - `r_after_degree`: nach Entfernen Top-Degree  
  - `r_after_closeness`: nach Entfernen Top-Closeness  
  - `rand_mean`, `rand_std`: Mittelwert/Std der Random-Removals  
  - `rand_q025`, `rand_q975`: 2.5%- und 97.5%-Quantile der Random-Removals  
  - `p_value`: Permutationstest für Effekt der Top-Knoten  
  - `removed_k`: Anzahl entfernter Knoten  
  - `n_random`: Anzahl Random-Runs  

- **removal_experiment_samples.csv**
  - `r_random`: alle r-Werte der einzelnen Random-Removals  

### Ego-Analysen (ego/)
- **ego_top.csv**
  - pro Top-5%-Knoten: `author_id`, `ego_size`, `female_share`, `known_neighbors`  

- **ego_ref.csv**
  - dieselben Kennzahlen für eine gleich große Zufalls-Referenzgruppe  

- **ego_summary.csv**
  - `group`: top5 oder ref  
  - `n_nodes`: Anzahl untersuchter Egos  
  - `ego_size_mean`: Durchschnittliche Ego-Größe  
  - `female_share_mean`: Durchschnittlicher Frauenanteil  

### Community-Bridging (metrics/)
- **community_bridging.csv**
  - `node`: Top-Knoten (Betweenness)  
  - `community1`, `community2`: IDs der verbundenen Communities  
  - `share_female_c1`, `share_female_c2`: Frauenanteile in den Communities  
  - `diff_share_female`: absolute Differenz  

- **community_bridging_summary.csv**
  - `n_bridging_nodes`: Anzahl Top-Knoten mit Bridging  
  - `n_bridging_pairs`: Anzahl Community-Paare  
  - `mean_diff`, `median_diff`, `q25_diff`, `q75_diff`, `max_diff`  

- **community_bridging_homogeneous_summary.csv**
  - dieselben Kennzahlen wie oben, aber nur für Paare „homogener“ Communities (Frauenquote <0.3 oder >0.7)
