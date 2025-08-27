import pandas as pd
import re

# ===== Pfade und Spaltennamen (Platzhalter anpassen) =====
AUTHORS_CSV      = "./data/raw/authors_filtered.csv"    # Tabelle mit vollständigen Namen
AUTH_ID_COL      = "author_id"               # Autoren-ID-Spalte in AUTHORS_CSV
AUTH_NAME_COL    = "canonical_fullname"                    # Voller Name in AUTHORS_CSV, z.B. "Ada Lovelace"
GENDER_MAP_CSV   = "./data/raw/results_io.csv"      # Tabelle mit 'firstname','gender'
OUT_AUTHORS_CSV  = "./data/processed/authors_with_gender.csv" # Ausgabedatei

# ===== Hilfen =====
MF = {"male","female"}

def extract_firstname(fullname: str) -> str:
    """
    Holt den Vornamen aus dem vollständigen Namen ohne die Schreibweise zu normalisieren.
    Strategie: erster nicht-leerer Token bis zum ersten Leerzeichen. Satzzeichen am Rand werden entfernt,
    Bindestriche bleiben erhalten.
    """
    if pd.isna(fullname):
        return ""
    # führende/trailing Satzzeichen sanft kappen
    s = str(fullname).strip()
    # erster Token bis zum ersten Leerzeichen
    token = s.split(" ", 1)[0]
    # Randzeichen wie , ; : ( ) ' " . entfernen, Bindestrich behalten
    token = re.sub(r"^[,;:()'\"\.]+|[,;:()'\"\.]+$", "", token)
    return token

# ===== Laden =====
authors = pd.read_csv(AUTHORS_CSV)
gmap    = pd.read_csv(GENDER_MAP_CSV)

# Grundchecks
if AUTH_NAME_COL not in authors.columns:
    raise ValueError(f"In {AUTHORS_CSV} fehlt die Spalte '{AUTH_NAME_COL}'.")
if "firstname" not in gmap.columns or "gender" not in gmap.columns:
    raise ValueError(f"In {GENDER_MAP_CSV} werden die Spalten 'firstname' und 'gender' erwartet.")

# Vornamen extrahieren
authors = authors.copy()
authors["firstname"] = authors[AUTH_NAME_COL].apply(extract_firstname)

# Merge auf exakt passenden Vornamen (keine Normalisierung)
merged = authors.merge(gmap[["firstname","gender"]], on="firstname", how="left")

# Unknown auffüllen
merged["gender"] = merged["gender"].where(merged["gender"].isin(MF), other="unknown").fillna("unknown")

# Ausgabe minimieren auf gewünschte Felder, aber die ID und der volle Name bleiben zur Rückverfolgbarkeit dabei
cols = [c for c in [AUTH_ID_COL, AUTH_NAME_COL, "firstname", "gender"] if c in merged.columns]
merged[cols].to_csv(OUT_AUTHORS_CSV, index=False)

known = int(merged["gender"].isin(MF).sum())
total = len(merged)
print(f"geschrieben: {OUT_AUTHORS_CSV} | Zeilen: {total} | zugewiesen: {known} | unknown: {total-known}")
