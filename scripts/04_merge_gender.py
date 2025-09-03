import pandas as pd
import re

# Column names
AUTHORS_CSV      = "./data/raw/authors_filtered.csv"    # Table with full names
AUTH_ID_COL      = "author_id"               # Author ID column in AUTHORS_CSV
AUTH_NAME_COL    = "canonical_fullname"                    # Full name in AUTHORS_CSV, e.g. "Ada Lovelace"
GENDER_MAP_CSV   = "./data/raw/results_io.csv"      # Table with 'firstname','gender'
OUT_AUTHORS_CSV  = "./data/processed/authors_with_gender.csv" # Output file

# Hilfen
MF = {"male","female"}

def extract_firstname(fullname: str) -> str:
    """
    Holt den Vornamen aus dem vollständigen Namen ohne die Schreibweise zu normalisieren.
    Strategie: erster nicht-leerer Token bis zum ersten Leerzeichen. Satzzeichen am Rand werden entfernt,
    Bindestriche bleiben erhalten.
    """
    if pd.isna(fullname):
        return ""
    # Cut leading/trailing punctuation marks
    s = str(fullname).strip()
    # first token up to the first space
    token = s.split(" ", 1)[0]
    # Remove marginal characters such as , ; : ( ) ' " . keep hyphen
    token = re.sub(r"^[,;:()'\"\.]+|[,;:()'\"\.]+$", "", token)
    return token

# Load
authors = pd.read_csv(AUTHORS_CSV)
gmap    = pd.read_csv(GENDER_MAP_CSV)

# Basic checks
if AUTH_NAME_COL not in authors.columns:
    raise ValueError(f"In {AUTHORS_CSV} fehlt die Spalte '{AUTH_NAME_COL}'.")
if "firstname" not in gmap.columns or "gender" not in gmap.columns:
    raise ValueError(f"In {GENDER_MAP_CSV} werden die Spalten 'firstname' und 'gender' erwartet.")

# Extract first names
authors = authors.copy()
authors["firstname"] = authors[AUTH_NAME_COL].apply(extract_firstname)

# Merge to exactly matching first names (no normalization)
merged = authors.merge(gmap[["firstname","gender"]], on="firstname", how="left")

# Fill up with Unknown
merged["gender"] = merged["gender"].where(merged["gender"].isin(MF), other="unknown").fillna("unknown")

# Minimize output to desired fields, but keep the ID and full name for traceability
cols = [c for c in [AUTH_ID_COL, AUTH_NAME_COL, "firstname", "gender"] if c in merged.columns]
merged[cols].to_csv(OUT_AUTHORS_CSV, index=False)

known = int(merged["gender"].isin(MF).sum())
total = len(merged)
print(f"geschrieben: {OUT_AUTHORS_CSV} | Zeilen: {total} | zugewiesen: {known} | unknown: {total-known}")
