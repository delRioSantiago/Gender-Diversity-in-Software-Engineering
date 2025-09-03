import pandas as pd

# Paths
FILE_IO  = "./data/raw/results_io.csv"   
FILE_COM = "./data/raw/results_com.csv"  
OUT_CSV  = "./data/raw/merged_genders.csv"


# genderapi.io-Datei
IO_COL_FIRSTNAME  = "first_name"
IO_COL_GENDER     = "gender"
IO_COL_PROB       = "probability"   

# gender-api.com-Datei
COM_COL_FIRSTNAME = "first_name"
COM_COL_GENDER    = "ga_gender"
COM_COL_ACCURACY  = "ga_accuracy"   
COM_COL_SAMPLES   = "ga_samples"    

# Parameters 
THRESHOLD   = 0.60   # Threshold value for safety in sex determination
MIN_SAMPLES = 5      # Minimum sample for genderapi.io file

MF = {"male", "female"}

def to_unit(p_series):
    p = pd.to_numeric(p_series, errors="coerce")
    return (p / 100.0).fillna(0.0)

def norm_gender(x):
    if x is None or pd.isna(x):
        return "unknown"
    t = str(x).strip().lower()
    if t == "male": return "male"
    if t == "female": return "female"
    return "unknown"

def load_io(path):
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "firstname": df[IO_COL_FIRSTNAME],
        "gender_io": df[IO_COL_GENDER].map(norm_gender),
        "prob_io":   to_unit(df[IO_COL_PROB]),
    })
    return out

def load_com(path):
    df = pd.read_csv(path)
    prob = to_unit(df[COM_COL_ACCURACY])
    small = pd.to_numeric(df[COM_COL_SAMPLES], errors="coerce").fillna(0) < MIN_SAMPLES
    prob = prob.mask(small, other=0.0)
    out = pd.DataFrame({
        "firstname": df[COM_COL_FIRSTNAME],
        "gender_com": df[COM_COL_GENDER].map(norm_gender),
        "prob_com":   prob,
    })
    return out

def decide_row(gi, pi, gc, pc):
    ci = (gi in MF) and (pi >= THRESHOLD)
    cc = (gc in MF) and (pc >= THRESHOLD)
    if ci and cc:
        return gi if gi == gc else "unknown"
    if ci and not cc:
        return gi
    if cc and not ci:
        return gc
    return "unknown"

io  = load_io(FILE_IO)
com = load_com(FILE_COM)

merged = pd.merge(io, com, on="firstname", how="outer", validate="one_to_one")

merged["gender"] = [
    decide_row(gi, pi, gc, pc)
    for gi, pi, gc, pc in zip(
        merged.get("gender_io").fillna("unknown"),
        merged.get("prob_io").fillna(0.0),
        merged.get("gender_com").fillna("unknown"),
        merged.get("prob_com").fillna(0.0),
    )
]

out = merged[["firstname", "gender"]]
out.to_csv(OUT_CSV, index=False)
print(f"geschrieben: {OUT_CSV}  | rows: {len(out)}  | known: {int((out.gender.isin(MF)).sum())}  | unknown: {int((out.gender=='unknown').sum())}")



