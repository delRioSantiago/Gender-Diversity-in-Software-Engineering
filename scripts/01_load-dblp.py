
import os, gzip, csv, uuid, html, shutil
from lxml import etree

# Paths & Outputs
DATA_DIR = os.environ.get("DBLP_DATA_DIR", "./data/raw")  
GZ_FILE = os.path.join(DATA_DIR, "dblp.xml.gz")
XML_FILE = os.path.join(DATA_DIR, "dblp.xml")

# Intermediates (temp; auto-deleted at end)
TMP_DIR = os.path.join(DATA_DIR, "_tmp_fetch")
ALIAS_PAPER  = os.path.join(TMP_DIR, "alias_paper.csv")
HOMEPAGES_MAP= os.path.join(TMP_DIR, "homepages_map.csv")

# Final outputs (only these three remain on disk)
PAPERS       = os.path.join(DATA_DIR, "papers.csv")
AUTHORS      = os.path.join(DATA_DIR, "authors.csv")
AUTHORSHIP   = os.path.join(DATA_DIR, "authorship.csv")

# Config
START_YEAR = int(os.environ.get("DBLP_START_YEAR", "2015"))
END_YEAR   = int(os.environ.get("DBLP_END_YEAR",   "2024"))

VENUE_NAMES = {
    "icse": ["icse", "international conference on software engineering"],
    "ase":  ["ase", "automated software engineering"],
    "fse":  ["fse", "foundations of software engineering", "esec/fse"]
}

# Helpers
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def ensure_unzipped():
    ensure_dir(DATA_DIR)
    if os.path.exists(XML_FILE) and os.path.getsize(XML_FILE) > 0:
        return XML_FILE
    if not os.path.exists(GZ_FILE) or os.path.getsize(GZ_FILE) == 0:
        raise FileNotFoundError(f"dblp.xml.gz nicht gefunden/leer unter {GZ_FILE}")
    with gzip.open(GZ_FILE, "rb") as fin, open(XML_FILE, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return XML_FILE

def lname(elem):
    return etree.QName(elem).localname if isinstance(elem.tag, str) else ""

def findtext_local(elem, name):
    for c in elem:
        if lname(c) == name:
            return (c.text or "")
    return None

def findall_local(elem, name):
    return [c for c in elem if lname(c) == name]

def normalize(t: str) -> str:
    return html.unescape((t or "").strip())

def deterministic_fallback_id(fullname: str) -> str:
    return f"custom_{uuid.uuid5(uuid.NAMESPACE_URL, fullname).hex[:12]}"

def extract_doi_from_text(t: str):
    if not t: return None
    tt = t.strip(); low = tt.lower()
    if low.startswith("https://doi.org/"): return tt.split("https://doi.org/", 1)[1]
    if low.startswith("http://doi.org/"):  return tt.split("http://doi.org/", 1)[1]
    if low.startswith("doi:"):             return tt[4:]
    if "/" in tt and " " not in tt and ":" not in tt and tt.count("/") >= 1: return tt
    return None

def extract_first_doi(elem):
    for ee in findall_local(elem, "ee"):
        doi = extract_doi_from_text(ee.text or "")
        if doi: return doi
    url = findtext_local(elem, "url") or ""
    if "doi.org/" in url: return url.split("doi.org/", 1)[1]
    return None

def classify_venue(key: str, booktitle: str):
    kl = (key or "").lower()
    bt = (booktitle or "").lower()
    if kl.startswith("conf/icse/"): return "icse"
    if kl.startswith(("conf/kbse/", "conf/ase/")): return "ase"
    if kl.startswith(("conf/fse/", "conf/sigsoft/", "conf/esec/")): return "fse"
    if any(x in bt for x in VENUE_NAMES["icse"]): return "icse"
    if any(x in bt for x in VENUE_NAMES["ase"]): return "ase"
    if any(x in bt for x in VENUE_NAMES["fse"]): return "fse"
    return "other"

def is_target_inproceedings(elem) -> bool:
    y = findtext_local(elem, "year")
    if not (y and y.isdigit()): return False
    iy = int(y)
    if not (START_YEAR <= iy <= END_YEAR): return False
    key = elem.get("key", "") or ""
    booktitle = (findtext_local(elem, "booktitle") or "")
    venue = classify_venue(key, booktitle)
    return venue in {"icse", "ase", "fse"}

def clear_elem(elem):
    elem.clear()
    p = elem.getparent()
    if p is not None:
        while elem.getprevious() is not None:
            del p[0]

def normalize_homepage_id(hp_key: str) -> str:
    if not hp_key: return ""
    if hp_key.startswith("homepages/"): return hp_key[len("homepages/"):]
    return hp_key


# Phase 1: Stream parse & write PAPERS + temp ALIAS_PAPER/HOMEPAGES_MAP
def stream_build(xml_path: str):
    ensure_dir(TMP_DIR)
    # final papers.csv directly
    with open(PAPERS, "w", newline="", encoding="utf-8") as f_papers,          open(ALIAS_PAPER, "w", newline="", encoding="utf-8") as f_ap,          open(HOMEPAGES_MAP, "w", newline="", encoding="utf-8") as f_hp:

        wp = csv.writer(f_papers); wp.writerow(["paper_id","doi","title","year","conference_key"])
        wap = csv.writer(f_ap);    wap.writerow(["paper_id","alias"])
        whp = csv.writer(f_hp);    whp.writerow(["alias","homepage_id"])

        paper_seq = 0
        inproceedings_count = 0
        homepages_count = 0

        ctx = etree.iterparse(
            xml_path,
            events=("end",),
            tag=("inproceedings","www"),
            load_dtd=False,
            resolve_entities=False,
            recover=True,
            huge_tree=True
        )
        for _, elem in ctx:
            tag = lname(elem)
            if tag == "inproceedings" and is_target_inproceedings(elem):
                inproceedings_count += 1
                if inproceedings_count % 2000 == 0:
                    print(f"{inproceedings_count} <inproceedings> verarbeitet...")
                key = elem.get("key","")
                year = int(findtext_local(elem, "year"))
                title = normalize(findtext_local(elem, "title") or "")
                doi = extract_first_doi(elem)
                conf_key = classify_venue(key, findtext_local(elem, "booktitle") or "")
                paper_id = f"paper_{paper_seq}"; paper_seq += 1
                wp.writerow([paper_id, doi, title, year, conf_key])
                for a in findall_local(elem, "author"):
                    if not a.text: continue
                    alias = normalize(a.text)
                    wap.writerow([paper_id, alias])
            elif tag == "www":
                key = elem.get("key","")
                title_text = (findtext_local(elem, "title") or "")
                if key.startswith("homepages/") and title_text == "Home Page":
                    homepages_count += 1
                    if homepages_count % 200000 == 0:
                        print(f"{homepages_count} homepages verarbeitet...")
                    for a in findall_local(elem, "author"):
                        if not a.text: continue
                        alias = normalize(a.text)
                        whp.writerow([alias, normalize_homepage_id(key)])
            clear_elem(elem)
        del ctx


# Phase 2: Build AUTHORS + AUTHORSHIP, scoped to aliases in papers
def materialize_final_from_temp():
    print("Materializing authors/authorship from temp...")

    # Aliases appearing in papers
    alias_from_papers = set()
    with open(ALIAS_PAPER, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            alias_from_papers.add(row["alias"])

    # Map aliases -> homepage_id (first seen)
    alias_to_homepage = {}
    with open(HOMEPAGES_MAP, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            a = row["alias"]
            if a in alias_from_papers and a not in alias_to_homepage:
                alias_to_homepage[a] = row["homepage_id"]

    # Build alias -> author_id (homepage id or deterministic fallback)
    def author_id_for(alias: str) -> str:
        hp = alias_to_homepage.get(alias)
        return hp if hp else deterministic_fallback_id(alias)

    # AUTHORS (unique author_ids, keep canonical_fullname=alias)
    authors_seen = set()
    with open(AUTHORS, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out); w.writerow(["author_id","canonical_fullname"])
        for alias in sorted(alias_from_papers):
            aid = author_id_for(alias)
            if aid in authors_seen: continue
            w.writerow([aid, alias])
            authors_seen.add(aid)

    # AUTHORSHIP (paper_id, author_id) with dedup
    with open(AUTHORSHIP, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out); w.writerow(["paper_id","author_id"])
        seen = set()
        with open(ALIAS_PAPER, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                alias = row["alias"]
                if alias not in alias_from_papers:
                    continue
                pid = row["paper_id"]
                aid = author_id_for(alias)
                key = (pid, aid)
                if key in seen: continue
                w.writerow([pid, aid])
                seen.add(key)

def cleanup_tmp():
    try:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
    except Exception as e:
        print("WARN: tmp cleanup failed:", e)
        
def _count_lines(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return -1

# Main
def main():
    ensure_unzipped()
    stream_build(XML_FILE)
    materialize_final_from_temp()
    cleanup_tmp()
    print("[STATS] papers.csv     :", _count_lines(PAPERS))
    print("[STATS] authors.csv    :", _count_lines(AUTHORS))
    print("[STATS] authorship.csv :", _count_lines(AUTHORSHIP))
    print("Fertig. Gespeichert wurden: papers.csv, authors.csv, authorship.csv.")

if __name__ == "__main__":
    main()
