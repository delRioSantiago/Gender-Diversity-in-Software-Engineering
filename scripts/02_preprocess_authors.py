
"""
03_preprocess_authors.py

Preprocess authors & authorship for co-authorship networks.

Inputs (from fetch step):
- authors.csv        (author_id, canonical_fullname)
- authorship.csv     (paper_id, author_id)
- papers.csv         (paper_id, doi, title, year, conference_key)   [optional]

Outputs:
- authors_filtered.csv
- authorship_filtered.csv
- papers_filtered.csv              (only if papers.csv exists)
- preprocess_summary.json          (counts & drop reasons incl. single-author drops)

Filtering (as discussed):
- Drop names with only one token (mononyms).
- Drop names where the *second character* in the full string is '.' or ' ' (simple initial heuristic).
- Digits in names are allowed.
- No organization filter.

Then:
- Restrict authorship to remaining authors.
- Drop papers with exactly one remaining author (count how many).
- Count authors who appear only on such single-author papers (and thus drop out).
- Optionally drop isolates (default: drop).

Usage:
  python 03_preprocess_authors.py --data-dir ./data [--keep-isolates]
"""

import os, csv, json, argparse, pandas as pd

from collections import defaultdict

def tokenize_name(fullname: str):
    return [tok for tok in fullname.strip().split() if tok]

def should_keep_author(fullname: str) -> bool:
    toks = tokenize_name(fullname)
    if len(toks) < 2:
        return False
    if len(fullname) > 1 and fullname[1] in {'.', ' '}:
        return False
    return True  # digits allowed

def load_authors(path):
    authors = {}
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            authors[row['author_id']] = row.get('canonical_fullname', '').strip()
    return authors

def load_authorship(path):
    pairs = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            pairs.append((row['paper_id'], row['author_id']))
    return pairs

def load_papers_optional(path):
    papers = {}
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return papers
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            papers[row['paper_id']] = row  # pass through
    return papers

def write_csv(path, header, rows_iter):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows_iter:
            w.writerow(row)

def extract_first_name(data_dir="./data/raw", input_csv="authors_filtered.csv",  out_csv="firstnames.csv"):
    authors = pd.read_csv(os.path.join(data_dir, input_csv))
    if 'canonical_fullname' not in authors.columns:
        raise ValueError(f"Expected 'canonical_fullname' column in {input_csv}")
    fn = (authors["canonical_fullname"].astype(str)
                                .str.strip()
                                .str.split().str[0])
    df = pd.DataFrame({"first_name": fn})
    df = df[ df["first_name"].ne("") ].drop_duplicates().sort_values("first_name")
    df.to_csv(os.path.join(data_dir, out_csv), index=False)
    print(f"Exported {len(df)} unique first names -> {os.path.join(data_dir, out_csv)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='./data/raw')
    ap.add_argument('--keep-isolates', action='store_true')
    args = ap.parse_args()

    data_dir = args.data_dir
    authors_path = os.path.join(data_dir, 'authors.csv')
    authorship_path = os.path.join(data_dir, 'authorship.csv')
    papers_path = os.path.join(data_dir, 'papers.csv')

    authors = load_authors(authors_path)
    authorship = load_authorship(authorship_path)
    papers = load_papers_optional(papers_path)

    # Step 1: Name filter
    keep_by_name = set(aid for aid, name in authors.items() if should_keep_author(name))
    drop_name_mononym = sum(1 for aid, name in authors.items() if len(tokenize_name(name)) < 2)
    drop_name_second_char = sum(1 for aid, name in authors.items() if len(name) > 1 and name[1] in {'.',' '})

    # Step 2: Filter authorship by keep set
    kept_pairs = [(pid, aid) for (pid, aid) in authorship if aid in keep_by_name]

    # Step 3: Build paper -> authors (after name filter)
    paper_to_authors = defaultdict(set)
    for pid, aid in kept_pairs:
        paper_to_authors[pid].add(aid)

    # Count & drop single-author papers
    single_author_papers = {pid for pid, s in paper_to_authors.items() if len(s) == 1}
    num_dropped_single_author_papers = len(single_author_papers)

    # Authors appearing on single-author papers
    authors_on_single = set()
    for pid in single_author_papers:
        authors_on_single |= paper_to_authors[pid]

    # Authors appearing on multi-author papers
    authors_on_multi = set()
    for pid, s in paper_to_authors.items():
        if len(s) >= 2:
            authors_on_multi |= s

    # Authors dropped because they occur only on single-author papers
    authors_dropped_due_to_single_only = authors_on_single - authors_on_multi
    num_authors_dropped_due_to_single_only = len(authors_dropped_due_to_single_only)

    # Keep only papers with >=2 authors
    kept_papers = {pid for pid, s in paper_to_authors.items() if len(s) >= 2}
    kept_pairs = [(pid, aid) for (pid, aid) in kept_pairs if pid in kept_papers]

    # Step 4: Drop isolates (unless kept)
    if not args.keep_isolates:
        authors_in_edges = {aid for _, aid in kept_pairs}
        keep_author_id = authors_in_edges
    else:
        keep_author_id = keep_by_name

    # Step 5: Write outputs
    out_authors = os.path.join(data_dir, 'authors_filtered.csv')
    out_authorship = os.path.join(data_dir, 'authorship_filtered.csv')
    out_papers = os.path.join(data_dir, 'papers_filtered.csv')

    write_csv(out_authors, ['author_id', 'canonical_fullname'],
              ((aid, authors[aid]) for aid in sorted(keep_author_id)))
    write_csv(out_authorship, ['paper_id', 'author_id'],
              ((pid, aid) for (pid, aid) in kept_pairs))

    if papers:
        header = list(next(iter(papers.values())).keys())
        rows = (papers[pid] for pid in kept_papers if pid in papers)
        write_csv(out_papers, header, ([row.get(col, '') for col in header] for row in rows))

    # Step 6: Summary
    summary = {
        'input_counts': {
            'authors.csv': len(authors),
            'authorship.csv': len(authorship),
            'papers.csv': len(papers) if papers else 0
        },
        'filters': {
            'drop_mononym_by_name': drop_name_mononym,
            'drop_second_char_is_dot_or_space': drop_name_second_char,
            'drop_isolates': not args.keep_isolates,
            'dropped_single_author_papers': num_dropped_single_author_papers,
            'authors_dropped_due_to_single_author_papers': num_authors_dropped_due_to_single_only
        },
        'output_counts': {
            'authors_filtered.csv': len(keep_author_id),
            'authorship_filtered.csv': len(kept_pairs),
            'papers_filtered.csv': len(kept_papers) if papers else 0
        }
    }
    with open(os.path.join(data_dir, 'preprocess_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('Done. Wrote:')
    print(' -', out_authors)
    print(' -', out_authorship)
    if papers:
        print(' -', out_papers)
    print('See preprocess_summary.json for counts.')
    print(f'Dropped single-author papers: {num_dropped_single_author_papers}')
    print(f'Authors dropped due to single-author-only participation: {num_authors_dropped_due_to_single_only}')
    extract_first_name(data_dir)

if __name__ == '__main__':
    main()
