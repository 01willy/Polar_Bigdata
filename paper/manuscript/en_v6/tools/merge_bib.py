#!/usr/bin/env python3
"""Merge paper/manuscript/en/refs_*.bib into references.bib (one entry per key).

Rules (manuscript assembly, 2026-10-05):
  * Entries with the same key that are identical after whitespace normalisation
    are merged silently.
  * If entries for a key differ, the more complete one is kept: more bibliographic
    fields first (annote/verified/note-like audit fields are not counted), then the
    longer total field text. Every such choice is written to the log.
  * KEY_ALIASES maps a duplicate key of the same record onto the kept key; the
    alias entry joins the comparison for the kept key and the alias must not be
    cited any more (the script fails if it is).
  * sn-nature.bst (template v3.1) drops the booktitle of an inproceedings entry without
    editors and stops with "can't pop an empty literal stack"; such entries are written as
    article with journal = booktitle (same bibliographic data, logged as TYPE).
  * BibTeX reads an at-sign even inside % comments, so the output header avoids it.
Usage: python3 tools/merge_bib.py  (run from paper/manuscript/en)
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "references.bib")
LOG = os.path.join(HERE, "build", "merge_bib.log")
AUDIT_FIELDS = {"annote", "verified"}
# same record under two keys (refs_legends.bib header says one key must be chosen)
KEY_ALIASES = {
    "linnenbrink2024_knndm": "Linnenbrink2024",              # refs_legends.bib header: one key must be chosen
    "MunozSabater2021": "munozsabater2021_era5_land",          # same DOI 10.5194/essd-13-4349-2021
    "gneiting2007_proper_scoring_rules": "GneitingRaftery2007",  # same DOI 10.1198/016214506000001437
    "dunn2023_hierarchical_conformal": "Dunn2023",             # same DOI 10.1080/01621459.2022.2060112
}


def parse_entries(text):
    """Return list of (type, key, body, fields) for every @type{key, ...} entry."""
    entries = []
    i = 0
    while True:
        m = re.compile(r"@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,").search(text, i)
        if not m:
            break
        depth, j = 1, m.end()
        while depth and j < len(text):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        body = text[m.end():j - 1]
        entries.append((m.group(1).lower(), m.group(2), body, parse_fields(body)))
        i = j
    return entries


def parse_fields(body):
    fields, k = {}, 0
    pat = re.compile(r"\s*([A-Za-z_-]+)\s*=\s*")
    while k < len(body):
        m = pat.match(body, k)
        if not m:
            k += 1
            continue
        name, k = m.group(1).lower(), m.end()
        if k < len(body) and body[k] == "{":
            depth, j = 1, k + 1
            while depth and j < len(body):
                depth += {"{": 1, "}": -1}.get(body[j], 0)
                j += 1
            val, k = body[k + 1:j - 1], j
        elif k < len(body) and body[k] == '"':
            j = body.index('"', k + 1)
            val, k = body[k + 1:j], j + 1
        else:
            m2 = re.compile(r"[^,\s]+").match(body, k)
            val, k = (m2.group(0), m2.end()) if m2 else ("", k + 1)
        fields[name] = " ".join(val.split())
    return fields


def score(fields):
    bib = {k: v for k, v in fields.items() if k not in AUDIT_FIELDS}
    return (len(bib), sum(len(v) for v in bib.values()))


def render(etype, key, fields):
    width = max(len(k) for k in fields)
    lines = [f"@{etype}{{{key},"]
    items = list(fields.items())
    for n, (k, v) in enumerate(items):
        comma = "," if n < len(items) - 1 else ""
        lines.append(f"  {k.ljust(width)} = {{{v}}}{comma}")
    lines.append("}")
    return "\n".join(lines)


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    files = sorted(glob.glob(os.path.join(HERE, "refs_*.bib")))
    seen, order, log = {}, [], []
    for f in files:
        for etype, key, body, fields in parse_entries(open(f, encoding="utf-8").read()):
            src = os.path.basename(f)
            kept_key = KEY_ALIASES.get(key, key)
            if kept_key != key:
                log.append(f"ALIAS  {key} ({src}) -> {kept_key}: same record, key unified")
            cand = (etype, fields, src)
            if kept_key not in seen:
                seen[kept_key] = [cand]
                order.append(kept_key)
            else:
                seen[kept_key].append(cand)
    out_entries = []
    for key in order:
        cands = seen[key]
        norm = {(c[0], tuple(sorted((k, v) for k, v in c[1].items() if k not in AUDIT_FIELDS))) for c in cands}
        best = max(cands, key=lambda c: score(c[1]))
        if len(cands) > 1:
            if len(norm) == 1:
                log.append(f"SAME   {key}: {len(cands)} copies identical except audit notes ({', '.join(c[2] for c in cands)})")
            else:
                others = [c for c in cands if c is not best]
                diff = []
                for c in others:
                    a, b = best[1], c[1]
                    ks = sorted((set(a) | set(b)) - AUDIT_FIELDS)
                    d = [k for k in ks if a.get(k) != b.get(k)]
                    diff.append(f"{c[2]} differs in {','.join(d) if d else 'type'}")
                log.append(f"DIFFER {key}: kept {best[2]} (fields {score(best[1])[0]}, chars {score(best[1])[1]}); " + "; ".join(diff))
        etype, fields = best[0], dict(best[1])
        if etype == "inproceedings" and "editor" not in fields and "booktitle" in fields:
            fields = {("journal" if k == "booktitle" else k): v for k, v in fields.items()}
            etype = "article"
            log.append(f"TYPE   {key}: inproceedings -> article, booktitle -> journal (sn-nature.bst inproceedings bug)")
        out_entries.append(render(etype, key, fields))
    # cited-key check
    cited = set()
    for tex in glob.glob(os.path.join(HERE, "sections", "*.tex")):
        for line in open(tex, encoding="utf-8"):
            line = re.sub(r"(?<!\\)%.*", "", line)
            for grp in re.findall(r"\\cite[a-z]*\{([^}]*)\}", line):
                cited.update(k.strip() for k in grp.split(","))
    missing = sorted(cited - set(order))
    uncited = sorted(set(order) - cited)
    aliases_cited = sorted(cited & set(KEY_ALIASES))
    # same record under two kept keys (same DOI) would print twice in the reference list
    by_doi = {}
    for key in order:
        doi = next((c[1].get("doi", "").lower() for c in seen[key] if c[1].get("doi")), "")
        if doi:
            by_doi.setdefault(doi, []).append(key)
    doi_dups = {d: ks for d, ks in by_doi.items() if len(ks) > 1}
    if doi_dups:
        log.append(f"DOIDUP {doi_dups} (add to KEY_ALIASES)")
    header = [
        "% references.bib : merged from refs_*.bib by tools/merge_bib.py (do not edit by hand;",
        "%   edit the refs_*.bib files and rerun). Merge log: build/merge_bib.log.",
        f"% Sources: {', '.join(os.path.basename(f) for f in files)}.",
        f"% Entries: {len(order)} unique keys.",
        "",
    ]
    open(OUT, "w", encoding="utf-8").write("\n".join(header) + "\n\n".join(out_entries) + "\n")
    log.append(f"TOTAL  {sum(len(v) for v in seen.values())} entries read, {len(order)} unique keys written")
    log.append(f"CHECK  cited keys {len(cited)}, missing in bib {missing}, uncited in bib {uncited}, alias keys still cited {aliases_cited}")
    open(LOG, "w", encoding="utf-8").write("\n".join(log) + "\n")
    print("\n".join(log))
    if missing or aliases_cited or doi_dups:
        sys.exit(1)


if __name__ == "__main__":
    main()
