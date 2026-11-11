"""verify_numbers.py: check that every decimal number in the paper traces back to an output file.

Usage:
    python scripts/verify_numbers.py PAPER [PAPER ...] --results DIR [DIR ...] [--raw RAW_RESULTS_CSV]

PAPER is the manuscript or supplement as .docx, .md or .txt. Every decimal number in the text (before the reference
list) is matched against the set of values found in the result CSV files under each DIR (recursively): each value is
also taken with its sign flipped and multiplied by 100 (percentages), and is rounded to 1-4 decimals. With --raw, the
v2.1 raw_results.csv is also aggregated as in the paper (seeds within regions, then regions; per method, outcome,
model, region and seed). Design constants (alpha = 0.10, thresholds 0.88, 0.80, clip quantiles, ...) are allowed.
Unmatched numbers are printed with their context; the exit code is the number of unmatched values.
"""
import argparse, glob, os, re, sys, zipfile
import numpy as np, pandas as pd

CONSTANTS = {0.9, 0.1, 0.88, 0.02, 0.6, 0.99, 0.95, 0.999, 0.0625, 0.05, 0.8, 0.5, 1.0, 0.0, 0.7, 0.08, 0.006, 0.01, 0.2, 6.5}


def read_text(path):
    if path.lower().endswith(".docx"):
        xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf8")
        xml = re.sub(r"</w:p>", "\n", xml)
        return re.sub(r"<[^>]+>", "", xml)
    return open(path, encoding="utf8").read()


def build_pool(dirs, raw=None):
    pool = set()
    def add(v):
        if not np.isfinite(v): return
        for x in (v, -v, 100 * v, -100 * v):
            for d in (1, 2, 3, 4): pool.add((d, round(x, d)))
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "*.csv*"), recursive=True):
            try: df = pd.read_csv(f, low_memory=False)
            except Exception: continue
            for c in df.select_dtypes("number").columns:
                for v in df[c].dropna().unique()[:300000]: add(float(v))
            for c in df.columns:
                if df[c].dtype == object:
                    for s in df[c].dropna().astype(str).unique()[:20000]:
                        for m in re.findall(r"[-+]?\d+\.\d+", s): add(float(m))
    if raw:
        R = pd.read_csv(raw, low_memory=False)
        T = R[(R.exp == "main") & (R.split == "target") & (R.variant.isin(["primary", "baseline"]))]
        cols = [c for c in T.select_dtypes("number").columns if c != "seed"]
        for keys in (["method"], ["method", "outcome"], ["method", "model"], ["method", "region"], ["method", "seed"], ["method", "region", "outcome"]):
            g = T.groupby(keys + [k for k in ["region", "seed"] if k not in keys])[cols].mean()
            for _ in [k for k in ["region", "seed"] if k not in keys]:
                g = g.groupby(level=list(range(g.index.nlevels - 1))).mean()
            for v in g.values.ravel(): add(float(v))
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("papers", nargs="+"); ap.add_argument("--results", nargs="+", required=True); ap.add_argument("--raw")
    a = ap.parse_args()
    pool = build_pool(a.results, a.raw)
    unmatched, total = [], 0
    for p in a.papers:
        body = re.split(r"\n#+ ?References|\nREFERENCES", read_text(p))[0]
        for m in re.finditer(r"(?<![\w.\[/])([−-]?\+?\d*\.\d+)%?", body):
            s = m.group(0); v = float(s.replace("−", "-").replace("%", "").replace("+", "")); total += 1
            d = len(s.replace("%", "").split(".")[1])
            if (d, round(v, d)) in pool or abs(v) in CONSTANTS: continue
            unmatched.append((p, s, body[max(0, m.start() - 70): m.start() + 20].replace("\n", " ")))
    print(f"numbers checked: {total}; unmatched: {len(unmatched)}")
    for p, s, c in unmatched: print(f"{os.path.basename(p)} | {s} | {c}")
    sys.exit(len(unmatched))


if __name__ == "__main__":
    main()
