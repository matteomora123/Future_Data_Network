import argparse
from pathlib import Path
import re

def prefix_dir(name: str):
    m = re.search(r'(\d+)$', name)
    return (name[:m.start()] + "#") if m else None

def compress_ranges(nums):
    if not nums: return ""
    nums = sorted(nums)
    ranges, s, p = [], nums[0], nums[0]
    for x in nums[1:]:
        if x == p + 1:
            p = x
        else:
            ranges.append((s, p)); s = p = x
    ranges.append((s, p))
    parts = [f"{a}" if a == b else f"{a}–{b}" for a,b in ranges]
    return "[" + ",".join(parts) + "]"

def group_dirs(dirs):
    groups, singles = {}, []
    for d in dirs:
        pref = prefix_dir(d.name)
        if pref: groups.setdefault(pref, []).append(d)
        else: singles.append(d)
    return groups, singles

def group_files(files):
    groups, singles = {}, []
    for f in files:
        m = re.search(r'(\d+)(?=\.[^.]+$)|(\d+)$', f.name)
        if m:
            key = re.sub(r'(\d+)(?=\.[^.]+$)|(\d+)$', '#', f.name).lower()
            groups.setdefault(key, []).append(f)
        else:
            singles.append(f)
    return groups, singles

def dir_prefix(level):
    return "│   " * (level - 1) + ("├── " if level > 0 else "")

def summarize_dir_sample(d: Path, prefix: str, sample_items: int):
    try:
        kids = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except Exception:
        print(prefix + "es.: [accesso non disponibile]")
        return
    subdirs = [k.name for k in kids if k.is_dir()]
    files   = [k for k in kids if k.is_file()]
    f_groups, f_singles = group_files(files)

    subdirs_txt = ", ".join(subdirs[:sample_items])
    if len(subdirs) > sample_items: subdirs_txt += f", ... +{len(subdirs)-sample_items}"
    files_txt_parts = []
    for k, lst in sorted(f_groups.items()):
        files_txt_parts.append(f"{k} ×{len(lst)}")
    for f in sorted(f_singles, key=lambda p: p.name.lower())[:sample_items]:
        files_txt_parts.append(f.name)
    if f_singles and len(f_singles) > sample_items:
        files_txt_parts.append(f"... +{len(f_singles)-sample_items}")
    files_txt = "; ".join(files_txt_parts) if files_txt_parts else "—"
    print(prefix + f"es. {d.name}: sottocartelle: {subdirs_txt or '—'}; file: {files_txt}")

def scan(root: Path, level=0, sample_items=4):
    pref = dir_prefix(level)
    if level == 0:
        print(root)
    else:
        print(f"{pref}[DIR] {root.name}")

    try:
        kids = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        print(pref + "accesso negato")
        return

    dirs  = [k for k in kids if k.is_dir()]
    files = [k for k in kids if k.is_file()]

    # 1) stampa gruppi di cartelle numerate (campione + molteplicità + campione contenuto)
    gdirs, sdirs = group_dirs(dirs)
    for key in sorted(gdirs.keys()):
        lst = sorted(gdirs[key], key=lambda p: p.name.lower())
        # range indici
        nums = []
        for d in lst:
            m = re.search(r'(\d+)$', d.name)
            if m: nums.append(int(m.group(1)))
        idx = compress_ranges(nums)
        print(f"{pref}{key}{idx} ×{len(lst)}")
        summarize_dir_sample(lst[0], pref, sample_items)

    # 2) stampa file di questa cartella raggruppati (solo molteplicità e campione)
    f_groups, f_singles = group_files(files)
    for k, lst in sorted(f_groups.items()):
        sample = ", ".join(sorted([p.name for p in lst])[:sample_items])
        more = f", ... +{len(lst)-sample_items}" if len(lst) > sample_items else ""
        print(f"{pref}{k} ×{len(lst)} [{sample}{more}]")
    for f in sorted(f_singles, key=lambda p: p.name.lower())[:sample_items]:
        print(f"{pref}{f.name}")
    if len(f_singles) > sample_items:
        print(f"{pref}... +{len(f_singles)-sample_items} altri file")

    # 3) ricorsione: entra in TUTTE le cartelle non numerate; per i gruppi numerati evita espansione completa
    for d in sdirs:
        scan(d, level+1, sample_items=sample_items)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--sample-items", type=int, default=4, help="quanti esempi mostrare per contenuto")
    args = ap.parse_args()
    if not args.root.exists():
        raise FileNotFoundError(args.root)
    scan(args.root, sample_items=args.sample_items)

if __name__ == "__main__":
    main()
