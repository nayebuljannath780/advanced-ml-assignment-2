#!/usr/bin/env python3
"""Assignment 02 G1/G2 data provenance, duplicate-review, fixed split gate.

Research scope: 4 models x 2 independent rice-leaf image classification datasets.
NO image augmentation, NO model training, NO test-set evaluation.
D1 is Mendeley hx6f852hw4 v2, using only Original photographs.
D2 is Mendeley fwcj7stb8r v2, using the user-mounted original archive.

Dataset sources MUST be verified and manually attached/mounted by the user.
Outputs are evidence and deterministic split manifests; not an assertion of dataset independence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def normalize_label(s: str) -> str:
    return ''.join(c for c in s.lower() if c.isalnum())

CLASS_NAMES = {
    'D1': ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf', 'Leaf Blast',
           'Leaf Scald', 'Narrow Brown Leaf Spot', 'Rice Hispa', 'Sheath Blight'],
    'D2': ['Bacterial Blight', 'Blast', 'Brown Spot', 'Tungro'],
}
ALIAS_EXTRA = {
    'D1': {
        'healthy': 'Healthy Rice Leaf', 'healthyrice': 'Healthy Rice Leaf',
        'narrowbrownspot': 'Narrow Brown Leaf Spot',
        'narrowbrownleafspot': 'Narrow Brown Leaf Spot',
        'bacterialblight': 'Bacterial Leaf Blight',
        'riceblast': 'Leaf Blast', 'blast': 'Leaf Blast',
        'hispa': 'Rice Hispa', 'sheathblight': 'Sheath Blight',
    },
    'D2': {
        'bacterialleafblight': 'Bacterial Blight',
        'bacterialblight': 'Bacterial Blight',
        'leafblast': 'Blast', 'riceblast': 'Blast',
        'brownspot': 'Brown Spot',
        'ricetungro': 'Tungro',
    },
}
ALIAS = {}
for dataset, names in CLASS_NAMES.items():
    d = {normalize_label(n): n for n in names}
    d.update(ALIAS_EXTRA[dataset]); ALIAS[dataset] = d


def sha_of_file(path: Path) -> str:
    dig = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            dig.update(chunk)
    return dig.hexdigest()


def dhash_64(img: Image.Image) -> int:
    gray = img.convert('L').resize((9, 8), Image.Resampling.LANCZOS)
    p = list(gray.getdata()); out = 0
    for y in range(8):
        for x in range(8):
            out = (out << 1) | int(p[y*9+x] > p[y*9+x+1])
    return out


def scan(root: Path, dataset: str):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f'{dataset}: folder not found: {root}')
    entries = sorted([p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not entries:
        raise ValueError(f'{dataset}: no image files found under {root}')
    records, errors, skipped_augmented = [], [], 0
    for path in entries:
        rel = path.relative_to(root)
        path_parts = list(rel.parent.parts)
        all_parts = [root.name] + path_parts
        if dataset == 'D1':
            # Mendeley v2 contains an Augmented subtree, which MUST be excluded.
            if any(normalize_label(x).startswith('augment') for x in all_parts):
                skipped_augmented += 1
                continue
            if not any(normalize_label(x) in ("original", "originalimages") for x in all_parts):
                errors.append({'path': str(rel), 'problem': 'D1 file not demonstrably inside Original directory'})
                continue
        labels = {ALIAS[dataset][normalize_label(x)] for x in path_parts if normalize_label(x) in ALIAS[dataset]}
        if len(labels) != 1:
            errors.append({'path': str(rel), 'problem': f'expected one recognizable class folder, got {sorted(labels)}'})
            continue
        label = next(iter(labels))
        try:
            with Image.open(path) as raw:
                raw.verify()
            with Image.open(path) as raw:
                im = ImageOps.exif_transpose(raw).convert('RGB')
                width, height = im.size
                h = hashlib.sha256()
                h.update(f'{width},{height}:'.encode()); h.update(im.tobytes())
                pixel_hash = h.hexdigest()
                perceptual = dhash_64(im)
            file_hash = sha_of_file(path)
        except (OSError, ValueError, UnidentifiedImageError) as e:
            errors.append({'path': str(rel), 'problem': f'decode error: {type(e).__name__}: {str(e)[:150]}'})
            continue
        rel_s = rel.as_posix()
        rid = hashlib.sha256(f'{dataset}|{rel_s}'.encode('utf-8')).hexdigest()[:20]
        records.append({
            'id': rid, 'dataset': dataset, 'rel_path': rel_s, 'class_name': label,
            'width': width, 'height': height, 'sha256_file': file_hash,
            'sha256_decoded_rgb': pixel_hash, 'dhash64': f'{perceptual:016x}',
        })
    return records, errors, skipped_augmented


def write_csv(path: Path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


class UnionFind:
    def __init__(self, ids):
        self.parent = {rid: rid for rid in ids}
    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a,b)] = min(a,b)


class BKTree:
    """Exact Hamming-distance retrieval without an all-pairs O(N^2) pass."""
    def __init__(self):
        self.root = None
    def add(self, value, idx):
        node = [value, idx, {}]
        if self.root is None:
            self.root = node
            return
        cur = self.root
        while True:
            d = (value ^ cur[0]).bit_count()
            if d not in cur[2]:
                cur[2][d] = node; return
            cur = cur[2][d]
    def query(self, value, radius):
        if self.root is None: return []
        found=[]; stack=[self.root]
        while stack:
            node=stack.pop(); d=(value^node[0]).bit_count()
            if d<=radius: found.append((node[1],d))
            for edge,nxt in node[2].items():
                if d-radius <= edge <= d+radius: stack.append(nxt)
        return found


def pairs_and_components(records, near_radius: int):
    uf = UnionFind([r['id'] for r in records])
    seen_exact={}; exact=[]
    for r in records:
        # Image-byte differences from metadata/compression are handled by decoded RGB hashes.
        key=r['sha256_decoded_rgb']
        if key in seen_exact:
            prev=seen_exact[key]
            exact.append({'id_a':prev['id'],'id_b':r['id'],'dataset_a':prev['dataset'],
                          'dataset_b':r['dataset'],'class_a':prev['class_name'],
                          'class_b':r['class_name'],'path_a':prev['rel_path'],
                          'path_b':r['rel_path'],'reason':'same decoded RGB pixels'})
            uf.union(prev['id'],r['id'])
        else: seen_exact[key]=r
    tree=BKTree(); near=[]
    for idx,r in enumerate(records):
        v=int(r['dhash64'],16)
        for j,dist in tree.query(v,near_radius):
            other=records[j]
            if r['sha256_decoded_rgb']==other['sha256_decoded_rgb']:
                continue
            a,b=sorted((r['id'],other['id']))
            pair_id=hashlib.sha256(f'{a}|{b}'.encode()).hexdigest()[:20]
            near.append({'pair_id':pair_id,'id_a':other['id'],'id_b':r['id'],
                         'dataset_a':other['dataset'],'dataset_b':r['dataset'],
                         'class_a':other['class_name'],'class_b':r['class_name'],
                         'path_a':other['rel_path'],'path_b':r['rel_path'],
                         'hamming_distance':dist,'decision':''})
        tree.add(v,idx)
    near.sort(key=lambda r:(r['hamming_distance'],r['dataset_a'],r['path_a'],r['path_b']))
    return uf, exact, near


def apply_grouping_decisions(exact_uf, near, review_file):
    """Preserve exact-duplicate groups; GROUP uncertain near pairs for leakage control.

    Do NOT deduplicate on dHash. Cross-class candidates need explicit review:
    keep_grouped = related scene without assuming identical label/image;
    distinct = visually distinct images. Same-class dHash candidates are grouped
    conservatively without being removed. Assessed risk remains documented.
    """
    leakage_uf = UnionFind(exact_uf.parent)
    for rid in exact_uf.parent:
        leakage_uf.union(rid, exact_uf.find(rid))
    reviews = {}
    blockers = []
    if review_file:
        with Path(review_file).open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not {'pair_id', 'decision'} <= set(reader.fieldnames or []):
                raise ValueError('Cross-class review CSV requires pair_id,decision')
            for r in reader:
                pid, dec = r['pair_id'].strip(), r['decision'].strip().lower()
                if pid in reviews:
                    raise ValueError(f'Duplicate review pair_id: {pid}')
                if dec not in {'keep_grouped', 'distinct'}:
                    raise ValueError(f'Unsupported review decision {dec!r} for {pid}')
                reviews[pid] = dec
    seen_review_ids = set()
    policy_counts = Counter()
    for pair in near:
        ds_a, ds_b = pair['dataset_a'], pair['dataset_b']
        different_classes = pair['class_a'] != pair['class_b']
        pid = pair['pair_id']
        decision = reviews.get(pid, '')
        if decision:
            seen_review_ids.add(pid)
        if ds_a != ds_b:
            # Do not assert dataset independence or mix different dataset partitions.
            blockers.append(f'Potential cross-dataset source overlap: {pid}; assess independently')
            pair['decision'] = 'cross_dataset_blocked'
            continue
        if different_classes:
            if decision == 'distinct':
                pair['decision'] = 'reviewed_distinct'
                policy_counts['reviewed_distinct_cross_class'] += 1
            elif decision == 'keep_grouped':
                leakage_uf.union(pair['id_a'], pair['id_b'])
                pair['decision'] = 'reviewed_keep_grouped'
                policy_counts['reviewed_grouped_cross_class'] += 1
            else:
                pair['decision'] = 'unreviewed_cross_class'
                blockers.append(f'Unreviewed cross-class near pair {pid} ({pair["class_a"]} vs {pair["class_b"]})')
        else:
            # Conservative train/val/test co-location; NO removal and NO relabeling.
            leakage_uf.union(pair['id_a'], pair['id_b'])
            pair['decision'] = 'conservative_same_class_group'
            policy_counts['conservative_same_class_pairs'] += 1
            if decision:
                blockers.append(f'Review decision for same-class pair {pid} is not supported by this policy')
    stale = sorted(set(reviews) - seen_review_ids)
    # Excluded/reprocessed images can eliminate former candidate pairs. Record
    # without approving unseen labels; stale rows are NOT used for decisions.
    return leakage_uf, blockers, dict(policy_counts), stale


def assign_group_split(records, leakage_uf, seed):
    """Deterministic grouped stratification, targets 70/15/15 per class.

    Multi-class groups are allowed ONLY for grouping already reviewed different-label
    photographs; grouping never merges their class labels or drops samples.
    """
    groups = defaultdict(list)
    for r in records:
        groups[leakage_uf.find(r['id'])].append(r)
    class_totals = Counter(r['class_name'] for r in records)
    if any(n < 3 for n in class_totals.values()):
        raise ValueError('Class with fewer than 3 distinct images')
    targets = {name: {'train': n * .70, 'val': n * .15, 'test': n * .15}
               for name,n in class_totals.items()}
    split_names = ('train','val','test')
    alloc = {sp: Counter() for sp in split_names}
    rng = random.Random(seed)
    sorted_groups = sorted(groups.items(), key=lambda kv: kv[0])
    rng.shuffle(sorted_groups)
    sorted_groups.sort(key=lambda kv: -len(kv[1]))
    assignments = {}
    for gid, group in sorted_groups:
        gc = Counter(x['class_name'] for x in group)
        def score(sp):
            # Prefer split with largest remaining weighted per-class deficit.
            deficits = sum((targets[c][sp] - alloc[sp][c]) * k / class_totals[c]
                           for c,k in gc.items())
            relative_load = sum(alloc[sp][c] / max(1.0, targets[c][sp])
                                for c in gc)
            return (deficits, -relative_load, -sum(alloc[sp].values()))
        chosen = max(split_names, key=score)
        assignments[gid] = chosen
        alloc[chosen].update(gc)
    assigned = [{**r, 'split': assignments[leakage_uf.find(r['id'])],
                 'leakage_group_id': leakage_uf.find(r['id'])}
                for r in records]
    for name in class_totals:
        for sp in split_names:
            if not any(r['class_name'] == name and r['split'] == sp for r in assigned):
                raise ValueError(f'Grouped split lacks {name} in {sp}')
    assert len({r['id'] for r in assigned}) == len(assigned)
    for group in groups.values():
        assert len({assignments[leakage_uf.find(r['id'])] for r in group}) == 1
    return sorted(assigned, key=lambda r:(r['split'], r['class_name'], r['rel_path']))


def run(args):
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    roots={'D1':Path(args.d1_root),'D2':Path(args.d2_root)}
    expected={'D1':args.expected_d1,'D2':args.expected_d2}
    all_rows=[]; errors=[]; ignored={}
    for ds in ('D1','D2'):
        rows,errs,skipped=scan(roots[ds],ds)
        all_rows.extend(rows)
        ignored[ds]=skipped
        errors.extend([{'dataset':ds,**e} for e in errs])
    inv_cols=['id','dataset','rel_path','class_name','width','height','sha256_file','sha256_decoded_rgb','dhash64']
    write_csv(out/'all_image_inventory.csv',all_rows,inv_cols)
    write_csv(out/'scan_errors.csv',errors,['dataset','path','problem'])

    exclusions=[]
    if args.exclude_csv:
        with Path(args.exclude_csv).open(newline='',encoding='utf-8') as f:
            exclusions=list(csv.DictReader(f))
        for x in exclusions:
            if not x.get('reason','').strip():
                raise ValueError('Every exclusion needs a documented reason')
        excluded={(r['dataset'],r['rel_path']) for r in exclusions}
        found={(r['dataset'],r['rel_path']) for r in all_rows}
        unknown=excluded-found
        if unknown: raise ValueError(f'exclusion file references unknown images: {sorted(unknown)[:10]}')
    else: excluded=set()
    rows=[r for r in all_rows if (r['dataset'],r['rel_path']) not in excluded]
    exact_uf, exact, near = pairs_and_components(rows, args.near_radius)
    write_csv(out/'exact_duplicate_pairs.csv',exact,
              ['id_a','id_b','dataset_a','dataset_b','class_a','class_b','path_a','path_b','reason'])
    leakage_uf, blockers, grouping_counts, stale_reviews = apply_grouping_decisions(
        exact_uf, near, args.review_file)
    near_cols=['pair_id','id_a','id_b','dataset_a','dataset_b','class_a','class_b',
               'path_a','path_b','hamming_distance','decision']
    write_csv(out/'near_duplicate_review.csv',near,near_cols)
    if args.review_file: shutil.copyfile(args.review_file,out/'submitted_review_decisions.csv')
    if args.exclude_csv: shutil.copyfile(args.exclude_csv,out/'submitted_exclusions.csv')

    counts={ds:sum(1 for r in all_rows if r['dataset']==ds) for ds in ('D1','D2')}
    class_counts={ds:dict(sorted(Counter(r['class_name'] for r in all_rows if r['dataset']==ds).items())) for ds in ('D1','D2')}
    for ds in ('D1','D2'):
        if counts[ds]!=expected[ds]: blockers.append(f'{ds}: found {counts[ds]} readable images; expected {expected[ds]} before exclusions')
        actual=set(class_counts[ds]); wanted=set(CLASS_NAMES[ds])
        if actual!=wanted: blockers.append(f'{ds}: class set mismatch; missing {sorted(wanted-actual)} / extras {sorted(actual-wanted)}')
    if errors: blockers.append(f'{len(errors)} images corrupt/unknown class/not under Original (see scan_errors.csv)')

    # Only pixel-identical images may be automatically deduplicated.
    # A near-duplicate candidate is never deleted solely due to dHash.
    exact_groups = defaultdict(list)
    for r in rows:
        exact_groups[exact_uf.find(r['id'])].append(r)
    included, deduped = [], []
    for group in exact_groups.values():
        group = sorted(group, key=lambda r: (r['dataset'],r['rel_path']))
        classes = {r['class_name'] for r in group}
        datasets = {r['dataset'] for r in group}
        if len(classes) > 1:
            blockers.append('Pixel-identical image with conflicting class labels: '+str(
                [(r['dataset'],r['rel_path'],r['class_name']) for r in group][:8]))
        if len(datasets) > 1:
            blockers.append('Pixel-identical image overlaps D1/D2: '+str(
                [(r['dataset'],r['rel_path']) for r in group][:8]))
        keep = group[0]
        included.append(keep)
        for r in group[1:]:
            deduped.append({'dataset':r['dataset'],'rel_path':r['rel_path'],
                            'kept_dataset':keep['dataset'],'kept_rel_path':keep['rel_path'],
                            'reason':'pixel-identical decoded RGB image; same-label exact copy'})
    write_csv(out/'deduplicated_exclusions.csv',deduped,
              ['dataset','rel_path','kept_dataset','kept_rel_path','reason'])
    # Post-exclusion review: the two explicitly excluded D1 conflicting records
    # cannot enter any split; stale old pair decisions are informational only.
    output_group_rows = []
    for r in included:
        output_group_rows.append({'dataset':r['dataset'],'rel_path':r['rel_path'],
                                  'class_name':r['class_name'],
                                  'group_id':leakage_uf.find(r['id'])})
    write_csv(out/'leakage_groups.csv',output_group_rows,
              ['dataset','rel_path','class_name','group_id'])

    if not blockers:
        try:
            splits={}
            for ds in ('D1','D2'):
                subset=[r for r in included if r['dataset']==ds]
                rows_split=assign_group_split(subset,leakage_uf,args.seed)
                class_id={name:i for i,name in enumerate(CLASS_NAMES[ds])}
                manifest=[]
                for r in rows_split:
                    manifest.append({'dataset':ds,'rel_path':r['rel_path'],
                                     'class_name':r['class_name'],'class_id':class_id[r['class_name']],
                                     'split':r['split'],'source_image_sha256':r['sha256_decoded_rgb'],
                                     'file_sha256':r['sha256_file'],
                                     'leakage_group_id':r['leakage_group_id']})
                assert len({r['rel_path'] for r in manifest})==len(manifest)
                by_split=defaultdict(set)
                for r in manifest: by_split[r['split']].add(r['source_image_sha256'])
                assert not (by_split['train']&by_split['val'] or by_split['train']&by_split['test'] or by_split['val']&by_split['test'])
                assert all(any(r['class_name']==n and r['split']==split for r in manifest) for n in CLASS_NAMES[ds] for split in ('train','val','test'))
                output_manifest=out/f'{ds}_split_manifest.csv'
                write_csv(output_manifest,manifest,['dataset','rel_path','class_name','class_id','split','source_image_sha256','file_sha256','leakage_group_id'])
                splits[ds]={'count':len(manifest),'by_split':dict(Counter(r['split'] for r in manifest)),
                            'by_class':dict(Counter(r['class_name'] for r in manifest)),
                            'sha256_manifest':sha_of_file(output_manifest),
                             'group_count':len({r['leakage_group_id'] for r in manifest}),
                            'class_id_map':class_id}
            (out/'class_maps.json').write_text(json.dumps({ds:{n:i for i,n in enumerate(CLASS_NAMES[ds])} for ds in CLASS_NAMES},indent=2))
        except (AssertionError,ValueError) as e:
            blockers.append(f'split failed: {e}')
            splits={}
    else: splits={}

    status={
        'status':'PASS_SPLITS_FROZEN' if not blockers else 'BLOCKED_NO_SPLITS',
        'source_urls':{'D1':'https://data.mendeley.com/datasets/hx6f852hw4/2',
                       'D2_original':'https://data.mendeley.com/datasets/fwcj7stb8r/2',
                       'D2_distributor':'original Mendeley archive in current experiment'},
        'actual_roots':{k:str(v.resolve()) for k,v in roots.items()},
        'reported_source_counts':{'D1':1701,'D2':5932},
        'expected_checked_counts':expected,
        'observed_image_counts':counts,
        'observed_class_counts':class_counts,
        'd1_skipped_augmented_files':ignored['D1'],
        'scan_errors':len(errors),'exact_duplicate_pairs':len(exact),
        'near_duplicate_pairs':len(near),
        'documented_manual_exclusions':len(exclusions),
        'auto_deduplicated_images':len(deduped),
        'grouping_policy':'Same-class near pairs conservatively co-located; cross-class pairs require explicit review; only exact pixel copies removed',
        'grouping_counts':grouping_counts,'stale_review_pairs_after_exclusions':stale_reviews,
        'seed':args.seed,'near_hamming_radius':args.near_radius,
        'review_csv_path':str(args.review_file) if args.review_file else None,
        'blockers':blockers,
        'splits':splits,
        'warning':'Conservative dHash groups are not confirmed same-photo identities. No independent leaf/session IDs. This gate is not proof of dataset independence or of no leakage.',
    }
    (out/'gate_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2)[:20000])
    return 0 if not blockers else 2


def main():
    p=argparse.ArgumentParser(description='Assignment 02 source data gate. No training.')
    p.add_argument('--d1-root',required=True,help='D1 extracted folder containing Original/ (Augmented excluded)')
    p.add_argument('--d2-root',required=True,help='D2 class-organized original/AgML images folder')
    p.add_argument('--output',required=True,help='Write reports and manifests here')
    p.add_argument('--expected-d1',type=int,default=1701)
    p.add_argument('--expected-d2',type=int,default=5932)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--near-radius',type=int,default=3,help='dHash Hamming radius for conservative grouping; cross-class candidates require review')
    p.add_argument('--review-file',default='',help='Cross-class candidate decisions CSV: pair_id,decision (keep_grouped or distinct)')
    p.add_argument('--exclude-csv',default='',help='Optional csv with dataset,rel_path,reason')
    a=p.parse_args()
    try: code=run(a)
    except Exception as e:
        print(f'FATAL DATASET AUDIT ERROR: {type(e).__name__}: {e}',file=sys.stderr)
        return 3
    return code

if __name__=='__main__':
    sys.exit(main())
