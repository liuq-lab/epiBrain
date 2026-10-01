#!/usr/bin/env python3
"""
epiBrainLLM-UKB: build AD/dementia case-control labels from the UKB EHR phecode
binary, for the clinical arm.

CRITICAL — TOKEN OFFSET: the EHR bin's `token` column is offset from the mapping
CSV `index` by `bin_token_offset = -1` (confirmed: bin_token == mapping_index - 1,
verified 10/10 on known fields incl. Female/Male/dementia phecodes). So to select a
phecode whose mapping index is X, we match bin token (X - 1).

EHR bin format (float32, 6 cols): eid, days_since_birth, token, source, year, lab.

Relevant phecodes (mapping index -> phecode -> name -> n):
  425 290.11 Alzheimer's disease (4483)
  424 290.1  Dementias (7779)            [actually 290.1 parent of dementias]
  426 290.12 Dementia w/ cerebral degenerations (551)
  427 290.16 Vascular dementia (2255)
  437 292.2  Mild cognitive impairment
  438 292.3  Memory loss

We emit several case definitions as separate columns so downstream can pick:
  ad_strict   = 290.11 only                     (the paper's "Alzheimer's")
  ad_dementia = 290.1 + 290.11 + 290.12         (degenerative dementias, no vascular)
  ad_alldem   = + 290.16 vascular               (all 290.x dementias)
  ad_mci      = ad_alldem + 292.2 MCI
Plus first-onset day/year per definition's earliest event (for conversion analysis).

Usage:
  python 06_build_ad_labels.py \
    --ehr_bin /work/.../ukb_real.bin \
    --mapping /work/.../phecode_token_category_ukb_v5.csv \
    --out phenotypes/ad_labels.tsv [--offset -1]
"""
import argparse
import numpy as np
import pandas as pd


# case definitions by MAPPING INDEX (we convert to bin tokens via the offset)
DEFS = {
    'ad_strict':   [425],                 # 290.11 Alzheimer's only
    'ad_dementia': [424, 425, 426],       # 290.1 + 290.11 + 290.12 (degenerative)
    'ad_alldem':   [424, 425, 426, 427],  # + 290.16 vascular
    'ad_mci':      [424, 425, 426, 427, 436],  # + 292.2 MCI (mapping index 436)
}


def main(args):
    m = pd.read_csv(args.mapping)
    # sanity: print the mapping names for the indices we use
    used = sorted({i for v in DEFS.values() for i in v})
    print("Phecode definitions (mapping index -> name), bin token = index + offset(%d):" % args.offset)
    for i in used:
        r = m[m['index'] == i]
        nm = r['name'].iloc[0] if len(r) else '?'
        print(f"  idx {i} (bin tok {i + args.offset}): {nm}")

    a = np.fromfile(args.ehr_bin, dtype=np.float32).reshape(-1, 6)
    print(f"loaded EHR bin: {a.shape[0]/1e6:.1f}M rows")
    df = pd.DataFrame({
        'eid': a[:, 0].astype(np.int64),
        'days': a[:, 1].astype(np.int64),
        'token': a[:, 2].astype(np.int64),
        'year': a[:, 4].astype(np.int64),
    })

    all_eids = df['eid'].drop_duplicates()
    last_year = df.groupby('eid')['year'].max().rename('last_record_year')
    out = last_year.reset_index()

    for name, idxs in DEFS.items():
        bin_tokens = [i + args.offset for i in idxs]
        sub = df[df['token'].isin(bin_tokens)]
        case_eids = set(sub['eid'].unique())
        out[name] = out['eid'].isin(case_eids).astype(int)
        # earliest onset for this definition
        first = sub.sort_values('days').groupby('eid').first()[['days', 'year']]
        first = first.rename(columns={'days': f'{name}_first_days', 'year': f'{name}_first_year'})
        out = out.merge(first.reset_index(), on='eid', how='left')
        print(f"  {name}: {out[name].sum()} cases")

    out = out.sort_values('eid')
    out.to_csv(args.out, sep='\t', index=False)
    print(f"\nTotal eids: {len(out)} -> {args.out}")
    print("case columns:", [c for c in out.columns if c.startswith('ad_') and not c.endswith(('_days', '_year'))])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--ehr_bin', required=True)
    ap.add_argument('--mapping', required=True)
    ap.add_argument('--offset', type=int, default=-1, help='bin_token = mapping_index + offset (default -1)')
    ap.add_argument('--out', required=True)
    main(ap.parse_args())
