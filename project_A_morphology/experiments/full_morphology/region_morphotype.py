"""
Region (tile) morphotype analysis — CORRECTED per GPT-Pro review (2026-05-27).

Fixes vs the first version:
  P1  branch/endpoint counts come from ONE global topology pass
      (clean_topology.merged_branches with exit>=3 + endpoints), assigned to
      tiles -> tile counts now sum to the whole-image counts; tile borders no
      longer split junction clusters or use the loose dilation count.
  P2  the canonical morphotype clustering uses SHAPE/topology features only
      (no coverage, no abundance) -> morphotype = morphology, not signal amount.
      A coverage-inclusive clustering is rerun ONLY as a labelled sensitivity.
  P3  densities reported per TISSUE-TILE area AND per FOREGROUND area AND per
      100 um skeleton -> separates "how much process" (abundance) from "how the
      process is shaped" (topology/fragmentation).
  P6  a pre-registered fragmentation score per tile, summarized per image.

Unit = 200px (~41um) tile with foreground >= 3% (an Iba1-positive tile, NOT a
true tissue-mask fraction — noted as a limitation).
"""
import sys, json, csv
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl
from pipeline import PIXEL_UM, STEMS, COND

OUT = pl.ROOT / 'experiments/full_morphology/out_region'
TILE = 200
MIN_FG_FRAC = 0.03
K = 4
TILE_MM2 = TILE * TILE * PIXEL_UM ** 2 / 1e6
# canonical clustering: SHAPE/topology only (no coverage, no abundance)
SHAPE = ['branch_per_100um', 'endpoint_per_100um', 'endpoint_branch_ratio',
         'mean_thickness_um', 'mean_seg_len_um']


def grid_counts(coords_y, coords_x, ntY, ntX, weights=None):
    g = np.zeros((ntY, ntX))
    iy = (np.asarray(coords_y) // TILE).astype(int)
    ix = (np.asarray(coords_x) // TILE).astype(int)
    ok = (iy >= 0) & (iy < ntY) & (ix >= 0) & (ix < ntX)
    np.add.at(g, (iy[ok], ix[ok]), 1.0 if weights is None else np.asarray(weights)[ok])
    return g


def tile_rows(stem):
    norm, binary, dist, skel = pl.prep(stem)
    J, E = pl.global_topology(skel)
    midy, midx, seglen = pl.segments_df(skel)
    H, W = skel.shape
    ntY, ntX = (H + TILE - 1) // TILE, (W + TILE - 1) // TILE
    # global topology -> per-tile grids (counts sum to whole-image totals)
    br_g = grid_counts(J[:, 0], J[:, 1], ntY, ntX) if len(J) else np.zeros((ntY, ntX))
    ep_g = grid_counts(E[:, 0], E[:, 1], ntY, ntX) if len(E) else np.zeros((ntY, ntX))
    sc_g = grid_counts(midy, midx, ntY, ntX)
    sl_g = grid_counts(midy, midx, ntY, ntX, weights=seglen)
    ys, xs = np.where(skel)
    skpx_g = grid_counts(ys, xs, ntY, ntX)

    rows = []
    for ty in range(ntY):
        for tx in range(ntX):
            sl = (slice(ty*TILE, ty*TILE+TILE), slice(tx*TILE, tx*TILE+TILE))
            tb = binary[sl]; fg_px = int(tb.sum())
            if fg_px / tb.size < MIN_FG_FRAC:
                continue
            tsk = skel[sl]
            skel_um = skpx_g[ty, tx] * PIXEL_UM
            if skel_um < 1:
                continue
            fg_mm2 = fg_px * PIXEL_UM ** 2 / 1e6
            nbr, nep = br_g[ty, tx], ep_g[ty, tx]
            thick = float(dist[sl][tsk].mean()) * 2 * PIXEL_UM if tsk.any() else 0.0
            rows.append(dict(
                image=stem, condition=COND[stem], ty=ty*TILE, tx=tx*TILE,
                fg_fraction=round(fg_px / tb.size, 4),
                # abundance (P3): per tissue-tile area AND per foreground area
                skel_per_tile_mm2=round(skel_um / TILE_MM2, 1),
                skel_per_fg_mm2=round(skel_um / fg_mm2, 1),
                # topology, denominator-robust (per 100um skeleton)
                branch_per_100um=round(100 * nbr / skel_um, 3),
                endpoint_per_100um=round(100 * nep / skel_um, 3),
                endpoint_branch_ratio=round(nep / max(nbr, 1), 3),
                mean_thickness_um=round(thick, 3),
                mean_seg_len_um=round(sl_g[ty, tx] / sc_g[ty, tx], 3) if sc_g[ty, tx] else 0.0,
                n_branch=int(nbr), n_endpoint=int(nep),
            ))
    return rows, (J, E, skel.shape)


def verdict(vals):
    """vals: dict stem->value (any number of WT/HET images)."""
    return pl.group_verdict(vals)[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    geo = {}
    print('extracting tiles (global topology assigned to tiles)...')
    for s in STEMS:
        rows, g = tile_rows(s)
        all_rows += rows
        geo[s] = g
        print(f'  {s}: {len(rows)} tiles')

    # ---- P1 validation: tile branch/endpoint counts now sum to whole-image ----
    print('\n=== P1 check: sum(tile counts) vs global topology ===')
    for s in STEMS:
        J, E, _ = geo[s]
        sb = sum(r['n_branch'] for r in all_rows if r['image'] == s)
        se = sum(r['n_endpoint'] for r in all_rows if r['image'] == s)
        print(f'  {s}: tiles branch={sb} global junctions={len(J)} | tiles ep={se} global ep={len(E)}'
              f'  (tiles drop <3% fg, so <= global)')

    # ---- fragmentation score (P6): z(ep/br)+z(ep/100um)-z(branch/100um)-z(seglen) ----
    def z(key):
        v = np.array([r[key] for r in all_rows], float)
        return (v - v.mean()) / (v.std() + 1e-9)
    frag = (z('endpoint_branch_ratio') + z('endpoint_per_100um')
            - z('branch_per_100um') - z('mean_seg_len_um'))
    for r, f in zip(all_rows, frag):
        r['frag_score'] = round(float(f), 3)
    frag_hi = np.percentile(frag, 75)

    # ---- canonical clustering: SHAPE only (P2) ----
    X = np.array([[r[k] for k in SHAPE] for r in all_rows], float)
    lab = KMeans(K, random_state=0, n_init=10).fit_predict(StandardScaler().fit_transform(X))
    rami = np.array([r['branch_per_100um'] for r in all_rows])
    order = np.argsort([-rami[lab == k].mean() for k in range(K)])   # C0 most ramified -> last de-ramified
    remap = {o: i for i, o in enumerate(order)}
    lab = np.array([remap[l] for l in lab])
    for r, l in zip(all_rows, lab):
        r['cluster'] = int(l)
    deram = K - 1  # last = least ramified = de-ramified

    print('\n=== canonical morphotype profiles (SHAPE-only clustering; C0 ramified -> C%d de-ramified) ===' % deram)
    for k in range(K):
        m = lab == k
        f = lambda c: np.mean([all_rows[i][c] for i in np.where(m)[0]])
        print(f'  C{k} (n={m.sum():>3}): branch/100um={f("branch_per_100um"):.2f} '
              f'ep/br={f("endpoint_branch_ratio"):.2f} thick={f("mean_thickness_um"):.2f} '
              f'seglen={f("mean_seg_len_um"):.2f} frag={f("frag_score"):+.2f}')

    # ---- per-image summary + replicate consistency ----
    print('\n=== per-image summary (corrected) ===')
    summ = {}
    cols = ['skel_per_tile_mm2', 'skel_per_fg_mm2', 'branch_per_100um', 'endpoint_per_100um',
            'endpoint_branch_ratio', 'mean_thickness_um', 'mean_seg_len_um', 'frag_score']
    for s in STEMS:
        r = [x for x in all_rows if x['image'] == s]
        idx = [i for i, x in enumerate(all_rows) if x['image'] == s]
        summ[s] = dict(condition=COND[s], n_tiles=len(r))
        for c in cols:
            summ[s][c] = round(float(np.mean([x[c] for x in r])), 3)
        summ[s]['deram_pct'] = round(100 * float(np.mean([lab[i] == deram for i in idx])), 1)
        summ[s]['frag_hotspot_pct'] = round(100 * float(np.mean([all_rows[i]['frag_score'] > frag_hi for i in idx])), 1)

    print(f'{"metric":>22}{("  ".join(STEMS)):>34}{"verdict":>16}')
    for m in cols + ['deram_pct', 'frag_hotspot_pct']:
        vals = {s: summ[s][m] for s in STEMS}
        cells = '  '.join(f'{vals[s]:.4g}' for s in STEMS)
        print(f'{m:>22}{cells:>34}{verdict(vals):>16}')

    # ---- coverage-inclusive sensitivity clustering (P2: labelled separately) ----
    Xc = np.array([[r[k] for k in SHAPE] + [r['fg_fraction'], r['skel_per_tile_mm2']]
                   for r in all_rows], float)
    labc = KMeans(K, random_state=0, n_init=10).fit_predict(StandardScaler().fit_transform(Xc))
    order_c = np.argsort([-rami[labc == k].mean() for k in range(K)])
    remap_c = {o: i for i, o in enumerate(order_c)}
    labc = np.array([remap_c[l] for l in labc])
    sens = {}
    for s in STEMS:
        idx = [i for i, x in enumerate(all_rows) if x['image'] == s]
        sens[s] = round(100 * float(np.mean([labc[i] == K-1 for i in idx])), 1)
    print('\n=== sensitivity: coverage-INCLUSIVE clustering, de-ramified %% (should still be HET>WT) ===')
    print('   ' + '  '.join(f'{s}={sens[s]}' for s in STEMS) + f'   {verdict(sens)}')

    # ---- save ----
    with open(OUT / 'region_features.csv', 'w', newline='') as f:
        cc = ['image', 'condition', 'ty', 'tx', 'fg_fraction', 'skel_per_tile_mm2',
              'skel_per_fg_mm2', 'branch_per_100um', 'endpoint_per_100um',
              'endpoint_branch_ratio', 'mean_thickness_um', 'mean_seg_len_um',
              'n_branch', 'n_endpoint', 'frag_score', 'cluster']
        w = csv.DictWriter(f, fieldnames=cc); w.writeheader()
        for r in all_rows:
            w.writerow({c: r[c] for c in cc})
    (OUT / 'region_morphotype.json').write_text(json.dumps(dict(
        tile_um=TILE * PIXEL_UM, k=K, canonical_features=SHAPE, deram_cluster=deram,
        note='canonical clustering is SHAPE-only (no coverage/abundance); '
             'coverage-inclusive is sensitivity only',
        summary=summ, coverage_inclusive_deram_pct=sens), indent=1))

    # ---- maps (canonical) ----
    cmap = ListedColormap(['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c'][:K])
    for s in STEMS:
        _, _, shape = geo[s]
        ntX = (shape[1] + TILE - 1) // TILE
        grid = np.full(((shape[0]+TILE-1)//TILE, ntX), np.nan)
        for i, r in enumerate(all_rows):
            if r['image'] == s:
                grid[r['ty']//TILE, r['tx']//TILE] = lab[i]
        plt.figure(figsize=(6, 6)); plt.imshow(grid, cmap=cmap, vmin=-0.5, vmax=K-0.5)
        cb = plt.colorbar(ticks=range(K)); cb.set_label('morphotype (C0 ramified -> C%d de-ramified)' % (K-1))
        plt.title(f'{s} ({COND[s]}) morphotype map (SHAPE-only)'); plt.axis('off')
        plt.savefig(OUT / f'{s}_morphotype_map.png', dpi=110, bbox_inches='tight'); plt.close()

    # ---- composition bar + fragmentation distribution ----
    plt.figure(figsize=(7, 5))
    x = np.arange(K); width = 0.35
    wtp = [100*np.mean([lab[i]==k for i,r in enumerate(all_rows) if r['condition']=='WT']) for k in range(K)]
    hetp = [100*np.mean([lab[i]==k for i,r in enumerate(all_rows) if r['condition']=='HET']) for k in range(K)]
    plt.bar(x-width/2, wtp, width, label='WT', color='C0')
    plt.bar(x+width/2, hetp, width, label='HET', color='C3')
    plt.xticks(x, [f'C{k}' for k in range(K)]); plt.ylabel('% of regions')
    plt.title('morphotype composition WT vs HET (SHAPE-only)'); plt.legend()
    plt.savefig(OUT / 'composition_bar.png', dpi=120, bbox_inches='tight'); plt.close()

    plt.figure(figsize=(8, 5))
    for s in STEMS:
        v = [r['frag_score'] for r in all_rows if r['image'] == s]
        c = 'C0' if COND[s] == 'WT' else 'C3'
        plt.hist(v, bins=np.linspace(frag.min(), frag.max(), 30), density=True,
                 histtype='step', lw=2, label=f'{s}({COND[s]})', color=c)
    plt.axvline(frag_hi, ls='--', c='k', lw=1, label='hotspot thr (pooled p75)')
    plt.xlabel('fragmentation score'); plt.ylabel('density'); plt.legend()
    plt.title('per-region fragmentation score, WT vs HET')
    plt.savefig(OUT / 'fragmentation_distribution.png', dpi=120, bbox_inches='tight'); plt.close()

    print('\nsaved to', OUT)


if __name__ == '__main__':
    main()
