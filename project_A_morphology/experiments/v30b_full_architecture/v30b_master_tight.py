"""Tighter crop version — ~500x500 px area, ~25 cells per panel."""
import sys
from pathlib import Path

V30B = Path('/Users/stephenyu/Documents/choroid-microglia/project_A_morphology'
            '/experiments/v30b_full_architecture')
sys.path.insert(0, str(V30B))
from v30b_master import make_master


if __name__ == '__main__':
    # half=250 => 500x500 px crop, ~20-30 cells per panel
    # figsize stays at module default (36x13); we override by tweaking after
    import v30b_master as vm
    import matplotlib.pyplot as plt

    targets = [
        ('F_WT_2',  1100, 1100, 250, 'tight'),
        ('F_HET_1', 2400, 1900, 250, 'tight'),
        ('F_HET_3', 1500, 1500, 250, 'tight'),
        ('F_HET_3',  874, 2390, 250, 'vessel'),
    ]
    for (s, y, x, half, tag) in targets:
        # call with custom half, then resize figure dims to taste:
        # we monkey-patch plt.subplots to use a smaller figsize this run
        orig_subplots = plt.subplots
        def smaller_subplots(*a, **kw):
            kw['figsize'] = (24, 9)
            return orig_subplots(*a, **kw)
        plt.subplots = smaller_subplots
        try:
            make_master(s, y, x, half=half, tag=tag)
        finally:
            plt.subplots = orig_subplots
