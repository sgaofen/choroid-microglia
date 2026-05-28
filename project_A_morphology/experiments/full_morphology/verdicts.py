def verdict(wt, h1, h3):
    g1, g3 = h1-wt, h3-wt
    # same side?
    if (g1>0) == (g3>0) and g1!=0 and g3!=0:
        gap = abs((g1+g3)/2)
        spread = abs(h1-h3)
        return "CLEAN" if spread < gap else "same-side"
    return "disagree"

metrics = {
 "% 3-way junctions":      (67.41, 68.92, 72.22),
 "% 4-way junctions":      (31.57, 29.32, 26.72),
 "% 5+-way junctions":     (1.024, 1.754, 1.058),
 "% high-order (>=4-way)": (32.59, 31.08, 27.78),
 "mean junction order":    (3.336, 3.328, 3.288),
 "junctions per mm2":      (1366.5, 927.8, 879.0),
 "junctions per 100um skel":(1.775, 1.537, 1.473),
 "endpoint:junction ratio":(12.33, 16.46, 16.70),
}
for n,(w,a,b) in metrics.items():
    mh=(a+b)/2
    pct=100*(mh-w)/w
    print(f"{n:28s} WT={w:8.3f} H1={a:8.3f} H3={b:8.3f} meanHET={mh:8.3f} pct={pct:7.2f}% -> {verdict(w,a,b)}")
