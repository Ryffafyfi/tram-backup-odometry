#!/usr/bin/env python3
"""Табличная модель тяги С УЧЁТОМ УКЛОНА: a = A(u, v) - g*sin(уклон).

Уклон в месте трамвая берётся из карты (z pathgraph и петель) по RTK-положению антенны master
(+9.873 м к base_link; уклон — по отрезку между тележками). Печатает остаточную ошибку
с учётом уклона и без и таблицу A(u, v) «на ровном пути».
"""
import glob
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from offline_eval import read_bag, sns  # noqa: E402
from tram_backup_odometry.geo import TransverseMercator  # noqa: E402
from tram_backup_odometry.track_map import TrackMap  # noqa: E402

G = 9.80665
V_BINS = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 30.0]
P = TransverseMercator()
M = HERE.parent / 'maps'
TM = TrackMap([str(M / f) for f in ('shchukinskaya_tallinskaya.json', 'tallinskaya_shchukinskaya.json',
                                    'terminal_loops.json')])


def slope_at(pid, s):
    """sin(уклона) по направлению пути: по хорде между тележками (s-7.55 .. s)."""
    p1 = TM.pose(pid, s)
    pid0, s0, _ = TM.advance(pid, s, -7.55)
    p0 = TM.pose(pid0, s0)
    d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    return (p1[2] - p0[2]) / max(d, 1.0)


def series(bag):
    msgs = read_bag(bag)
    f = np.array([(sns(m.header) * 1e-9, m.velocity / 3.6) for _, k, m in msgs if k == 'front'])
    r = np.array([(sns(m.header) * 1e-9, m.velocity / 3.6) for _, k, m in msgs if k == 'rear'])
    c = np.array([(sns(m.header) * 1e-9, m.position) for _, k, m in msgs if k == 'cmd'])
    fx = []
    for _, k, m in msgs:
        if k == 'master_fix' and m.status.status == 2:
            p = P.to_map(m.latitude, m.longitude, 0)
            cands = TM.project(p[0], p[1], max_dist=1.5)
            if cands:
                d, pid, s_, yaw, side = cands[0]
                pid2, s2, _ = TM.advance(pid, s_, 9.873)
                fx.append((sns(m.header) * 1e-9, slope_at(pid2, s2)))
    if len(f) < 500 or len(r) < 500 or len(c) < 500 or len(fx) < 500:
        return None
    for a in (f, r, c):
        a[:] = a[np.argsort(a[:, 0], kind='stable')]
    fx = np.array(sorted(fx))
    t0 = max(f[0, 0], r[0, 0], c[0, 0], fx[0, 0])
    t1 = min(f[-1, 0], r[-1, 0], c[-1, 0], fx[-1, 0])
    T = np.arange(t0, t1, 0.05)
    vf = np.interp(T, f[:, 0], f[:, 1])
    vr = np.interp(T, r[:, 0], r[:, 1])
    ok = np.abs(vf - vr) < 0.3
    # только там, где есть свежий RTK-фикс (иначе уклон неизвестен)
    idx = np.clip(np.searchsorted(fx[:, 0], T), 1, len(fx) - 1)
    near = np.minimum(np.abs(fx[idx, 0] - T), np.abs(fx[idx - 1, 0] - T)) < 0.3
    ok &= near
    v = 0.5 * (vf + vr)
    k = 20
    vs = np.convolve(v, np.ones(k) / k, mode='same')
    a = np.gradient(vs, T)
    sl = np.interp(T, fx[:, 0], fx[:, 1])
    ci = np.searchsorted(c[:, 0], T - 0.2, side='right') - 1
    cmd = c[np.clip(ci, 0, len(c) - 1), 1]
    return v[ok], a[ok], cmd[ok], sl[ok]


def fit(V, A, C):
    vb = np.digitize(V, V_BINS) - 1
    pred = np.zeros_like(A)
    table = []
    for n in range(-15, 16):
        row = []
        for j in range(len(V_BINS) - 1):
            mm = (C == n) & (vb == j)
            if mm.sum() > 30:
                med = float(np.median(A[mm]))
                pred[mm] = med
                row.append(round(med, 3))
            else:
                row.append(None)
        table.append(row)
    return table, pred


def main():
    rows = [s for s in (series(b) for b in sorted(glob.glob(sys.argv[1] + '/30618_*/'))) if s is not None]
    V = np.concatenate([r[0] for r in rows]); A = np.concatenate([r[1] for r in rows])
    C = np.concatenate([r[2] for r in rows]); S = np.concatenate([r[3] for r in rows])
    print('samples', len(V), 'bags', len(rows), 'slope p5/p95 %.3f/%.3f' % tuple(np.percentile(S, [5, 95])))
    t0, p0 = fit(V, A, C)
    Aflat = A + G * S
    t1, p1 = fit(V, Aflat, C)
    moving = V > 0.5
    print('rms resid (moving): no grade %.3f, with grade %.3f' % (
        np.sqrt(np.mean((A - p0)[moving] ** 2)), np.sqrt(np.mean((Aflat - p1)[moving] ** 2))))
    for row in t1:
        for j in range(len(row)):
            if row[j] is None:
                cand = [(abs(j - jj), row[jj]) for jj in range(len(row)) if row[jj] is not None]
                row[j] = min(cand)[1] if cand else 0.0
    print('traction_table_flat:')
    for n, row in zip(range(-15, 16), t1):
        print('  - %s   # %+d' % (row, n))


if __name__ == '__main__':
    main()
