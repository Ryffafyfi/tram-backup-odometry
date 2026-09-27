#!/usr/bin/env python3
"""Табличная модель продольной динамики a(позиция контроллера, скорость) по обучающим записям.

Ускорение берётся из колёсной скорости (среднее тележек, 1/3.6), сглаженной по 1 с, с учётом
запаздывания отклика тяги (ищется сдвиг команды, дающий лучшую предсказуемость).
Результат (медиана по ячейкам) пишется в YAML для config/baseline_params.yaml: traction_table.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from offline_eval import read_bag, sns  # noqa: E402

V_BINS = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 30.0]


def series(bag):
    msgs = read_bag(bag)
    f = np.array([(sns(m.header) * 1e-9, m.velocity / 3.6) for _, k, m in msgs if k == 'front'])
    r = np.array([(sns(m.header) * 1e-9, m.velocity / 3.6) for _, k, m in msgs if k == 'rear'])
    c = np.array([(sns(m.header) * 1e-9, m.position) for _, k, m in msgs if k == 'cmd'])
    if len(f) < 500 or len(r) < 500 or len(c) < 500:
        return None
    for a in (f, r, c):
        a[:] = a[np.argsort(a[:, 0], kind='stable')]
    t0 = max(f[0, 0], r[0, 0], c[0, 0])
    t1 = min(f[-1, 0], r[-1, 0], c[-1, 0])
    T = np.arange(t0, t1, 0.05)
    vf = np.interp(T, f[:, 0], f[:, 1])
    vr = np.interp(T, r[:, 0], r[:, 1])
    ok = np.abs(vf - vr) < 0.3
    v = 0.5 * (vf + vr)
    # сглаживание 1 c и производная
    k = 20
    vs = np.convolve(v, np.ones(k) / k, mode='same')
    a = np.gradient(vs, T)
    idx = np.searchsorted(c[:, 0], T, side='right') - 1
    cmd = c[np.clip(idx, 0, len(c) - 1), 1]
    return T, v, a, cmd, ok


def main():
    data = sys.argv[1]
    lag_steps = [0, 4, 8, 12, 16, 20, 30]   # по 0.05 c
    acc = {L: [] for L in lag_steps}
    rows = []
    for b in sorted(glob.glob(data + '/30618_*/')):
        s = series(b)
        if s is None:
            continue
        rows.append(s)
    print('bags', len(rows))
    best = None
    for L in lag_steps:
        cells = {}
        allv, alla, allc = [], [], []
        for T, v, a, cmd, ok in rows:
            cs = np.roll(cmd, L)   # команда L шагов назад
            m = ok.copy()
            m[:L] = False
            allv.append(v[m]); alla.append(a[m]); allc.append(cs[m])
        V = np.concatenate(allv); A = np.concatenate(alla); C = np.concatenate(allc)
        vb = np.digitize(V, V_BINS) - 1
        pred = np.zeros_like(A)
        for n in range(-15, 16):
            for j in range(len(V_BINS) - 1):
                mm = (C == n) & (vb == j)
                if mm.sum() > 30:
                    pred[mm] = np.median(A[mm])
        err = np.sqrt(np.mean((A - pred) ** 2))
        print('lag %.2fs rms err %.3f' % (L * 0.05, err))
        if best is None or err < best[0]:
            best = (err, L, V, A, C, vb)
    err, L, V, A, C, vb = best
    print('best lag %.2f s' % (L * 0.05))
    table = []
    for n in range(-15, 16):
        row = []
        for j in range(len(V_BINS) - 1):
            mm = (C == n) & (vb == j)
            row.append(round(float(np.median(A[mm])), 3) if mm.sum() > 30 else None)
        table.append(row)
    # заполнение пропусков: ближайшее по скорости в той же строке, затем по позиции
    for i, row in enumerate(table):
        for j in range(len(row)):
            if row[j] is None:
                cand = [(abs(j - jj), row[jj]) for jj in range(len(row)) if row[jj] is not None]
                row[j] = min(cand)[1] if cand else None
    for i in range(len(table)):
        if all(x is None for x in table[i]):
            near = min((abs(i - ii), ii) for ii in range(len(table)) if table[ii][0] is not None)[1]
            table[i] = list(table[near])
    print('traction_lag_sec: %.2f' % (L * 0.05))
    print('traction_v_bins: %s' % V_BINS)
    print('traction_table:   # строки: позиция -15..15, столбцы: интервалы скорости')
    for n, row in zip(range(-15, 16), table):
        print('  - %s   # %+d' % (row, n))


if __name__ == '__main__':
    main()
