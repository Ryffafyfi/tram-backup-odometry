#!/usr/bin/env python3
"""Разброс отклика на позицию контроллера: «коридор» ускорения для детектора проскальзывания.

Для каждой ячейки (позиция, интервал скорости) — насколько измеренное ускорение (среднее тележек,
сглаживание 0.5 с, запаздывание команды 0.2 с) может уходить от медианы в сторону проскальзывания:
при тяге — p95 − медиана (боксование разгоняет колёса), при торможении — медиана − p05 (юз
замедляет), на выбеге — большее из двух. Детектор проскальзывания обеих тележек срабатывает только
за пределами этого коридора, поэтому неоднозначные позиции (например, −8: ~1 с тормозит, потом
держит скорость) не дают ложных срабатываний.

    python3 fit_traction_envelope.py <папка с записями> [--list runs.txt] [--q 0.05]
Печатает `traction_spread_table` для config/baseline_params.yaml.
"""
import argparse
import glob
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from offline_eval import read_bag, sns  # noqa: E402

V_BINS = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 30.0]
LAG = 0.2


def series(bag):
    try:
        msgs = read_bag(bag)
    except Exception:   # noqa: BLE001
        return None
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
    vs = np.convolve(v, np.ones(10) / 10, mode='same')
    a = np.gradient(vs, T)
    ci = np.clip(np.searchsorted(c[:, 0], T - LAG, side='right') - 1, 0, len(c) - 1)
    cmd = c[ci, 1]
    return v[ok], a[ok], cmd[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--list', default='')
    ap.add_argument('--q', type=float, default=0.05)
    args = ap.parse_args()
    if args.list:
        bags = [os.path.join(args.data, x.strip()) for x in open(args.list) if x.strip()]
    else:
        bags = sorted(glob.glob(os.path.join(args.data, '3*_*/')))
    with mp.Pool(2) as pool:
        rows = [s for s in pool.map(series, bags) if s is not None]
    V = np.concatenate([r[0] for r in rows])
    A = np.concatenate([r[1] for r in rows])
    C = np.concatenate([r[2] for r in rows])
    vb = np.digitize(V, V_BINS) - 1
    q = 100.0 * args.q
    table = []
    for n in range(-15, 16):
        row = []
        for j in range(len(V_BINS) - 1):
            mm = (C == n) & (vb == j)
            if mm.sum() > 30:
                med = np.median(A[mm])
                hi = np.percentile(A[mm], 100 - q) - med
                lo = med - np.percentile(A[mm], q)
                row.append(round(float(hi if n > 0 else lo if n < 0 else max(hi, lo)), 3))
            else:
                row.append(None)
        table.append(row)
    for row in table:
        known = [x for x in row if x is not None]
        fill = max(known) if known else 0.5
        for j in range(len(row)):
            if row[j] is None:
                row[j] = fill
    print('# bags %d, samples %d, quantile %.2f' % (len(rows), len(V), args.q))
    print('traction_spread_table:   # строки: позиция -15..15, столбцы: интервалы скорости traction_v_bins')
    for n, row in zip(range(-15, 16), table):
        print('  - %s   # %+d' % (row, n))


if __name__ == '__main__':
    main()
