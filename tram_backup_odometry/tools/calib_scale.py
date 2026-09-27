#!/usr/bin/env python3
"""Калибровка масштаба колёсной скорости по обучающим записям (RTK GNSS + карта).

Для каждой записи: точки антенны master со status=2 проецируются на выданные пути
(ST/TS), берётся дуговая координата s(t); колёсный путь D(t) = интеграл (front+rear)/2 * scale0.
Линейная регрессия s = k*D + c на участке одного пути -> k (поправка масштаба).
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from offline_eval import read_bag, sns  # noqa: E402
from tram_backup_odometry.geo import TransverseMercator  # noqa: E402
from tram_backup_odometry.track_map import TrackMap  # noqa: E402

P = TransverseMercator()
tm = TrackMap([str(HERE.parent / 'maps' / f) for f in
               ('shchukinskaya_tallinskaya.json', 'tallinskaya_shchukinskaya.json')])


def one(bag):
    msgs = read_bag(bag)
    f = np.array([(sns(m.header) * 1e-9, m.velocity / 3.6) for _, k, m in msgs if k == 'front'])
    r = np.array([(sns(m.header) * 1e-9, m.velocity / 3.6) for _, k, m in msgs if k == 'rear'])
    if len(f) < 100 or len(r) < 100:
        return None
    T = np.sort(np.concatenate([f[:, 0], r[:, 0]]))
    v = 0.5 * (np.interp(T, f[:, 0], f[:, 1]) + np.interp(T, r[:, 0], r[:, 1]))
    D = np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(T))])
    pts = []
    for _, k, m in msgs:
        if k != 'master_fix' or m.status.status != 2:
            continue
        p = P.to_map(m.latitude, m.longitude, m.altitude)
        c = tm.project(p[0], p[1], max_dist=1.0)
        if not c:
            continue
        d, pid, s, yaw, side = c[0]
        if 20 < s < tm.length(pid) - 20:
            pts.append((sns(m.header) * 1e-9, pid, s))
    if len(pts) < 200:
        return None
    pts = np.array(pts)
    res = []
    for pid in (0, 1):
        q = pts[pts[:, 1] == pid]
        if len(q) < 200 or q[-1, 2] - q[0, 2] < 1500:
            continue
        Dq = np.interp(q[:, 0], T, D)
        A = np.column_stack([Dq, np.ones(len(Dq))])
        k, c = np.linalg.lstsq(A, q[:, 2], rcond=None)[0]
        resid = q[:, 2] - (k * Dq + c)
        # робастно: повтор без выбросов
        ok = np.abs(resid) < 3 * max(np.std(resid), 0.3)
        k, c = np.linalg.lstsq(A[ok], q[ok, 2], rcond=None)[0]
        resid = q[ok, 2] - (k * Dq[ok] + c)
        res.append({'pid': int(pid), 'k': float(k), 'span_m': float(q[-1, 2] - q[0, 2]),
                    'resid_std': float(np.std(resid)), 'n': int(ok.sum())})
    return res


if __name__ == '__main__':
    out = {}
    for b in sorted(glob.glob(sys.argv[1] + '/*/')):
        name = os.path.basename(b.rstrip('/'))
        try:
            r = one(b)
        except Exception as e:  # noqa: BLE001
            r = None
            print(name, 'ERR', e)
        if r:
            out[name] = r
            print(name, ' '.join('pid%d k=%.5f span=%.0f std=%.2f' % (x['pid'], x['k'], x['span_m'], x['resid_std']) for x in r), flush=True)
    os.makedirs('out', exist_ok=True)
    json.dump(out, open(os.path.join('out', 'calib_scale.json'), 'w'), indent=1)
    for veh in ('30618', '30639'):
        ks = [x['k'] for n, rr in out.items() if n.startswith(veh) for x in rr if x['resid_std'] < 1.5]
        if ks:
            print(veh, 'n=%d median k=%.5f mean=%.5f std=%.5f' % (len(ks), np.median(ks), np.mean(ks), np.std(ks)))
