#!/usr/bin/env python3
"""Места остановок (стоп-линии, платформы) по обучающим записям: кластеры вдоль путей карты.

Остановка: скорость тележек < 0.05 м/с не меньше 5 с. Положение base_link на остановке —
по RTK-фиксам master (медиана), проекция на пути карты (включая петли). Кластеризация вдоль
дуги каждого пути (порог 1.5 м, не менее 4 остановок). Пишет maps/stop_landmarks.json.

    python3 tools/stop_landmarks.py <папка_с_записями> [--reuse]
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
M = HERE.parent / 'maps'
tm = TrackMap([str(M / f) for f in ('shchukinskaya_tallinskaya.json', 'tallinskaya_shchukinskaya.json',
                                    'terminal_loops.json')])
OFF = 9.873


def stops_of(bag):
    msgs = read_bag(bag)
    f = np.array([(sns(m.header) * 1e-9, m.velocity / 3.6) for _, k, m in msgs if k == 'front'])
    r = np.array([(sns(m.header) * 1e-9, m.velocity / 3.6) for _, k, m in msgs if k == 'rear'])
    fx = [(sns(m.header) * 1e-9, m) for _, k, m in msgs if k == 'master_fix' and m.status.status == 2]
    rv = [(sns(m.header) * 1e-9, m) for _, k, m in msgs if k == 'rover_fix' and m.status.status == 2]
    if len(f) < 100 or len(fx) < 100:
        return []
    T = np.arange(max(f[0, 0], r[0, 0]), min(f[-1, 0], r[-1, 0]), 0.1)
    v = 0.5 * (np.interp(T, f[:, 0], f[:, 1]) + np.interp(T, r[:, 0], r[:, 1]))
    still = v < 0.05
    out = []
    i = 0
    n = len(T)
    ft = np.array([t for t, _ in fx])
    rt = np.array([t for t, _ in rv]) if rv else np.array([])
    while i < n:
        if not still[i]:
            i += 1
            continue
        j = i
        while j < n and still[j]:
            j += 1
        if T[j - 1] - T[i] >= 5.0:
            a, b = T[i] + 1.0, T[j - 1] - 1.0
            sel = [m for t, m in fx if a <= t <= b]
            selr = [m for t, m in rv if a <= t <= b]
            if len(sel) >= 10 and len(selr) >= 10:
                pm = np.median([P.to_map(m.latitude, m.longitude, 0)[:2] for m in sel], axis=0)
                pr = np.median([P.to_map(m.latitude, m.longitude, 0)[:2] for m in selr], axis=0)
                yaw = float(np.arctan2(pr[1] - pm[1], pr[0] - pm[0]))
                c = tm.project(pm[0], pm[1], max_dist=2.0)
                for d, pid, s_ant, ypath, _ in c:
                    dh = abs((ypath - yaw + np.pi) % (2 * np.pi) - np.pi)
                    if dh < np.radians(45):
                        pid2, s_bl, _ = tm.advance(pid, s_ant, OFF)
                        out.append((tm.paths[pid2].name, float(s_bl), float(T[j - 1] - T[i])))
                        break
        i = j
    return out


CLUSTER_GAP = 1.5     # м — остановки ближе этого к соседней относятся к одному месту
MIN_STOPS = 4         # место остановки должно встретиться хотя бы столько раз
MERGE_DIST = 4.0      # более слабое место ближе этого к сильному — отбрасывается (неразличимы)


def cluster(allst):
    """Кластеры мест остановок вдоль каждого пути (одиночная связь с малым порогом).

    Порог мал (1.5 м): у некоторых платформ два устойчивых места остановки в 5–6 м друг от
    друга (например, ST 2410.3 и 2415.6) — это два разных ориентира, а не один «размытый».
    """
    res = []
    for name in sorted({s[1] for s in allst}):
        ss = sorted(s[2] for s in allst if s[1] == name)
        cl = []
        for s in ss:
            if cl and s - cl[-1][-1] <= CLUSTER_GAP:
                cl[-1].append(s)
            else:
                cl.append([s])
        cand = []
        for c in cl:
            if len(c) >= MIN_STOPS:
                c = np.array(c)
                cand.append({'track': name, 's': float(np.median(c)), 'n': int(len(c)),
                             'std': float(np.std(c)),
                             'mad': float(np.median(np.abs(c - np.median(c))))})
        cand.sort(key=lambda r: -r['n'])
        kept = []
        for r in cand:
            if all(abs(r['s'] - k['s']) >= MERGE_DIST for k in kept):
                kept.append(r)
        res += sorted(kept, key=lambda r: r['s'])
    return res


def main():
    raw = os.path.join('out', 'stops_raw.json')
    if len(sys.argv) > 2 and sys.argv[2] == '--reuse' and os.path.exists(raw):
        allst = json.load(open(raw))
    else:
        allst = []
        for b in sorted(glob.glob(sys.argv[1] + '/30618_*/')):
            st = stops_of(b)
            allst += [(os.path.basename(b.rstrip('/')),) + s for s in st]
            print(os.path.basename(b.rstrip('/')), len(st), flush=True)
        os.makedirs(os.path.dirname(raw), exist_ok=True)
        json.dump(allst, open(raw, 'w'))
    res = cluster(allst)
    for r in res:
        print('%-28s s=%8.1f n=%3d std=%.2f mad=%.2f' % (r['track'], r['s'], r['n'], r['std'], r['mad']))
    json.dump(res, open(str(M / 'stop_landmarks.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
