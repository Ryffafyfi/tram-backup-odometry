#!/usr/bin/env python3
"""Исправление высот пути tallinskaya_lower в maps/terminal_loops.json.

Путь был восстановлен по записи 30618_21dd3af3, у которой высота RTK на этом участке
«уплыла» на 2.7 м (плановое положение при этом верное — длина пути совпадает с колёсами
в пределах 0.2 м на 20 м). Ложный «подъём» 14 % в конце пути ломал модель тяги с уклоном.

Исправление: на участке, где путь идёт по обратной ветви петли loop_tallinskaya, высота
берётся с петли; на своём участке — по высотам RTK (status 2) другой записи (30618_f19a4ac3,
медиана в окне ±2 м); в начале (где второй записи нет) — плавная стыковка с исходной высотой.

    python3 tools/fix_lower_z.py [--bag /path/30618_f19a4ac3]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from offline_eval import read_bag  # noqa: E402
from tram_backup_odometry.geo import TransverseMercator  # noqa: E402
from tram_backup_odometry.track_map import TrackMap  # noqa: E402

P = TransverseMercator()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True,
                    help='запись с верной высотой RTK на нижнем пути Таллинской (в работе: 30618_f19a4ac3)')
    ap.add_argument('--file', default=str(HERE.parent / 'maps' / 'terminal_loops.json'))
    a = ap.parse_args()
    data = json.load(open(a.file))
    tm = TrackMap([a.file])
    low = [i for i, p in enumerate(tm.paths) if p.name == 'tallinskaya_lower'][0]
    loop = [i for i, p in enumerate(tm.paths) if p.name == 'loop_tallinskaya'][0]
    idx = [p for p in data['paths'] if p['ext_id'] == 'tallinskaya_lower'][0]['point_indices']
    pts = [data['points'][i] for i in idx]
    s_pts = np.r_[0, np.cumsum([math.hypot(b['x'] - a_['x'], b['y'] - a_['y'])
                                for a_, b in zip(pts[:-1], pts[1:])])]
    # высоты RTK второй записи вдоль пути: высота антенны master - 3 м = высота рельса под ней
    samples = []
    for t, k, m in read_bag(a.bag):
        if k != 'master_fix' or m.status.status != 2:
            continue
        q = P.to_map(m.latitude, m.longitude, m.altitude)
        c = [cc for cc in tm.project(q[0], q[1], max_dist=1.0) if cc[1] == low]
        if c:
            samples.append((c[0][2], q[2] - 3.0))   # высота рельса в точке, где стоит антенна
    samples = np.array(samples)
    z_old = np.array([p['z'] for p in pts])
    z_new = z_old.copy()
    have = np.zeros(len(pts), bool)
    for i, (s, p) in enumerate(zip(s_pts, pts)):
        # участок по обратной ветви петли: высота с петли
        c = [cc for cc in tm.project(p['x'], p['y'], max_dist=1.0) if cc[1] == loop]
        if c:
            z_new[i] = tm.pose(loop, c[0][2])[2]
            have[i] = True
            continue
        sel = samples[np.abs(samples[:, 0] - s) < 2.0] if len(samples) else []
        if len(sel) >= 3:
            z_new[i] = float(np.median(sel[:, 1]))
            have[i] = True
    # стыковка: где данных нет — исходная высота со сдвигом, плавно переходящим к ближайшей
    # точке с данными
    first = int(np.argmax(have))
    if have.any() and first > 0:
        off = z_new[first] - z_old[first]
        for i in range(first):
            z_new[i] = z_old[i] + off * (s_pts[i] / max(s_pts[first], 1e-6))
    # сглаживание окном 5 точек (~5 м), концы на месте
    zs = z_new.copy()
    for i in range(2, len(zs) - 2):
        z_new[i] = zs[i - 2:i + 3].mean()
    for p, z in zip(pts, z_new):
        p['z'] = round(float(z), 5)
    json.dump(data, open(a.file, 'w'), indent=1)
    for s in range(0, int(s_pts[-1]) + 1, 20):
        i = int(np.argmin(np.abs(s_pts - s)))
        print('s=%5.1f z %.2f -> %.2f' % (s_pts[i], z_old[i], z_new[i]))


if __name__ == '__main__':
    main()
