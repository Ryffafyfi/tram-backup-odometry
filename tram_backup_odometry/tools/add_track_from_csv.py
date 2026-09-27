#!/usr/bin/env python3
"""Добавить в maps/terminal_loops.json путь, вырезанный из линии пути (track_*.csv).

Нужен для второго пути отправления на конечной «Таллинская»: большинство записей (18 из 26
обучающих, начинающихся там) стоят на пути, параллельном обратной ветви петли в ~4.9 м от неё,
которого не было в карте -> на стоянке и первых ~100 м ошибка поперёк пути ~5 м.
Путь берётся из линии пути track_*.csv (построена по GNSS многих записей) по диапазону s и, при
необходимости, продлевается назад по прямой (трамваи стоят и до первой точки линии).

    python3 tools/add_track_from_csv.py <track_*.csv> <имя_пути> <s_от> <s_до> [--extend-back м]
"""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('name')
    ap.add_argument('s_from', type=float)
    ap.add_argument('s_to', type=float)
    ap.add_argument('--extend-back', type=float, default=0.0)
    ap.add_argument('--file', default=str(HERE.parent / 'maps' / 'terminal_loops.json'))
    a = ap.parse_args()
    rows = []
    with open(a.csv, encoding='utf-8-sig') as fh:
        head = fh.readline().strip().split(',')
        for line in fh:
            d = dict(zip(head, line.strip().split(',')))
            s = float(d['s'])
            if a.s_from <= s <= a.s_to:
                rows.append((s, float(d['x']), float(d['y']), float(d['z'])))
    rows.sort()
    pts = [(x, y, z) for _, x, y, z in rows]
    if a.extend_back > 0 and len(pts) >= 6:
        # направление по первым ~5 м, продление назад по прямой с шагом 1 м
        x0, y0, z0 = pts[0]
        x1, y1, _ = pts[5]
        d = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        ext = [(x0 - ux * k, y0 - uy * k, z0) for k in range(int(a.extend_back), 0, -1)]
        pts = ext + pts
    data = json.load(open(a.file))
    data['paths'] = [p for p in data['paths'] if p.get('ext_id') != a.name]
    base = len(data['points'])
    for i, (x, y, z) in enumerate(pts):
        j = min(i, len(pts) - 2)
        tang = math.atan2(pts[j + 1][1] - pts[j][1], pts[j + 1][0] - pts[j][0])
        data['points'].append({'x': round(x, 5), 'y': round(y, 5), 'z': round(z, 5),
                               'tang': tang, 'curv': 0.0})
    data['paths'].append({'ext_id': a.name, 'point_indices': list(range(base, base + len(pts)))})
    json.dump(data, open(a.file, 'w'), indent=1)
    L = sum(math.hypot(b[0] - c[0], b[1] - c[1]) for b, c in zip(pts[:-1], pts[1:]))
    print('%s: %d точек, %.1f м -> %s' % (a.name, len(pts), L, a.file))


if __name__ == '__main__':
    main()
