#!/usr/bin/env python3
"""Ветка пути, которой нет в карте: средний путь на конечной «Таллинская».

После остановки на прибытии часть трамваев (3 из 25 обучающих проездов через стрелку) уходит
со стрелки на петле (loop_tallinskaya, s ≈ 82 м) не по петле, а на средний путь (между петлёй и
нижним путём, ~3.5 м от нижнего) и встаёт там в конце записи. Геометрия восстанавливается по
RTK-псевдоэталону base_link ОБУЧАЮЩИХ записей (проверочная запись не используется):

  * точки после расхождения с петлёй (> 0.3 м от неё), по времени;
  * все проезды приводятся к общей дуговой координате от стрелки и усредняются по бинам 2 м;
  * сглаживание, ресемплинг через 1 м; начало — точка стрелки на петле;
  * высота — с ближайшей точки нижнего пути (он в 3.5 м, высоты выверены), у стрелки — с петли.

    python3 tools/build_branch_track.py <папка_с_записями> [записи через запятую] [--s-switch 82]
Пишет maps/tallinskaya_middle.json (формат pathgraph).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from offline_eval import default_cfg, pseudo_reference, read_bag  # noqa: E402
from tram_backup_odometry.track_map import TrackMap  # noqa: E402

MAPS = HERE.parent / 'maps'
DEFAULT_RUNS = '30618_49fe4c54,30618_8158f0b0,30618_a869780d'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('runs', nargs='?', default=DEFAULT_RUNS)
    ap.add_argument('--s-switch', type=float, default=82.0)
    ap.add_argument('--out', default=str(MAPS / 'tallinskaya_middle.json'))
    args = ap.parse_args()
    tm = TrackMap([str(MAPS / f) for f in ('shchukinskaya_tallinskaya.json', 'tallinskaya_shchukinskaya.json',
                                            'terminal_loops.json')])
    names = [p.name for p in tm.paths]
    loop = names.index('loop_tallinskaya')
    lower = names.index('tallinskaya_lower')
    x0, y0, z0, yaw0, _ = tm.pose(loop, args.s_switch)
    tracks = []
    for run in args.runs.split(','):
        ref = pseudo_reference(read_bag(os.path.join(args.data, run)), default_cfg())
        pts, started = [], False
        for _, ts, x, y, z, _ in ref:
            if x > 99100 or y > 85030:
                continue
            c = {tm.paths[p].name: d for d, p, s, yaw, side in tm.project(x, y, max_dist=40)}
            d_loop = c.get('loop_tallinskaya', 99.0)
            if not started:
                # вперёд от стрелки по ходу петли и уже отошли от неё
                ahead = (x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0)
                started = ahead > 0.0 and d_loop > 0.3
            if started:
                pts.append((x, y))
        if len(pts) > 20:
            tracks.append(pts)
    # общая дуговая координата: проекция на самый длинный проезд
    base = max(tracks, key=len)
    bs = [0.0]
    for (a, b), (c, d) in zip(base, base[1:]):
        bs.append(bs[-1] + math.hypot(c - a, d - b))

    def along(x, y):
        best = None
        for i in range(len(base) - 1):
            ax, ay = base[i]
            bx, by = base[i + 1]
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            u = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
            px, py = ax + u * dx, ay + u * dy
            d2 = (x - px) ** 2 + (y - py) ** 2
            if best is None or d2 < best[0]:
                best = (d2, bs[i] + u * math.sqrt(L2))
        return best[1]
    bins = {}
    for pts in tracks:
        for x, y in pts:
            k = int(along(x, y) // 2.0)
            bins.setdefault(k, []).append((x, y))
    cen = [(sum(p[0] for p in v) / len(v), sum(p[1] for p in v) / len(v)) for k, v in sorted(bins.items())]
    cen = [(x0, y0)] + cen
    # сглаживание (скользящее среднее по 5, концы не трогаем)
    sm = list(cen)
    for i in range(2, len(cen) - 2):
        sm[i] = (sum(c[0] for c in cen[i - 2:i + 3]) / 5, sum(c[1] for c in cen[i - 2:i + 3]) / 5)
    # ресемплинг через 1 м
    out = [sm[0]]
    acc = 0.0
    for (ax, ay), (bx, by) in zip(sm, sm[1:]):
        seg = math.hypot(bx - ax, by - ay)
        t = 1.0 - acc
        while t <= seg:
            out.append((ax + (bx - ax) * t / seg, ay + (by - ay) * t / seg))
            t += 1.0
        acc = seg - (t - 1.0)
    pts_json = []
    for i, (x, y) in enumerate(out):
        c = tm.project(x, y, max_dist=10)
        zl = [tm.pose(p, s)[2] for d, p, s, yaw, side in c if p == lower]
        z = zl[0] if zl and i > 10 else z0
        pts_json.append({'x': round(x, 3), 'y': round(y, 3), 'z': round(z, 3)})
    data = {'points': pts_json,
            'paths': [{'ext_id': 'tallinskaya_middle', 'point_indices': list(range(len(pts_json)))}],
            'meta': {'switch_path': 'loop_tallinskaya', 'switch_s': args.s_switch,
                     'source_runs': args.runs.split(','), 'passes': len(tracks)}}
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=0)
    print('written %s: %d points (%.0f m), passes %d, start (%.1f, %.1f), end (%.1f, %.1f)' % (
        args.out, len(pts_json), len(pts_json) - 1, len(tracks), out[0][0], out[0][1], out[-1][0], out[-1][1]))


if __name__ == '__main__':
    main()
