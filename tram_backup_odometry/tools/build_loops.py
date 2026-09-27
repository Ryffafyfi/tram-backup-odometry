#!/usr/bin/env python3
"""Достройка карты: разворотные петли у конечных (их нет в выданном pathgraph).

Каждая запись начинается/заканчивается на конечной — на петле, которой нет в картах
«щукинская - таллинская» / «таллинская - щукинская» (в проверочной записи первые ~550 м
пути — вне карты). Петли восстанавливаются по GNSS-антенне master ОБУЧАЮЩИХ записей
(проверочная запись не используется — на ней меряем):

  * точки master (status >= 0) -> система карты (UTM 37N - (300000, 6100000));
  * берутся участки вне выданных путей, упорядоченные по ходу движения;
  * прибытие (конец пути -> остановка) и отправление (остановка -> начало пути) сшиваются;
  * ресемплинг через 1 м, сглаживание, поправка на вынос антенны на кривой: антенна master
    стоит на оси кузова в 2.323 м за осью задней тележки, на кривой радиуса R она смещена
    наружу на e*(L+e)/(2R) ~ 11.47/R м (L = 7.55 м между тележками) — сдвигаем внутрь;
  * z = высота антенны - 3.0 м.

Результат — файл pathgraph того же формата (maps/terminal_loops.json) с путями:
  loop_shchukinskaya: конец «таллинская-щукинская» -> петля -> начало «щукинская-таллинская»
  loop_tallinskaya:   конец «щукинская-таллинская» -> петля -> начало «таллинская-щукинская»
  tallinskaya_lower:  второй (нижний) путь отправления на Таллинской -> начало «таллинская-щукинская»
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tram_backup_odometry.geo import TransverseMercator  # noqa: E402
from tram_backup_odometry.track_map import TrackMap  # noqa: E402

sys.path.insert(0, str(HERE))
from offline_eval import read_bag  # noqa: E402

PROJ = TransverseMercator()
ANT_H = 3.0
K_MASTER = 2.323 * 9.873 / 2.0   # смещение антенны наружу = K * кривизна


def master_track(bag):
    pts = []
    for t, kind, m in read_bag(bag):
        if kind != 'master_fix' or m.status.status != 2:   # только RTK (status 2): status 0/1 смещены на метры
            continue
        p = PROJ.to_map(m.latitude, m.longitude, m.altitude)
        if p is None:
            continue
        pts.append((t * 1e-9, p[0], p[1], p[2] - ANT_H, m.status.status))
    return np.array(pts)


def dedup_moving(P, min_step=0.25):
    """Оставить точки с шагом >= min_step м (убрать «облако» на стоянке)."""
    out = [P[0]]
    for p in P[1:]:
        if math.hypot(p[1] - out[-1][1], p[2] - out[-1][2]) >= min_step:
            out.append(p)
    return np.array(out)


def off_map_mask(P, tm, tol=1.2, end_margin=1.0):
    mask = []
    for p in P:
        c = tm.project(p[1], p[2], max_dist=30.0)
        if not c:
            mask.append(True)
            continue
        d, pid, s, yaw, side = c[0]
        L = tm.length(pid)
        mask.append(d > tol or s < end_margin or s > L - end_margin)
    return np.array(mask)


def resample(xy, step=1.0):
    d = np.r_[0, np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))]
    keep = np.r_[True, np.diff(d) > 1e-6]
    xy, d = xy[keep], d[keep]
    n = max(2, int(d[-1] / step) + 1)
    s = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(s, d, xy[:, k]) for k in range(xy.shape[1])])


def smooth(xy, win=5, fixed_ends=True):
    out = xy.copy()
    h = win // 2
    for i in range(len(xy)):
        a, b = max(0, i - h), min(len(xy), i + h + 1)
        out[i] = xy[a:b].mean(axis=0)
    if fixed_ends:
        out[0], out[-1] = xy[0], xy[-1]
    return out


def curvature(xy):
    x, y = xy[:, 0], xy[:, 1]
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)


def antenna_to_centerline(xyz, k=K_MASTER):
    """Сдвинуть трек антенны внутрь кривой на k*kappa."""
    kap = curvature(xyz[:, :2])
    kap = np.convolve(kap, np.ones(7) / 7, mode='same')
    dx, dy = np.gradient(xyz[:, 0]), np.gradient(xyz[:, 1])
    n = np.hypot(dx, dy)
    nx, ny = -dy / n, dx / n                # левая нормаль
    shift = k * kap                         # kappa>0 — поворот влево, центр слева
    out = xyz.copy()
    out[:, 0] += nx * shift
    out[:, 1] += ny * shift
    return out


def join(a, b, tol=2.0):
    """a затем b: обрезать a в точке, ближайшей к началу b."""
    d = np.hypot(a[:, 0] - b[0, 0], a[:, 1] - b[0, 1])
    i = int(np.argmin(d))
    if d[i] > tol:
        print('  WARNING: join gap %.2f m' % d[i])
    return np.vstack([a[:i], b])


def piece(bag, which, tm, t_window=600.0):
    """which: 'start' — от начала записи до выхода на путь; 'end' — от схода с пути до конца."""
    P = master_track(bag)
    if which == 'start':
        P = P[P[:, 0] < P[0, 0] + t_window]
    else:
        P = P[P[:, 0] > P[-1, 0] - t_window]
    P = dedup_moving(P)
    m = off_map_mask(P, tm)
    if which == 'start':
        i = int(np.argmin(m)) if not m.all() else len(m)   # первая точка на пути
        seg = P[:i]
    else:
        idx = np.where(~m)[0]
        i = idx[-1] + 1 if len(idx) else 0                 # после последней точки на пути
        seg = P[i:]
    print('  %s %s: %d pts, %.0f m' % (Path(bag).name, which, len(seg),
                                        np.hypot(np.diff(seg[:, 1]), np.diff(seg[:, 2])).sum()
                                        if len(seg) > 1 else 0))
    return seg[:, 1:4]


def refine_z(xyz, bags, data_dir, radius=1.5):
    """z по медиане высот RTK-фиксов master из нескольких записей (меньше шума, чем по одной)."""
    from tram_backup_odometry.track_map import TrackPath
    path = TrackPath('tmp', xyz[:, 0], xyz[:, 1], xyz[:, 2])
    tmp = TrackMap([])
    tmp.paths = [path]
    tmp.finalize()
    samples = [[] for _ in range(len(xyz))]
    for b in bags:
        P = master_track(str(Path(data_dir) / b))
        for p in P:
            c = tmp.project(p[1], p[2], max_dist=radius)
            if c:
                i = int(round(c[0][2]))
                if 0 <= i < len(samples):
                    samples[i].append(p[3])
    out = xyz.copy()
    n_ok = 0
    for i, sm in enumerate(samples):
        if len(sm) >= 5:
            out[i, 2] = float(np.median(sm))
            n_ok += 1
    # сглаживание по z (окно 5 м), концы оставляем как у стыкуемых путей
    z = out[:, 2].copy()
    for i in range(2, len(z) - 2):
        out[i, 2] = z[i - 2:i + 3].mean()
    print('  refine_z: %d/%d points from %d bags' % (n_ok, len(xyz), len(bags)))
    return out


def to_pathgraph(named):
    pts, paths = [], []
    for name, xyz in named:
        idx = []
        kap = curvature(xyz[:, :2])
        for i, (x, y, z) in enumerate(xyz):
            j = min(i, len(xyz) - 2)
            tang = math.atan2(xyz[j + 1, 1] - xyz[j, 1], xyz[j + 1, 0] - xyz[j, 0])
            idx.append(len(pts))
            pts.append({'x': round(float(x), 5), 'y': round(float(y), 5), 'z': round(float(z), 5),
                        'tang': tang, 'curv': float(kap[i])})
        paths.append({'ext_id': name, 'point_indices': idx})
    return {'points': pts, 'paths': paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True,
                    help='папка с распакованными записями датасета (rosbag2), например data/bags')
    ap.add_argument('--maps', default=str(HERE.parent / 'maps'))
    ap.add_argument('--out', default=str(HERE.parent / 'maps' / 'terminal_loops.json'))
    ap.add_argument('--sh-arrive', default='30618_073f08d1')
    ap.add_argument('--sh-depart', default='30618_01f73500')
    ap.add_argument('--ta-arrive', default='30618_0652866c')
    ap.add_argument('--ta-depart', default='30618_073f08d1')
    ap.add_argument('--ta-lower', default='30618_21dd3af3')
    # уточнение z по медиане нескольких записей (выключено: по проверочной записи хуже, чем z одной записи)
    ap.add_argument('--z-bags', default='')
    a = ap.parse_args()
    maps = Path(a.maps)
    st = maps / 'shchukinskaya_tallinskaya.json'
    ts = maps / 'tallinskaya_shchukinskaya.json'
    tm = TrackMap([str(st), str(ts)])
    P_st, P_ts = tm.paths[0], tm.paths[1]
    D = Path(a.data)

    def ends(path):
        return (np.array([path.xs[0], path.ys[0], path.zs[0]]),
                np.array([path.xs[-1], path.ys[-1], path.zs[-1]]))
    st0, st1 = ends(P_st)
    ts0, ts1 = ends(P_ts)

    def trim(raw, start_pt, start_yaw, end_pt, end_yaw, zone=40.0, gap=1.0):
        # убрать ведущие точки «позади» конца предыдущего пути и хвостовые — «за» началом следующего
        u0 = np.array([math.cos(start_yaw), math.sin(start_yaw)])
        u1 = np.array([math.cos(end_yaw), math.sin(end_yaw)])
        i = 0
        while i < len(raw) and np.hypot(*(raw[i, :2] - start_pt[:2])) < zone and \
                (raw[i, :2] - start_pt[:2]) @ u0 < gap:
            i += 1
        j = len(raw)
        while j > i and np.hypot(*(raw[j - 1, :2] - end_pt[:2])) < zone and \
                (end_pt[:2] - raw[j - 1, :2]) @ u1 < gap:
            j -= 1
        print('  trim: %d leading, %d trailing points' % (i, len(raw) - j))
        return raw[i:j]

    def build(arrive, depart, start_pt, end_pt, start_yaw, end_yaw):
        arr = piece(str(D / arrive), 'end', tm)
        dep = piece(str(D / depart), 'start', tm)
        raw = join(arr, dep)
        raw = trim(raw, start_pt, start_yaw, end_pt, end_yaw)
        raw = np.vstack([start_pt, raw, end_pt])
        c = resample(raw, 1.0)
        c = smooth(c, 5)
        c = antenna_to_centerline(c)
        c[0], c[-1] = start_pt, end_pt
        c = resample(c, 1.0)
        return c

    print('Щукинская:')
    sh = build(a.sh_arrive, a.sh_depart, ts1, st0, P_ts.seg_yaw[-1], P_st.seg_yaw[0])
    print('Таллинская:')
    ta = build(a.ta_arrive, a.ta_depart, st1, ts0, P_st.seg_yaw[-1], P_ts.seg_yaw[0])
    print('Таллинская, нижний путь отправления:')
    low = piece(str(D / a.ta_lower), 'start', tm)
    low = trim(low, low[0], 0.0, ts0, P_ts.seg_yaw[0], zone=0.0)
    u1 = np.array([math.cos(P_ts.seg_yaw[0]), math.sin(P_ts.seg_yaw[0])])
    keep = [k for k in range(len(low))
            if not (np.hypot(*(low[k, :2] - ts0[:2])) < 40 and (ts0[:2] - low[k, :2]) @ u1 < 1.0)]
    low = np.vstack([low[keep], ts0])
    low = resample(low, 1.0)
    low = smooth(low, 5)
    low = antenna_to_centerline(low)
    low[-1] = ts0
    low = resample(low, 1.0)
    if a.z_bags:
        zb = [b.strip() for b in a.z_bags.split(',') if b.strip()]
        sh = refine_z(sh, zb, a.data)
        ta = refine_z(ta, zb, a.data)
        low = refine_z(low, zb, a.data)
    out = to_pathgraph([('loop_shchukinskaya', sh), ('loop_tallinskaya', ta),
                        ('tallinskaya_lower', low)])
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    for name, c in (('loop_shchukinskaya', sh), ('loop_tallinskaya', ta), ('tallinskaya_lower', low)):
        L = np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])).sum()
        print('%s: %d pts, %.1f m' % (name, len(c), L))
    print('written', a.out)


if __name__ == '__main__':
    main()
