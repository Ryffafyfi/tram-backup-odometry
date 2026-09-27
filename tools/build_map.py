"""Карта в удобном виде: для каждого направления map/track_<направление>.csv.

Столбцы: s (м вдоль пути), x, y, z (уровень рельса), heading (рад, от оси x против часовой),
curvature (1/м, + — поворот влево), grade (уклон dz/ds, сглажен на ~20 м), source.
pathgraph не покрывает разворотные кольца, с которых начинаются и которыми кончаются записи,
поэтому куски до начала линии (s < 0) и после конца (s > длины) достраиваются по GNSS
обучающих записей (source = gnss_loop): точки base_link усредняются по записям вдоль пути.
Запуск: python tools/build_map.py
"""
import sys

import numpy as np
import pandas as pd

from geo import ANTENNA_BASELINE, ANTENNAS, lla_to_map_np
from track import DIRECTIONS, MAP_DIR, PATHGRAPH_DIR, Track, detect_direction, load_pathgraph

sys.path.insert(0, str(PATHGRAPH_DIR.parent / 'analysis'))
from catalog import OUT_DIR, load  # noqa: E402

JOIN_DIST = 1.5     # м: точка считается «на линии pathgraph»
JOIN_S = 40.0       # м: насколько близко к концу линии должна быть точка стыка
MIN_RUNS = 5        # минимум записей для точки кольца
MAX_SPREAD = 3.0    # м: медианный разброс записей, после которого кольцо дальше не продлеваем
GRADE_WINDOW = 21   # точек (~20 м) для сглаживания уклона


def base_link_points(bag):
    """Точки base_link по записи: rover и master одновременно, оба со статусом 2, база ~12,44 м."""
    d = load(bag)
    r, m = d['rover_fix'], d['master_fix']
    if not len(r) or not len(m):
        return None
    r = r[r.status == 2].sort_values('t_hdr')
    m = m[m.status == 2].sort_values('t_hdr')
    mr = pd.merge_asof(r[['t_hdr', 'lat', 'lon', 'alt']], m[['t_hdr', 'lat', 'lon', 'alt']],
                       on='t_hdr', tolerance=20_000_000, direction='nearest', suffixes=('_r', '_m')).dropna()
    xr, yr, zr = lla_to_map_np(mr.lat_r, mr.lon_r, mr.alt_r)
    xm, ym, zm = lla_to_map_np(mr.lat_m, mr.lon_m, mr.alt_m)
    base = np.hypot(xr - xm, yr - ym)
    ok = np.abs(base - ANTENNA_BASELINE) < 0.5
    h = np.arctan2(yr - ym, xr - xm)
    dx, _, dz = ANTENNAS['rover']
    return pd.DataFrame({'x': xr - dx * np.cos(h), 'y': yr - dx * np.sin(h), 'z': zr - dz})[ok].reset_index(drop=True)


def thin(pts, step=0.5):
    """Оставляет точки не чаще чем через step м — убирает стоянки, иначе шум GNSS удлиняет путь."""
    keep, last = [0], pts[0]
    for i in range(1, len(pts)):
        if np.hypot(*(pts[i, :2] - last[:2])) >= step:
            keep.append(i)
            last = pts[i]
    return pts[keep]


def by_arclength(pts):
    """Точки по порядку → x, y, z через каждый 1 м пути (от первой точки)."""
    pts = thin(pts)
    if len(pts) < 2:
        return None
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(pts[:, :2], axis=0).T))]
    grid = np.arange(0, s[-1], 1.0)
    return np.c_[[np.interp(grid, s, pts[:, i]) for i in range(3)]].T


def loop_piece(pieces):
    """Медиана кусков колец по номеру метра от стыка; обрезка там, где записей мало или они расходятся."""
    n = max(len(p) for p in pieces)
    out = []
    for k in range(n):
        at_k = np.array([p[k] for p in pieces if len(p) > k])
        if len(at_k) < MIN_RUNS:
            break
        med = np.median(at_k, axis=0)
        spread = np.median(np.hypot(*(at_k[:, :2] - med[:2]).T))
        if spread > MAX_SPREAD:
            break
        out.append(med)
    return np.array(out)


def smooth(v, w):
    pad = np.pad(v, w // 2, mode='edge')
    return np.convolve(pad, np.ones(w) / w, mode='valid')


def finish(df):
    """Кольца сглаживаются; s — настоящие метры вдоль линии (0 — начало pathgraph);
    heading и curvature колец — по геометрии, уклон — по всей линии."""
    loop = (df.source == 'gnss_loop').to_numpy()
    xs, ys = smooth(df.x.to_numpy(), 5), smooth(df.y.to_numpy(), 5)
    df.loc[loop, 'x'], df.loc[loop, 'y'] = xs[loop], ys[loop]
    s = np.r_[0, np.cumsum(np.hypot(np.diff(df.x), np.diff(df.y)))]
    df['s'] = s - s[np.argmax(~loop)]
    h = np.unwrap(np.arctan2(np.gradient(df.y.to_numpy()), np.gradient(df.x.to_numpy())))
    df.loc[loop, 'heading'] = np.angle(np.exp(1j * h))[loop]
    df.loc[loop, 'curvature'] = smooth(np.gradient(h, df.s), 9)[loop]
    df['grade'] = np.gradient(smooth(df.z.to_numpy(), GRADE_WINDOW), df.s)
    return df


def main():
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(OUT_DIR / 'catalog.csv', keep_default_na=False, na_values=[''])
    bags = cat[cat.duplicate_of.isna() & ~cat.short & (cat.n_master_fix > 0) & (cat.n_rover_fix > 0)].bag
    tracks = {k: load_pathgraph(k) for k in DIRECTIONS}
    lines = {k: Track(df) for k, df in tracks.items()}
    pre = {k: [] for k in DIRECTIONS}
    post = {k: [] for k in DIRECTIONS}
    for bag in bags:
        pts = base_link_points(bag)
        if pts is None or len(pts) < 100:
            continue
        found = detect_direction(lines, pts.x, pts.y, JOIN_DIST)
        if found is None:
            continue
        k, s, c = found
        on = np.abs(c) < JOIN_DIST
        L = lines[k].s[-1]
        idx = np.where(on)[0]
        first = idx[s[idx] < JOIN_S]
        last = idx[s[idx] > L - JOIN_S]
        arr = pts[['x', 'y', 'z']].to_numpy()
        if len(first):
            i = first[0]  # куски от стыка назад, в обратном порядке
            piece = by_arclength(arr[:i + 1][::-1])
            if piece is not None:
                pre[k].append(piece)
        if len(last):
            j = last[-1]
            piece = by_arclength(arr[j:])
            if piece is not None:
                post[k].append(piece)

    for k, df in tracks.items():
        df['source'] = 'pathgraph'
        L = df.s.iloc[-1]
        parts = []
        a = loop_piece(pre[k]) if pre[k] else np.empty((0, 3))
        if len(a) > 1:
            parts.append(pd.DataFrame({'s': -np.arange(len(a))[::-1] - 1.0, 'x': a[::-1, 0],
                                       'y': a[::-1, 1], 'z': a[::-1, 2], 'source': 'gnss_loop'}))
        parts.append(df)
        b = loop_piece(post[k]) if post[k] else np.empty((0, 3))
        if len(b) > 1:
            parts.append(pd.DataFrame({'s': L + np.arange(len(b)) + 1.0, 'x': b[:, 0], 'y': b[:, 1],
                                       'z': b[:, 2], 'source': 'gnss_loop'}))
        full = finish(pd.concat(parts, ignore_index=True))
        cols = ['s', 'x', 'y', 'z', 'heading', 'curvature', 'grade', 'source']
        full[cols].to_csv(MAP_DIR / f'track_{k}.csv', index=False, float_format='%.4f')
        print(f'{k}: pathgraph {L:.0f} м, кольцо до начала {len(a)} м ({len(pre[k])} записей), '
              f'после конца {len(b)} м ({len(post[k])} записей)')


if __name__ == '__main__':
    main()
