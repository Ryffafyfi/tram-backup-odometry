"""Линия пути: загрузка, проекция точки (s — метры вдоль пути, cross — отклонение вбок), точка по s."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / 'tram_backup_odometry' / 'maps'  # карта с кольцами: track_<направление>.csv (в пакете ROS 2)
PATHGRAPH_DIR = ROOT / 'pathgraph'  # исходный pathgraph организаторов (в репозиторий не входит)
# направление → (файл pathgraph, начальная конечная, конечная конечная)
DIRECTIONS = {
    'tallinskaya_shchukinskaya': ('таллинская - щукинская.json', 'Таллинская', 'Щукинская'),
    'shchukinskaya_tallinskaya': ('щукинская - таллинская.json', 'Щукинская', 'Таллинская'),
}


def load_pathgraph(direction):
    """Линия pathgraph как таблица s, x, y, z, heading, curvature (s — номер точки, шаг 1 м)."""
    name = DIRECTIONS[direction][0]
    path = PATHGRAPH_DIR / name if (PATHGRAPH_DIR / name).exists() else MAP_DIR / f'{direction}.json'
    d = json.loads(path.read_text(encoding='utf-8'))
    pts = [d['points'][i] for i in d['paths'][0]['point_indices']]
    df = pd.DataFrame({'x': [p['x'] for p in pts], 'y': [p['y'] for p in pts],
                       'z': [p['z'] for p in pts], 'heading': [p['tang'] for p in pts],
                       'curvature': [p['curv'] for p in pts]})
    df['heading'] = np.angle(np.exp(1j * df.heading))  # в диапазон (−π, π]
    step = np.hypot(np.diff(df.x), np.diff(df.y))
    df.insert(0, 's', np.r_[0, np.cumsum(step)])
    return df


def detect_direction(lines, x, y, on_dist=1.5):
    """По точкам поездки выбирает направление: линию, вдоль которой точки идут с ростом s.

    lines — {направление: Track}. Возвращает (направление, s, cross) или None.
    """
    best = None
    for k, line in lines.items():
        s, c = line.project(x, y)
        on = np.abs(c) < on_dist
        if on.sum() > 100 and np.median(np.diff(s[on])) > 0 and (best is None or on.sum() > best[0]):
            best = (on.sum(), k, s, c)
    return None if best is None else best[1:]


class Track:
    """Линия пути одного направления (из track_<направление>.csv или таблицы)."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.s = df.s.to_numpy()
        self.xy = df[['x', 'y']].to_numpy()
        self.a, self.b = self.xy[:-1], self.xy[1:]
        self._tree = cKDTree((self.a + self.b) / 2)

    @classmethod
    def load(cls, direction):
        return cls(pd.read_csv(MAP_DIR / f'track_{direction}.csv'))

    def project(self, x, y, k=8):
        """Точки → (s вдоль пути, cross — расстояние до линии со знаком: + слева)."""
        p = np.c_[np.atleast_1d(x), np.atleast_1d(y)]
        _, ii = self._tree.query(p, k=k)
        a, b = self.a[ii], self.b[ii]
        ab = b - a
        t = np.clip(((p[:, None] - a) * ab).sum(-1) / (ab ** 2).sum(-1), 0, 1)
        q = a + t[..., None] * ab
        dist = np.linalg.norm(p[:, None] - q, axis=-1)
        j = dist.argmin(1)
        rows = np.arange(len(p))
        seg = ii[rows, j]
        s = self.s[seg] + t[rows, j] * (self.s[seg + 1] - self.s[seg])
        side = np.sign(ab[rows, j, 0] * (p[:, 1] - a[rows, j, 1]) - ab[rows, j, 1] * (p[:, 0] - a[rows, j, 0]))
        return s, side * dist[rows, j]

    def at(self, s):
        """Точка на линии по s (линейная интерполяция): x, y, z, heading."""
        s = np.atleast_1d(s)
        x = np.interp(s, self.s, self.df.x)
        y = np.interp(s, self.s, self.df.y)
        z = np.interp(s, self.s, self.df.z)
        h = np.interp(s, self.s, np.unwrap(self.df.heading))
        return x, y, z, np.angle(np.exp(1j * h))
