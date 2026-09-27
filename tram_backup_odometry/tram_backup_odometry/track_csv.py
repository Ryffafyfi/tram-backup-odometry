"""Карта пути в формате track_*.csv (maps/track_<направление>.csv) — без pandas и scipy.

Повторяет интерфейс класса ``Track`` из ``tools/track.py`` (``s``, ``xy``, ``df``,
``project(x, y)``, ``at(s)``), но работает только на numpy — в окружении жюри (ros-base) есть
numpy, а pandas и scipy нет. Используется альтернативным оценщиком через переходник
(``agreed_api.py``): он получает ``track_map`` — словарь ``{направление: TrackLine}``.

Столбцы CSV: s, x, y, z, heading, curvature, grade, source (s — метры вдоль пути, петли у конечных
достроены: s < 0 до начала линии pathgraph и s > 4708 после конца).
"""

import csv
import json
import math
import os

DIRECTION_KEYS = ('tallinskaya_shchukinskaya', 'shchukinskaya_tallinskaya')


class _Columns:
    """Мини-таблица: столбцы как numpy-массивы, доступ ``df.x`` и ``df['x']``."""

    def __init__(self, cols):
        self._cols = dict(cols)
        self.columns = list(self._cols)

    def __getitem__(self, key):
        return self._cols[key]

    def __getattr__(self, key):
        try:
            return self.__dict__['_cols'][key]
        except KeyError:
            raise AttributeError(key) from None

    def __contains__(self, key):
        return key in self._cols

    def __len__(self):
        return len(self._cols[self.columns[0]]) if self.columns else 0


class TrackLine:
    """Линия пути одного направления: точка -> (s, отклонение вбок), s -> (x, y, z, курс)."""

    def __init__(self, cols, name=''):
        import numpy as np
        self.name = name
        num = {}
        for k, v in cols.items():
            try:
                num[k] = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                num[k] = list(v)  # текстовые столбцы (source)
        self.df = _Columns(num)
        self.s = num['s']
        self.xy = np.c_[num['x'], num['y']]
        self.a, self.b = self.xy[:-1], self.xy[1:]
        self._ab = self.b - self.a
        self._ab2 = np.maximum((self._ab ** 2).sum(-1), 1e-12)
        self._heading_unwrapped = np.unwrap(num['heading']) if 'heading' in num else None

    @classmethod
    def from_csv(cls, path, name=None):
        with open(path, newline='', encoding='utf-8-sig') as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError('empty track file: %s' % path)
        cols = {k: [r[k] for r in rows] for k in rows[0]}
        if name is None:
            base = os.path.splitext(os.path.basename(path))[0]
            name = base[len('track_'):] if base.startswith('track_') else base
        return cls(cols, name)

    def project(self, x, y, chunk=256):
        """Точки -> (s вдоль пути, cross — расстояние до линии со знаком: + слева). Массивы."""
        import numpy as np
        p = np.c_[np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))]
        s_out = np.empty(len(p))
        c_out = np.empty(len(p))
        for i0 in range(0, len(p), chunk):
            q = p[i0:i0 + chunk]
            d = q[:, None, :] - self.a[None, :, :]
            t = np.clip((d * self._ab[None]).sum(-1) / self._ab2[None], 0.0, 1.0)
            proj = self.a[None] + t[..., None] * self._ab[None]
            dist = np.linalg.norm(q[:, None, :] - proj, axis=-1)
            j = dist.argmin(1)
            rows = np.arange(len(q))
            tj = t[rows, j]
            s_out[i0:i0 + len(q)] = self.s[j] + tj * (self.s[j + 1] - self.s[j])
            ab = self._ab[j]
            side = np.sign(ab[:, 0] * (q[:, 1] - self.a[j, 1]) - ab[:, 1] * (q[:, 0] - self.a[j, 0]))
            c_out[i0:i0 + len(q)] = side * dist[rows, j]
        return s_out, c_out

    def at(self, s):
        """Точка на линии по s (линейная интерполяция): массивы x, y, z, heading."""
        import numpy as np
        s = np.atleast_1d(np.asarray(s, dtype=float))
        x = np.interp(s, self.s, self.df.x)
        y = np.interp(s, self.s, self.df.y)
        z = np.interp(s, self.s, self.df.z) if 'z' in self.df else np.zeros_like(s)
        if self._heading_unwrapped is not None:
            h = np.interp(s, self.s, self._heading_unwrapped)
            h = np.angle(np.exp(1j * h))
        else:
            h = np.zeros_like(s)
        return x, y, z, h

    def value_at(self, column, s):
        """Любой числовой столбец (grade, curvature, ...) по s."""
        import numpy as np
        return np.interp(np.atleast_1d(np.asarray(s, dtype=float)), self.s, self.df[column])

    @property
    def s_min(self):
        return float(self.s[0])

    @property
    def s_max(self):
        return float(self.s[-1])


class TrackMapCSV(dict):
    """``{направление: TrackLine}`` + несколько удобств.

    * ``detect_direction(x, y)`` — как ``tools/track.detect_direction``;
    * интерфейс «одной текущей линии» (так карту использует альтернативный оценщик, mathpython.py):
      ``select_direction(x, y, heading)`` -> направление (запоминается как текущее),
      ``project_to_path(x, y)`` -> s, ``get_pose_at(s)`` -> (x, y, z, курс), ``get_special_points()``;
    * ``graph`` — граф путей ноды (pathgraph + петли, ``track_map.TrackMap``), грузится по запросу;
    * ``files`` — пути к CSV, ``map_files`` — пути к JSON-картам ноды.
    """

    def __init__(self, csv_files, map_files=None, landmarks_file=None):
        super().__init__()
        self.files = list(csv_files or [])
        self.map_files = list(map_files or [])
        self._graph = None
        self.current = None
        for f in self.files:
            line = TrackLine.from_csv(f)
            self[line.name] = line
        self.special_points = self._load_landmarks(landmarks_file)

    def _load_landmarks(self, path):
        """Места остановок на основных линиях (s — как в track_*.csv)."""
        if not path or not os.path.isfile(path):
            return []
        try:
            with open(path, encoding='utf-8') as fh:
                items = json.load(fh)
        except (OSError, ValueError):
            return []
        out = []
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and it.get('track') in self:
                out.append({'type': 'stop', 'direction': it['track'], 's': float(it['s']),
                            'n': int(it.get('n', 0)), 'std': float(it.get('std', 0.0))})
        return out

    # ---------------------------------------------------------------- «одна текущая линия»
    def select_direction(self, x, y, heading=None):
        """Линия, ближайшая к точке и совпадающая по направлению с курсом (если он дан)."""
        best = None
        for k, line in self.items():
            s, c = line.project(x, y)
            if heading is not None:
                h = float(line.at(s[0])[3][0])
                if abs(math.remainder(h - float(heading), 2 * math.pi)) > math.pi / 2:
                    continue
            if best is None or abs(c[0]) < best[1]:
                best = (k, abs(float(c[0])))
        self.current = best[0] if best is not None else self.nearest(x, y)[0]
        return self.current

    def _line(self, x=None, y=None):
        if self.current is None:
            self.current = self.nearest(x, y)[0] if x is not None else next(iter(self))
        return self[self.current]

    def project_to_path(self, x, y):
        s, _ = self._line(x, y).project(x, y)
        return float(s[0])

    def get_pose_at(self, s):
        x, y, z, h = self._line().at(s)
        return float(x[0]), float(y[0]), float(z[0]), float(h[0])

    def get_special_points(self):
        return list(self.special_points)

    def bounds(self):
        """(xmin, xmax, ymin, ymax, zmin, zmax) по всем линиям."""
        xs = [(float(v.df.x.min()), float(v.df.x.max())) for v in self.values()]
        ys = [(float(v.df.y.min()), float(v.df.y.max())) for v in self.values()]
        zs = [(float(v.df.z.min()), float(v.df.z.max())) for v in self.values() if 'z' in v.df]
        return (min(a for a, _ in xs), max(b for _, b in xs), min(a for a, _ in ys),
                max(b for _, b in ys), min(a for a, _ in zs) if zs else math.nan,
                max(b for _, b in zs) if zs else math.nan)

    @property
    def directions(self):
        return list(self.keys())

    @property
    def graph(self):
        if self._graph is None and self.map_files:
            from .track_map import TrackMap
            self._graph = TrackMap(self.map_files)
        return self._graph

    def detect_direction(self, x, y, on_dist=1.5, min_points=3):
        """По точкам поездки выбрать линию, вдоль которой они идут с ростом s.

        Возвращает (направление, s, cross) или None.
        """
        import numpy as np
        best = None
        for k, line in self.items():
            s, c = line.project(x, y)
            on = np.abs(c) < on_dist
            if on.sum() >= min_points:
                ds = np.diff(s[on])
                if len(ds) and np.median(ds) > 0 and (best is None or on.sum() > best[0]):
                    best = (int(on.sum()), k, s, c)
        return None if best is None else best[1:]

    def nearest(self, x, y):
        """Ближайшая линия к точке: (направление, s, cross) — без учёта направления движения."""
        best = None
        for k, line in self.items():
            s, c = line.project(x, y)
            if best is None or abs(c[0]) < abs(best[2]):
                best = (k, float(s[0]), float(c[0]))
        return best


def load_track_map(csv_files, map_files=None, log=None, landmarks_file=None):
    """TrackMapCSV или None (с сообщением в лог), если файлов нет или numpy недоступен."""
    log = log or (lambda level, msg: None)
    files = [f for f in (csv_files or []) if os.path.isfile(f)]
    if not files:
        log('warn', 'track csv files not found -> track_map for team estimator is None')
        return None
    try:
        tm = TrackMapCSV(files, map_files, landmarks_file)
    except Exception as e:  # noqa: BLE001
        log('error', 'cannot load track csv files %s: %r -> track_map is None' % (files, e))
        return None
    lengths = ', '.join('%s %.0f..%.0f m' % (k, v.s_min, v.s_max) for k, v in tm.items())
    log('info', 'track_map for team estimator: %s' % lengths)
    return tm
