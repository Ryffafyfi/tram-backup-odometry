"""Карта пути для оценщика (Estimator из mathpython.py). Чистый Python без numpy и pandas — можно прямо в ноду.

Реализует интерфейсы TrackMap и GnssTransformer из mathpython.py:
  TrackMap(map_dir)                       загружает map/track_<направление>.csv (линии с разворотными кольцами)
  select_direction(x, y, heading) → str   выбирает линию по положению base_link в начале записи
  project_to_path(x, y) → s               метры вдоль выбранной линии
  get_pose_at(s) → (x, y, z, heading)     точка линии: z — уровень рельса, heading — курс пути, рад
  get_grade(s), get_curvature(s)          уклон dz/ds и кривизна 1/м — для модели
  get_special_points()                    концы линии и границы pathgraph (дальше — кольца по GNSS)
  GnssToMap().to_utm(lat, lon, alt)       GNSS → координаты карты = UTM 37 минус (300 000, 6 100 000)

Пример:
  track = TrackMap('map'); gnss = GnssToMap()
  x, y, z = gnss.to_utm(lat, lon, alt)                  # точка антенны в координатах карты
  direction = track.select_direction(x_bl, y_bl, heading)
  s = track.project_to_path(x_bl, y_bl)
  x, y, z, heading = track.get_pose_at(s + пройденный_путь)
"""
import bisect
import csv
import math
from pathlib import Path

try:
    from .geo import lla_to_map
except ImportError:
    from geo import lla_to_map

DIRECTIONS = ('tallinskaya_shchukinskaya', 'shchukinskaya_tallinskaya')
NEAR_M = 3.0         # м: точка ближе этого к линии считается лежащей на ней
START_M = 300.0      # м: «у начала линии» — кольцо до pathgraph и первые метры после его начала
START_NEAR_M = 15.0  # м: на конечной трамвай может стоять на соседнем пути кольца


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class _Line:
    """Одна линия: точки через ~1 м, s — метры вдоль пути (0 — начало pathgraph, s < 0 — кольцо до него)."""

    def __init__(self, path):
        with open(path, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        self.s = [float(r['s']) for r in rows]
        self.x = [float(r['x']) for r in rows]
        self.y = [float(r['y']) for r in rows]
        self.z = [float(r['z']) for r in rows]
        self.grade = [float(r['grade']) for r in rows]
        self.curv = [float(r['curvature']) for r in rows]
        self.heading = [float(rows[0]['heading'])]   # без скачков через ±π — чтобы интерполировать
        for r in rows[1:]:
            h = float(r['heading'])
            self.heading.append(self.heading[-1] + _wrap(h - self.heading[-1]))
        pg = [float(r['s']) for r in rows if r['source'] == 'pathgraph']
        self.pathgraph = (pg[0], pg[-1])

    def _interp(self, values, s):
        i = bisect.bisect_right(self.s, s) - 1
        i = min(max(i, 0), len(self.s) - 2)
        k = (s - self.s[i]) / (self.s[i + 1] - self.s[i])
        k = min(max(k, 0.0), 1.0)   # за концами линии — крайняя точка
        return values[i] + k * (values[i + 1] - values[i])

    def at(self, s):
        return (self._interp(self.x, s), self._interp(self.y, s), self._interp(self.z, s),
                _wrap(self._interp(self.heading, s)))

    def project(self, x, y, s_hint=None, window=100.0):
        """Ближайшая точка линии → (s, расстояние). s_hint — искать только в ±window м от него."""
        lo, hi = 0, len(self.s) - 1
        if s_hint is not None:
            lo = max(bisect.bisect_left(self.s, s_hint - window) - 1, 0)
            hi = min(bisect.bisect_right(self.s, s_hint + window), len(self.s) - 1)
        best = (float('inf'), 0.0)
        for i in range(lo, hi):
            ax, ay = self.x[i], self.y[i]
            dx, dy = self.x[i + 1] - ax, self.y[i + 1] - ay
            k = ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)
            k = min(max(k, 0.0), 1.0)
            d = math.hypot(x - ax - k * dx, y - ay - k * dy)
            if d < best[0]:
                best = (d, self.s[i] + k * (self.s[i + 1] - self.s[i]))
        return best[1], best[0]


class TrackMap:
    def __init__(self, map_dir='map'):
        self.lines = {k: _Line(Path(map_dir) / f'track_{k}.csv') for k in DIRECTIONS}
        self.direction = None

    @property
    def line(self):
        return self.lines[self.direction or DIRECTIONS[0]]

    def select_direction(self, x, y, heading=None):
        """Линия, по которой поедет трамвай. heading — курс вагона, рад (можно None).

        Записи начинаются со стоянки на начальной конечной. Кольцо конечной входит в обе линии:
        в одну как начало, в другую как конец. Поэтому сначала ищется линия, у начала которой
        стоит трамвай: до START_M после начала pathgraph, не дальше START_NEAR_M от линии
        (на конечной он может стоять на соседнем пути кольца). Если такой нет — линия, на которой
        лежит точка (ближе NEAR_M), иначе ближайшая. Курс, если задан, должен совпадать с курсом
        линии (отсекает встречный путь)."""
        cands = []
        for k, line in self.lines.items():
            s, d = line.project(x, y)
            same_way = heading is None or math.cos(heading - line.at(s)[3]) > 0
            cands.append((k, s, d, same_way))
        start = [(d, k) for k, s, d, ok in cands
                 if ok and d < START_NEAR_M and s < self.lines[k].pathgraph[0] + START_M]
        on = [(s, k) for k, s, d, ok in cands if ok and d < NEAR_M]
        if start:
            self.direction = min(start)[1]
        elif on:
            self.direction = min(on)[1]
        else:
            self.direction = min(cands, key=lambda c: c[2])[0]
        return self.direction

    def project_to_path(self, x, y, s_hint=None):
        """s точки на выбранной линии, м. s_hint (текущая оценка s) ускоряет поиск и не даёт
        перескочить на соседнюю ветку кольца."""
        return self.line.project(x, y, s_hint)[0]

    def distance_to_path(self, x, y, s_hint=None):
        return self.line.project(x, y, s_hint)[1]

    def get_pose_at(self, s):
        return self.line.at(s)

    def get_grade(self, s):
        return self.line._interp(self.line.grade, s)

    def get_curvature(self, s):
        return self.line._interp(self.line.curv, s)

    def get_special_points(self):
        line = self.line
        return [{'s': line.s[0], 'name': 'начало линии (кольцо по GNSS)'},
                {'s': line.pathgraph[0], 'name': 'начало pathgraph'},
                {'s': line.pathgraph[1], 'name': 'конец pathgraph'},
                {'s': line.s[-1], 'name': 'конец линии (кольцо по GNSS)'}]


class GnssToMap:
    """GnssTransformer для Estimator. Возвращает координаты карты и эталона (UTM зона 37 минус
    (300 000, 6 100 000)), а не «сырые» UTM — иначе ответ съедет на сотни километров."""

    def to_utm(self, lat, lon, alt):
        return lla_to_map(lat, lon, alt)
