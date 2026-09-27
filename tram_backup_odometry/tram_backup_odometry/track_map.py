"""Карта путей (формат pathgraph) и операции над ней.

Формат файла (как в выданных «щукинская - таллинская.json» / «таллинская - щукинская.json»):

    {"points": [{"x":..,"y":..,"z":..,"tang":..,"curv":..}, ...],
     "paths":  [{"ext_id": null, "point_indices": [0, 1, 2, ...]}, ...]}

Каждый path — направленная ломаная (направление движения трамвая). Файлов и путей
может быть сколько угодно (например, отдельный файл с разворотными петлями). Концы путей
автоматически сшиваются в граф: конец A -> начало B, если точки ближе ``connect_tol`` и
направления совпадают. Позиция на карте задаётся парой (индекс пути, дуга s в метрах).

Только стандартная библиотека Python — никаких внешних зависимостей.
"""

import bisect
import json
import math
import os


def _wrap(a):
    """Нормализация угла в (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class TrackPath:
    """Одна направленная ломаная с натуральной параметризацией по длине дуги (в плоскости XY)."""

    def __init__(self, name, xs, ys, zs):
        if len(xs) < 2:
            raise ValueError('path %s has < 2 points' % name)
        # выбросить повторяющиеся точки (нулевые сегменты)
        px, py, pz = [float(xs[0])], [float(ys[0])], [float(zs[0])]
        for x, y, z in zip(xs[1:], ys[1:], zs[1:]):
            if math.hypot(x - px[-1], y - py[-1]) > 1e-6:
                px.append(float(x))
                py.append(float(y))
                pz.append(float(z))
        if len(px) < 2:
            raise ValueError('path %s is degenerate' % name)
        self.name = name
        self.xs, self.ys, self.zs = px, py, pz
        self.s = [0.0]
        self.seg_yaw = []
        self.seg_pitch = []
        for i in range(len(px) - 1):
            dx = px[i + 1] - px[i]
            dy = py[i + 1] - py[i]
            dz = pz[i + 1] - pz[i]
            d = math.hypot(dx, dy)
            self.s.append(self.s[-1] + d)
            self.seg_yaw.append(math.atan2(dy, dx))
            self.seg_pitch.append(math.atan2(dz, d))
        self.length = self.s[-1]
        self.n_seg = len(px) - 1

    # --- доступ по дуге -------------------------------------------------------------
    def seg_index(self, s):
        if s <= 0.0:
            return 0
        if s >= self.length:
            return self.n_seg - 1
        i = bisect.bisect_right(self.s, s) - 1
        return min(max(i, 0), self.n_seg - 1)

    def pose(self, s):
        """(x, y, z, yaw, pitch) в точке дуги s (s за пределами пути — линейная экстраполяция)."""
        i = self.seg_index(s)
        s0 = self.s[i]
        L = self.s[i + 1] - s0
        u = (s - s0) / L if L > 0 else 0.0
        x = self.xs[i] + (self.xs[i + 1] - self.xs[i]) * u
        y = self.ys[i] + (self.ys[i + 1] - self.ys[i]) * u
        z = self.zs[i] + (self.zs[i + 1] - self.zs[i]) * u
        # курс: плавная интерполяция между серединами соседних сегментов
        yaw = self.seg_yaw[i]
        if 0.0 <= s <= self.length and self.n_seg > 1:
            if u < 0.5 and i > 0:
                w = 0.5 + u
                yaw = self.seg_yaw[i - 1] + _wrap(self.seg_yaw[i] - self.seg_yaw[i - 1]) * w
            elif u >= 0.5 and i < self.n_seg - 1:
                w = u - 0.5
                yaw = self.seg_yaw[i] + _wrap(self.seg_yaw[i + 1] - self.seg_yaw[i]) * w
        return x, y, z, _wrap(yaw), self.seg_pitch[i]

    def project_on_segment(self, i, x, y):
        """Проекция точки на сегмент i: (s, квадрат расстояния, знак бокового смещения)."""
        x0, y0 = self.xs[i], self.ys[i]
        dx = self.xs[i + 1] - x0
        dy = self.ys[i + 1] - y0
        L2 = dx * dx + dy * dy
        u = ((x - x0) * dx + (y - y0) * dy) / L2 if L2 > 0 else 0.0
        u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
        px = x0 + dx * u
        py = y0 + dy * u
        d2 = (x - px) ** 2 + (y - py) ** 2
        side = dx * (y - y0) - dy * (x - x0)  # >0 — точка слева от направления пути
        return self.s[i] + math.sqrt(L2) * u, d2, side


class TrackMap:
    """Набор путей + граф переходов + пространственный индекс для быстрой проекции."""

    def __init__(self, files=(), connect_tol=6.0, connect_heading_tol_deg=60.0, cell=25.0):
        self.paths = []
        self.successors = {}
        self.predecessors = {}
        self.cell = float(cell)
        self._grid = {}
        self.connect_tol = float(connect_tol)
        self.connect_heading_tol = math.radians(connect_heading_tol_deg)
        self.loaded_files = []
        for f in files:
            self.load_file(f)
        self.finalize()

    # --- загрузка ---------------------------------------------------------------------
    def load_file(self, path):
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        pts = data.get('points', [])
        paths = data.get('paths') or [{'point_indices': list(range(len(pts)))}]
        stem = os.path.splitext(os.path.basename(path))[0]
        for k, p in enumerate(paths):
            idx = p.get('point_indices') or []
            xs, ys, zs = [], [], []
            for j in idx:
                q = pts[int(j)]
                x, y = float(q['x']), float(q['y'])
                z = float(q.get('z', 0.0) or 0.0)
                if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
            name = p.get('ext_id') or (stem if len(paths) == 1 else '%s#%d' % (stem, k))
            tp = TrackPath(str(name), xs, ys, zs)
            tp.source = stem          # файл карты, из которого путь (для оценки точности геометрии)
            self.paths.append(tp)
        self.loaded_files.append(path)

    def finalize(self):
        self._grid = {}
        c = self.cell
        for pid, p in enumerate(self.paths):
            for i in range(p.n_seg):
                x0, x1 = sorted((p.xs[i], p.xs[i + 1]))
                y0, y1 = sorted((p.ys[i], p.ys[i + 1]))
                for gx in range(int(math.floor(x0 / c)), int(math.floor(x1 / c)) + 1):
                    for gy in range(int(math.floor(y0 / c)), int(math.floor(y1 / c)) + 1):
                        self._grid.setdefault((gx, gy), []).append((pid, i))
        # граф: конец A -> начало B
        self.successors = {pid: [] for pid in range(len(self.paths))}
        for a, pa in enumerate(self.paths):
            ex, ey = pa.xs[-1], pa.ys[-1]
            eyaw = pa.seg_yaw[-1]
            cands = []
            for b, pb in enumerate(self.paths):
                if a == b:
                    continue
                d = math.hypot(pb.xs[0] - ex, pb.ys[0] - ey)
                dh = abs(_wrap(pb.seg_yaw[0] - eyaw))
                if d <= self.connect_tol and dh <= self.connect_heading_tol:
                    cands.append((d, b))
            cands.sort()
            self.successors[a] = [b for _, b in cands]
        self.predecessors = {pid: [] for pid in range(len(self.paths))}
        for a, lst in self.successors.items():
            for b in lst:
                self.predecessors[b].append(a)

    # --- запросы ------------------------------------------------------------------------
    @property
    def empty(self):
        return not self.paths

    def pose(self, pid, s):
        return self.paths[pid].pose(s)

    def length(self, pid):
        return self.paths[pid].length

    def advance(self, pid, s, ds, max_hops=8):
        """Сдвинуться по графу на ds метров (ds < 0 — назад по пути).

        Возвращает (pid, s, ушли_за_край_графа). За краем — линейная экстраполяция.
        """
        s = s + ds
        hops = 0
        while s > self.paths[pid].length and hops < max_hops:
            nxt = self.successors.get(pid) or []
            if not nxt:
                return pid, s, True
            s -= self.paths[pid].length
            pid = nxt[0]
            hops += 1
        while s < 0.0 and hops < max_hops:
            prv = self.predecessors.get(pid) or []
            if not prv:
                return pid, s, True
            pid = prv[0]
            s += self.paths[pid].length
            hops += 1
        return pid, s, False

    def project(self, x, y, max_dist=1e9, per_path=True):
        """Ближайшие точки путей к (x, y).

        Возвращает список кандидатов (dist, pid, s, yaw_пути, side), отсортированный по dist;
        при per_path=True — не более одного кандидата на путь.
        """
        if not self.paths:
            return []
        c = self.cell
        gx0 = int(math.floor(x / c))
        gy0 = int(math.floor(y / c))
        best = {}
        seen = set()
        r = 1
        max_r = int(max_dist / c) + 2 if max_dist < 1e8 else 10 ** 6
        while True:
            for gx in range(gx0 - r, gx0 + r + 1):
                for gy in range(gy0 - r, gy0 + r + 1):
                    if (gx, gy) in seen:
                        continue
                    seen.add((gx, gy))
                    for pid, i in self._grid.get((gx, gy), ()):
                        s, d2, side = self.paths[pid].project_on_segment(i, x, y)
                        key = pid if per_path else (pid, i)
                        if key not in best or d2 < best[key][0]:
                            best[key] = (d2, pid, s, i, side)
            if best:
                dmin = math.sqrt(min(v[0] for v in best.values()))
                if dmin <= (r - 0.5) * c:
                    break
            if r >= max_r or len(seen) > 200000:
                break
            r = r * 2
        out = []
        for d2, pid, s, i, side in best.values():
            d = math.sqrt(d2)
            if d <= max_dist:
                out.append((d, pid, s, self.paths[pid].seg_yaw[i], side))
        out.sort(key=lambda v: v[0])
        return out
