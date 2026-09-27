"""Перевод GNSS (широта, долгота, высота) в координаты карты pathgraph и пересчёт антенны в base_link.

Карта pathgraph — это UTM зона 37N (WGS84) со сдвигом: x = E − 300 000, y = N − 6 100 000
(угол квадрата MGRS). z карты — уровень рельса, в той же системе высот, что и GNSS.
Проверено на записях 30618: расстояние от пересчитанной точки rover до линии карты — медиана 5–9 см.

Функции без numpy — их можно импортировать прямо в ноду. Для массивов — lla_to_map_np.
"""
import math

ZONE = 37
MAP_OFFSET_X = 300_000.0
MAP_OFFSET_Y = 6_100_000.0

# положение антенн в системе base_link (x — вперёд, y — влево, z — вверх), м; данные организаторов
ANTENNAS = {'rover': (2.563, 0.0, 3.0), 'master': (-9.873, 0.0, 3.0)}
ANTENNA_BASELINE = ANTENNAS['rover'][0] - ANTENNAS['master'][0]  # 12,436 м

_A, _F, _K0 = 6_378_137.0, 1 / 298.257223563, 0.9996
_N = _F / (2 - _F)
_AR = _A / (1 + _N) * (1 + _N ** 2 / 4 + _N ** 4 / 64)
_ALPHA = (_N / 2 - 2 * _N ** 2 / 3 + 5 * _N ** 3 / 16,
          13 * _N ** 2 / 48 - 3 * _N ** 3 / 5,
          61 * _N ** 3 / 240)
_C = 2 * math.sqrt(_N) / (1 + _N)
_LON0 = math.radians(ZONE * 6 - 183)


def _utm(lat, lon, m):
    """Прямая проекция UTM (ряд Крюгера); m — модуль math или numpy."""
    atanh = math.atanh if m is math else m.arctanh
    atan = math.atan if m is math else m.arctan
    phi, lam = m.radians(lat), m.radians(lon) - _LON0
    t = m.sinh(atanh(m.sin(phi)) - _C * atanh(_C * m.sin(phi)))
    xi = atan(t / m.cos(lam))
    eta = atanh(m.sin(lam) / m.sqrt(1 + t ** 2))
    e, n = eta, xi
    for j, a in enumerate(_ALPHA, 1):
        e = e + a * m.cos(2 * j * xi) * m.sinh(2 * j * eta)
        n = n + a * m.sin(2 * j * xi) * m.cosh(2 * j * eta)
    return 500_000.0 + _K0 * _AR * e, _K0 * _AR * n


def lla_to_map(lat, lon, alt):
    """Широта, долгота (градусы), высота (м) → x, y, z в координатах карты (точка антенны)."""
    e, n = _utm(lat, lon, math)
    return e - MAP_OFFSET_X, n - MAP_OFFSET_Y, alt


def lla_to_map_np(lat, lon, alt):
    """То же для массивов numpy."""
    import numpy as np
    e, n = _utm(np.asarray(lat, float), np.asarray(lon, float), np)
    return e - MAP_OFFSET_X, n - MAP_OFFSET_Y, np.asarray(alt, float)


def heading_from_antennas(master_x, master_y, rover_x, rover_y):
    """Курс вагона, рад: направление от master (сзади) к rover (спереди), от оси x против часовой."""
    return math.atan2(rover_y - master_y, rover_x - master_x)


def antenna_to_base_link(x, y, z, heading, antenna):
    """Координаты антенны в карте + курс → координаты base_link (центр передней тележки, уровень рельса)."""
    dx, dy, dz = ANTENNAS[antenna]
    c, s = math.cos(heading), math.sin(heading)
    return x - (dx * c - dy * s), y - (dx * s + dy * c), z - dz
