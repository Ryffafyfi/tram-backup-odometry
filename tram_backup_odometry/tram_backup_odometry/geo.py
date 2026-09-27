"""Перевод GNSS (WGS-84 широта/долгота/высота) в метрическую систему карты.

Система карты (frame ``map``) проверена по эталону ``/localization/kinematic_state``
проверочной записи: это UTM зона 37N (поперечная Меркатора, центральный меридиан 39°,
масштаб 0.9996, ложное восточное смещение 500 000 м) со сдвигом начала координат:

    x_map = E_utm - 300 000
    y_map = N_utm - 6 100 000
    z_map = altitude (высота из NavSatFix, та же, что поле z в pathgraph)

Точность совпадения с эталоном по антенне master ~0.1-0.2 м.
Все параметры задаются в YAML (см. config/tram_backup_odometry.yaml).

Реализация — ряды Крюгера до n^6 (погрешность << 1 мм в пределах зоны),
без внешних зависимостей (pyproj не нужен — сборка без интернета).
"""

import math

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563


class TransverseMercator:
    """Прямая проекция WGS-84 -> поперечная Меркатора (+ сдвиг в систему карты)."""

    def __init__(self, lon0_deg=39.0, k0=0.9996, false_easting=500000.0,
                 false_northing=0.0, map_offset_x=-300000.0,
                 map_offset_y=-6100000.0, map_offset_z=0.0,
                 a=WGS84_A, f=WGS84_F):
        self.lon0 = math.radians(float(lon0_deg))
        self.k0 = float(k0)
        self.fe = float(false_easting)
        self.fn = float(false_northing)
        self.ox = float(map_offset_x)
        self.oy = float(map_offset_y)
        self.oz = float(map_offset_z)
        n = f / (2.0 - f)
        self.e = math.sqrt(f * (2.0 - f))
        self.A1 = a / (1.0 + n) * (1.0 + n ** 2 / 4.0 + n ** 4 / 64.0 + n ** 6 / 256.0)
        self.alpha = (
            0.0,
            n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16 + 41 * n ** 4 / 180
            - 127 * n ** 5 / 288 + 7891 * n ** 6 / 37800,
            13 * n ** 2 / 48 - 3 * n ** 3 / 5 + 557 * n ** 4 / 1440
            + 281 * n ** 5 / 630 - 1983433 * n ** 6 / 1935360,
            61 * n ** 3 / 240 - 103 * n ** 4 / 140 + 15061 * n ** 5 / 26880
            + 167603 * n ** 6 / 181440,
            49561 * n ** 4 / 161280 - 179 * n ** 5 / 168 + 6601661 * n ** 6 / 7257600,
            34729 * n ** 5 / 80640 - 3418889 * n ** 6 / 1995840,
            212378941 * n ** 6 / 319334400,
        )

    def forward(self, lat_deg, lon_deg):
        """Широта/долгота (градусы) -> (E, N) в метрах проекции."""
        phi = math.radians(lat_deg)
        lam = math.radians(lon_deg) - self.lon0
        e = self.e
        t = math.sinh(math.atanh(math.sin(phi)) - e * math.atanh(e * math.sin(phi)))
        xi_p = math.atan2(t, math.cos(lam))
        eta_p = math.atanh(math.sin(lam) / math.sqrt(1.0 + t * t))
        xi = xi_p
        eta = eta_p
        for j in range(1, 7):
            a_j = self.alpha[j]
            xi += a_j * math.sin(2 * j * xi_p) * math.cosh(2 * j * eta_p)
            eta += a_j * math.cos(2 * j * xi_p) * math.sinh(2 * j * eta_p)
        return (self.fe + self.k0 * self.A1 * eta,
                self.fn + self.k0 * self.A1 * xi)

    def to_map(self, lat_deg, lon_deg, alt=0.0):
        """Широта/долгота/высота -> (x, y, z) в системе карты. None, если вход плохой."""
        try:
            lat = float(lat_deg)
            lon = float(lon_deg)
            alt = float(alt)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return None
        if abs(lat) > 89.0 or abs(lon) > 180.0 or (lat == 0.0 and lon == 0.0):
            return None
        if not math.isfinite(alt):
            alt = 0.0
        e, n = self.forward(lat, lon)
        return (e + self.ox, n + self.oy, alt + self.oz)


def make_projection(cfg):
    """Создать проекцию из словаря параметров (ключи как в YAML ноды)."""
    return TransverseMercator(
        lon0_deg=cfg.get('gnss_tm_lon0_deg', 39.0),
        k0=cfg.get('gnss_tm_k0', 0.9996),
        false_easting=cfg.get('gnss_tm_false_easting', 500000.0),
        false_northing=cfg.get('gnss_tm_false_northing', 0.0),
        map_offset_x=cfg.get('gnss_map_offset_x', -300000.0),
        map_offset_y=cfg.get('gnss_map_offset_y', -6100000.0),
        map_offset_z=cfg.get('gnss_map_offset_z', 0.0),
    )
