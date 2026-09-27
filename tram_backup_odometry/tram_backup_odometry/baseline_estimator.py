"""Оценщик скорости и положения трамвая: две тележки + модель по позиции контроллера + карта путей.

Основной оценщик решения (estimator.py ссылается на этот класс). Нода держит второй экземпляр в
«теневом» режиме как страховку, если подключён другой оценщик и он бросит исключение или вернёт NaN.
Модель описана в docs/model.md.

Скорость (всё в м/с, v_тележки = raw * velocity_scale; в записях raw — км/ч):
  * прогноз v_pred = v + a*dt, где a — сглаженное измеренное ускорение, а без колёсных
    данных — табличная модель тяги/торможения a(позиция контроллера, v) (tools/fit_traction_table.py);
  * показание тележки принимается, если попадает в строб |v_i - v_pred| <= gate;
    обе в стробе — среднее; одна — берём её (вторая: боксование/юз/отказ датчика);
  * обе вне строба — проскальзывание обеих или резкий манёвр: шаг к ближайшей не больше
    физически возможного ускорения; если тележки согласны между собой дольше resync_sec —
    доверяем им (защита от «залипания» оценки);
  * нет свежих колёсных данных — движение по модели тяги (прогноз в пропусках).
Положение:
  * GNSS-старт: base_link из антенн master/rover (смещения из YAML, веса по status: RTK=2
    лучше всего), привязка к ближайшему пути карты с учётом направления кузова; медиана
    привязок по одометру (так работает и если трамвай тронулся в окне);
  * далее s += v*dt вдоль пути с переходами по графу карты (разворотные петли включены);
  * курс — хорда между задней (s - 7.55 м) и передней (s) тележками, как у кузова;
  * вне карты — плоское счисление с курсом по антеннам.
"""

import bisect
import json
import math

from .estimator_api import EstimatorState
from .geo import make_projection
from .track_map import TrackMap, _wrap

G = 9.80665

# Модель a(позиция, v) по обучающим записям 30618 (tools/fit_traction_table.py), м/с^2
DEFAULT_V_BINS = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 30.0]
DEFAULT_TABLE = [
    [-0.293, 0.048, -0.154, -1.467, -1.092, -1.672, -1.672, -1.672, -1.672],
    [-0.647, -1.081, -1.571, -1.663, -1.526, -1.526, -1.526, -1.526, -1.526],
    [-0.896, -1.173, -1.401, -1.423, -1.364, -1.292, -1.259, -1.259, -1.259],
    [-0.92, -1.102, -1.192, -1.415, -1.318, -1.225, -1.259, -1.259, -1.259],
    [-0.715, -1.042, -1.169, -1.344, -1.286, -1.165, -1.144, -1.144, -1.144],
    [-0.835, -0.988, -1.074, -1.239, -1.228, -1.097, -1.091, -1.091, -1.091],
    [-0.052, -0.934, -0.911, -1.014, -1.008, -1.011, -0.961, -0.961, -0.961],
    [-0.01, -0.767, -0.131, -0.063, -0.329, -0.118, -0.069, -0.029, -0.029],
    [0.0, -0.774, -0.754, -0.792, -0.807, -0.777, -0.757, -0.775, -0.775],
    [0.0, -0.714, -0.687, -0.723, -0.753, -0.738, -0.684, -0.727, -0.727],
    [0.0, -0.672, -0.661, -0.684, -0.704, -0.689, -0.677, -0.67, -0.67],
    [0.0, -0.606, -0.617, -0.664, -0.679, -0.674, -0.634, -0.583, -0.583],
    [0.0, -0.546, -0.569, -0.628, -0.626, -0.63, -0.568, -0.445, -0.445],
    [0.0, -0.394, -0.353, -0.15, -0.321, -0.379, -0.432, -0.294, -0.294],
    [0.0, -0.204, -0.181, -0.029, -0.152, -0.07, -0.287, -0.167, -0.167],
    [0.0, -0.066, -0.034, 0.038, -0.018, 0.026, 0.031, -0.013, -0.013],
    [0.0, 0.019, 0.041, 0.109, 0.056, 0.046, 0.064, 0.052, 0.052],
    [0.019, 0.1, 0.15, 0.232, 0.12, 0.086, 0.161, 0.087, 0.087],
    [0.279, 0.27, 0.272, 0.287, 0.153, 0.187, 0.205, 0.152, 0.152],
    [0.424, 0.44, 0.389, 0.38, 0.305, 0.029, 0.23, 0.144, 0.144],
    [0.562, 0.586, 0.481, 0.465, 0.393, 0.27, 0.24, 0.233, 0.233],
    [0.662, 0.711, 0.543, 0.545, 0.472, 0.413, 0.264, 0.278, 0.278],
    [0.71, 0.764, 0.63, 0.643, 0.53, 0.448, 0.324, 0.319, 0.319],
    [0.765, 0.888, 0.841, 0.783, 0.598, 0.471, 0.371, 0.344, 0.344],
    [0.732, 0.922, 0.961, 0.901, 0.632, 0.506, 0.42, 0.364, 0.364],
    [0.719, 0.901, 0.955, 0.896, 0.8, 0.625, 0.468, 0.425, 0.425],
    [0.864, 0.864, 0.929, 0.878, 0.823, 0.671, 0.519, 0.52, 0.52],
    [0.803, 0.803, 0.909, 0.86, 0.828, 0.674, 0.592, 0.548, 0.548],
    [0.795, 0.795, 0.909, 0.835, 0.819, 0.746, 0.624, 0.624, 0.624],
    [1.052, 1.052, 0.888, 0.818, 0.802, 0.761, 0.613, 0.613, 0.613],
    [1.081, 1.081, 0.875, 0.764, 0.743, 0.657, 0.594, 0.594, 0.594],
]


# То же с учётом уклона (tools/fit_traction_grade.py): a = A_flat(u, v) - g*sin(уклон).
# Остаточная ошибка 0.202 против 0.221 м/с^2 без уклона (59 записей с RTK).
DEFAULT_TABLE_FLAT = [
    [-0.354, 0.354, -1.722, -1.759, -0.907, -0.907, -0.907, -0.907, -0.907],
    [-0.692, -0.75, -1.56, -1.683, -1.615, -1.615, -1.615, -1.615, -1.615],
    [-0.908, -1.216, -1.361, -1.562, -1.45, -1.374, -1.374, -1.374, -1.374],
    [-0.907, -1.165, -1.301, -1.478, -1.425, -1.296, -1.296, -1.296, -1.296],
    [-0.607, -1.11, -1.281, -1.309, -1.284, -1.232, -1.144, -1.144, -1.144],
    [-0.783, -1.048, -1.161, -1.245, -1.189, -1.143, -1.082, -1.082, -1.082],
    [-0.141, -1.028, -1.004, -1.144, -1.102, -1.102, -1.021, -1.021, -1.021],
    [-0.108, -0.534, -0.08, -0.05, -0.074, -0.021, -0.142, -0.142, -0.142],
    [-0.063, -0.874, -0.875, -0.935, -0.947, -0.923, -0.875, -0.805, -0.805],
    [-0.062, -0.763, -0.78, -0.856, -0.865, -0.838, -0.799, -0.773, -0.773],
    [-0.062, -0.665, -0.679, -0.742, -0.763, -0.733, -0.717, -0.688, -0.688],
    [-0.036, -0.544, -0.568, -0.631, -0.637, -0.642, -0.624, -0.62, -0.62],
    [-0.045, -0.467, -0.474, -0.534, -0.552, -0.533, -0.507, -0.468, -0.468],
    [-0.026, -0.402, -0.393, -0.444, -0.44, -0.402, -0.363, -0.293, -0.293],
    [-0.021, -0.276, -0.23, -0.375, -0.247, -0.31, -0.217, -0.226, -0.226],
    [-0.01, -0.06, -0.047, -0.061, -0.054, -0.028, -0.034, -0.041, -0.041],
    [0.005, 0.027, 0.071, 0.102, 0.106, 0.138, 0.091, 0.095, 0.095],
    [0.038, 0.154, 0.195, 0.218, 0.195, 0.203, 0.177, 0.144, 0.144],
    [0.289, 0.255, 0.317, 0.331, 0.252, 0.256, 0.224, 0.187, 0.187],
    [0.433, 0.434, 0.419, 0.401, 0.33, 0.333, 0.271, 0.246, 0.246],
    [0.576, 0.552, 0.513, 0.47, 0.386, 0.366, 0.31, 0.278, 0.278],
    [0.661, 0.668, 0.585, 0.538, 0.445, 0.424, 0.346, 0.312, 0.312],
    [0.751, 0.744, 0.657, 0.605, 0.5, 0.469, 0.392, 0.349, 0.349],
    [0.856, 0.849, 0.743, 0.687, 0.551, 0.507, 0.426, 0.384, 0.384],
    [0.949, 0.913, 0.845, 0.772, 0.618, 0.546, 0.452, 0.417, 0.417],
    [1.015, 0.963, 0.918, 0.812, 0.698, 0.598, 0.488, 0.445, 0.445],
    [0.993, 0.993, 0.983, 0.9, 0.732, 0.636, 0.53, 0.463, 0.463],
    [1.056, 1.056, 1.018, 0.922, 0.804, 0.655, 0.558, 0.558, 0.558],
    [1.091, 1.091, 1.103, 0.956, 0.841, 0.682, 0.567, 0.567, 0.567],
    [1.107, 1.107, 1.156, 0.969, 0.868, 0.721, 0.619, 0.619, 0.619],
    [1.109, 1.109, 1.156, 1.014, 0.884, 0.753, 0.652, 0.652, 0.652],
]


def _f(cfg, key, default):
    try:
        v = float(cfg.get(key, default))
        return v if math.isfinite(v) else float(default)
    except (TypeError, ValueError):
        return float(default)


class TractionModel:
    """a(позиция контроллера, v) — таблица с линейной интерполяцией по скорости."""

    def __init__(self, table=None, v_bins=None, sanitize=True):
        src = table if _valid_table(table) else DEFAULT_TABLE
        rows = [[float(x) for x in r] for r in src]
        self.table = self._sanitize(rows) if sanitize else rows
        self.v_bins = v_bins if (isinstance(v_bins, (list, tuple)) and
                                 len(v_bins) == len(self.table[0]) + 1) else DEFAULT_V_BINS
        self.v_mid = [0.5 * (a + b) for a, b in zip(self.v_bins[:-1], self.v_bins[1:])]
        self.v_mid[-1] = min(self.v_mid[-1], self.v_bins[-2] + 2.0)

    @staticmethod
    def _sanitize(t):
        """Физические ограничения таблицы (ячейки с малой статистикой у остановки шумные).

        Тормозные позиции тормозят (a <= -0.05), на малых скоростях тормозной эффект не меньше,
        чем при 3–5 м/с; тяговые позиции не тормозят сильнее сопротивления движению.
        """
        out = [list(r) for r in t]
        for i, row in enumerate(out):
            u = i - 15
            if u <= -1:
                for j in range(len(row)):
                    row[j] = min(row[j], -0.05)
                if u <= -2 and len(row) > 2:
                    for j in (0, 1):
                        row[j] = min(row[j], row[2])
            elif u >= 1:
                for j in range(len(row)):
                    row[j] = max(row[j], -0.1)
        return out

    def lookup(self, cmd, v):
        """Значение таблицы с линейной интерполяцией по скорости."""
        row = self.table[int(max(-15, min(15, cmd))) + 15]
        vm = self.v_mid
        if v <= vm[0]:
            return row[0]
        if v >= vm[-1]:
            return row[-1]
        j = bisect.bisect_right(vm, v) - 1
        u = (v - vm[j]) / (vm[j + 1] - vm[j])
        return row[j] + (row[j + 1] - row[j]) * u

    def accel(self, cmd, v):
        a = self.lookup(cmd, v)
        if v <= 0.05 and a < 0:
            a = 0.0
        return a


def _valid_table(t):
    try:
        return (isinstance(t, (list, tuple)) and len(t) == 31 and
                all(len(r) == len(t[0]) and all(math.isfinite(float(x)) for x in r) for r in t))
    except (TypeError, ValueError):
        return False


class _Hist:
    """Кольцевой буфер состояний для оценки в прошлом (компенсация задержки, перестановки)."""

    def __init__(self, horizon=3.0):
        self.horizon = horizon
        self.items = []  # (t, D, v, a)

    def push(self, t, D, v, a):
        it = self.items
        if it and t < it[-1][0]:
            return
        if it and t == it[-1][0]:
            it[-1] = (t, D, v, a)
        else:
            it.append((t, D, v, a))
        if len(it) > 64 and it[0][0] < t - self.horizon:
            k = 0
            while k < len(it) - 2 and it[k][0] < t - self.horizon:
                k += 1
            del it[:k]

    def at(self, t):
        """(D, v, a) в момент t внутри буфера (линейная интерполяция) или None, если t позже."""
        it = self.items
        if not it or t >= it[-1][0]:
            return None
        if t <= it[0][0]:
            t0, D0, v0, a0 = it[0]
            return D0 - v0 * (t0 - t), v0, a0
        lo, hi = 0, len(it) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if it[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        t0, D0, v0, a0 = it[lo]
        t1, D1, v1, a1 = it[hi]
        u = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return D0 + (D1 - D0) * u, v0 + (v1 - v0) * u, a0 + (a1 - a0) * u


class BaselineEstimator:
    """Реализация контракта estimator_api (см. там описание методов)."""

    def __init__(self, config=None):
        cfg = dict(config or {})
        # свои коэффициенты — config/baseline_params.yaml (config['baseline_coefficients']);
        # config/params.yaml (config['coefficients']) — коэффициенты физической модели для основного оценщика.
        # Без baseline_coefficients (юнит-тесты, старые конфиги) — как раньше, из coefficients.
        if isinstance(cfg.get('baseline_coefficients'), dict):
            coef = cfg['baseline_coefficients']
        else:
            coef = cfg.get('coefficients') or {}
        if isinstance(coef, dict):
            cfg.update(coef)  # коэффициенты из файла важнее значений по умолчанию
        self.cfg = cfg
        veh = str(cfg.get('vehicle_id', '') or '')
        by_veh = cfg.get('velocity_scale_by_vehicle') or {}
        scale = _f(cfg, 'velocity_scale', 1.00034 / 3.6)
        if isinstance(by_veh, dict) and veh and veh in {str(k) for k in by_veh}:
            scale = _f({k: v for k, v in ((str(a), b) for a, b in by_veh.items())}, veh, scale)
        self.scale0 = scale
        # раздельная калибровка тележек: ρ = ln(v_передней / v_задней) на чистых участках (обе едут,
        # согласны, без проскальзывания, почти без ускорения); передняя делится на e^(ρ/2), задняя
        # умножается — средний масштаб не меняется (его калибруют ориентиры и детекторы масштаба).
        # Начальное значение — по трамваю (bogie_ratio_by_vehicle), дальше уточняется на ходу.
        self.bogie_calib = bool(cfg.get('bogie_calibration', True))
        self.bogie_tau = _f(cfg, 'bogie_ratio_tau_sec', 200.0)
        self.bogie_max = _f(cfg, 'bogie_ratio_max', 0.03)
        by_ratio = cfg.get('bogie_ratio_by_vehicle') or {}
        rho0 = 0.0
        if isinstance(by_ratio, dict) and veh:
            for k_, v_ in by_ratio.items():
                if str(k_) == veh:
                    try:
                        rho0 = math.log(float(v_))
                    except (TypeError, ValueError):
                        rho0 = 0.0
        self.bogie_rho0 = max(-self.bogie_max, min(self.bogie_max, rho0))
        self.scale = scale
        self.stale = _f(cfg, 'bogie_stale_sec', 0.6)
        self.delay = _f(cfg, 'output_delay_sec', 0.0)
        self.vel_delay = _f(cfg, 'velocity_delay_sec', 0.0)
        self.vel_tau = _f(cfg, 'velocity_output_tau_sec', 0.0)
        self.a_trac_max = _f(cfg, 'max_traction_accel', 1.8)
        self.a_brake_max = _f(cfg, 'max_brake_decel', 3.2)
        self.gate0 = _f(cfg, 'gate_abs', 0.4)
        self.gate_k = _f(cfg, 'gate_per_sec', 2.0)
        self.resync = _f(cfg, 'resync_sec', 1.0)
        self.slip_abs = _f(cfg, 'slip_diff_abs', 0.5)
        self.slip_rel = _f(cfg, 'slip_diff_rel', 0.08)
        self.v_max = _f(cfg, 'max_speed', 25.0)
        self.extrap_max = _f(cfg, 'max_extrapolation_sec', 30.0)
        self.map_match_max = _f(cfg, 'map_match_max_dist', 12.0)
        self.capture_dist = _f(cfg, 'map_capture_dist', 6.0)
        self.mu_nominal = _f(cfg, 'adhesion_nominal', 0.25)
        self.bogie_dist = _f(cfg, 'bogie_distance', 7.55)
        self.traction_lag = _f(cfg, 'traction_lag_sec', 0.2)
        self.gate_model_margin = _f(cfg, 'gate_model_margin', 0.3)  # м/с², 0 — без ограничения
        self.accel_tau = _f(cfg, 'accel_filter_tau_sec', 0.4)     # сглаживание измеренного ускорения
        self.accel_min_dt = _f(cfg, 'accel_min_dt_sec', 0.04)
        # проверка колёс по модели тяги (боксование/юз обеих тележек)
        self.slip_check = bool(cfg.get('slip_model_check', False))
        self.slip_thr_trac = _f(cfg, 'slip_model_threshold_traction', 0.5)
        self.slip_thr_brake = _f(cfg, 'slip_model_threshold_brake', 0.0)   # 0 — юз по модели не ищем
        self.slip_window = _f(cfg, 'slip_model_window_sec', 1.0)
        self.slip_max_sec = _f(cfg, 'slip_model_max_sec', 4.0)
        self.slip_min_speed = _f(cfg, 'slip_model_min_speed', 2.0)
        self.slip_min_notch = int(_f(cfg, 'slip_model_min_notch', 3))
        # продольная динамика с уклоном из карты: a = A_flat(u, v) - g*sin(уклон)
        self.use_grade = bool(cfg.get('traction_use_grade', True))
        default_table = DEFAULT_TABLE_FLAT if self.use_grade else DEFAULT_TABLE
        table = cfg.get('traction_table')
        self.model = TractionModel(table if _valid_table(table) else default_table,
                                   cfg.get('traction_v_bins'))
        # адаптация модели на ходу (загрузка/масса, состояние привода): a = k·A_ровн(u, v) − g·sin(уклон),
        # k — рекурсивный МНК с забыванием отдельно для тяги и торможения (Vahidi, Stefanopoulou, Peng,
        # «RLS with forgetting for online estimation of vehicle mass and road grade», VSD 2005)
        self.adapt = bool(cfg.get('model_adapt', True))
        self.adapt_tau = _f(cfg, 'model_adapt_tau_sec', 90.0)
        self.adapt_lo = _f(cfg, 'model_adapt_min', 0.6)
        self.adapt_hi = _f(cfg, 'model_adapt_max', 1.5)
        self.adapt_min_phi = _f(cfg, 'model_adapt_min_accel', 0.2)
        self.adapt_noise = _f(cfg, 'model_adapt_noise', 0.2) ** 2
        self.adapt_hold = _f(cfg, 'model_adapt_hold_sec', 1.0)
        self.adapt_reset_stop = _f(cfg, 'model_adapt_reset_stop_sec', 60.0)
        # проскальзывание обеих тележек сразу: ускорение колёс против модели с адаптивным порогом
        # (невязка «колёса − модель», её разброс оценивается на обычной езде)
        self.slip_acc = bool(cfg.get('slip_accel_check', True))
        self.sa_thr_min = _f(cfg, 'slip_accel_threshold_min', 0.4)
        self.sa_k = _f(cfg, 'slip_accel_k_sigma', 4.0)
        self.sa_persist = _f(cfg, 'slip_accel_persist_sec', 0.25)
        self.sa_max = _f(cfg, 'slip_accel_max_sec', 6.0)
        self.sa_min_speed = _f(cfg, 'slip_accel_min_speed', 2.0)
        self.sa_tau = _f(cfg, 'slip_accel_tau_sec', 0.3)
        self.sa_confirm = _f(cfg, 'slip_accel_confirm_sec', 3.5)   # разрыв не закрывается — ложная тревога
        self.sa_block_sec = _f(cfg, 'slip_accel_block_sec', 20.0)   # после неё не повторять на той же позиции
        # коридор разброса отклика по ячейкам (позиция, скорость) — tools/fit_traction_envelope.py;
        # порог детектора не меньше коридора: неоднозначные позиции (−8, −15) не дают ложных срабатываний
        spread = cfg.get('traction_spread_table')
        self.spread = TractionModel(spread, cfg.get('traction_v_bins'), sanitize=False) \
            if _valid_table(spread) else None
        self.sa_spread_margin = _f(cfg, 'slip_accel_spread_margin', 0.1)
        self.sa_spread_default = _f(cfg, 'slip_accel_spread_default', 0.3)
        m = cfg.get('antenna_master_xyz', [-9.873, 0.0, 3.0])
        r = cfg.get('antenna_rover_xyz', [2.563, 0.0, 3.0])
        self.ant = {'master': [float(v) for v in m], 'rover': [float(v) for v in r]}
        self.ant_w = {'master': _f(cfg, 'gnss_weight_master', 1.0),
                      'rover': _f(cfg, 'gnss_weight_rover', 0.15)}
        self.status_w = {2: 1.0, 0: _f(cfg, 'gnss_weight_status0', 0.7),
                         1: _f(cfg, 'gnss_weight_status1', 0.02)}
        self.proj = make_projection(cfg)
        files = [f for f in (cfg.get('map_files') or []) if f]
        self.map = TrackMap(files) if files else TrackMap([])
        # места остановок (стоп-линии/платформы) по обучающим записям — продольные ориентиры
        self.stop_min_sec = _f(cfg, 'stop_landmark_min_sec', 5.0)
        self.stop_gate_max = _f(cfg, 'stop_landmark_gate_max', 8.0)
        self.stop_sigma_min = _f(cfg, 'stop_landmark_sigma_min', 0.3)
        self.stop_sigma_std = str(cfg.get('stop_landmark_sigma_mode', 'std')) == 'std'
        self.drift_rate = _f(cfg, 'along_track_drift_rate', 0.0015)
        # самокалибровка масштаба колёс по ориентирам: фильтр Калмана [ошибка пути, ошибка масштаба]
        self.scale_learn = bool(cfg.get('scale_learning', False))
        self.scale_sigma0 = _f(cfg, 'scale_sigma0', 0.003)        # априорный разброс масштаба (0.3 %)
        self.scale_max_dev = _f(cfg, 'scale_max_dev', 0.025)      # предел поправки масштаба
        self.q_along = _f(cfg, 'along_track_noise', 2.5e-4)       # м²/м — случайная ошибка пути (буксование и т.п.)
        self.q_scale = _f(cfg, 'scale_noise', 1e-11)              # 1/м — дрейф масштаба
        # восстановленные по GNSS пути (петли у конечных) менее точны: на них продольная
        # неопределённость растёт быстрее (длина ломаной, стыки с выданными путями)
        rec = cfg.get('reconstructed_map_files', ['terminal_loops'])
        rec = {str(x) for x in rec} if isinstance(rec, (list, tuple)) else {'terminal_loops'}
        self.q_rec = _f(cfg, 'reconstructed_path_noise', 0.0)     # м²/м
        self.stop_ambiguity = _f(cfg, 'stop_landmark_ambiguity', 10.0)  # во сколько раз вероятнее
        self.stop_secondary_w = _f(cfg, 'stop_landmark_secondary_weight', 0.1)
        # грубая ошибка масштаба колёс (так бывает: 03.09 масштаб отличался на 0.63 %):
        # два подряд интервала между ориентирами на выданных путях с одинаковой по знаку
        # «скоростью ухода» больше порога -> поправка масштаба
        self.scale_detect = bool(cfg.get('scale_anomaly_detect', True))
        self.scale_detect_thr = _f(cfg, 'scale_anomaly_threshold', 0.003)
        self.scale_detect_min_dD = _f(cfg, 'scale_anomaly_min_interval', 150.0)
        self.scale_detect_max_slip = _f(cfg, 'scale_anomaly_max_slip_sec', 1.0)  # с проскальзывания — интервал не в счёт
        # грубая ошибка масштаба (1–1.5 %, 03.09 утром): ориентиры уже не попадают в строб, поэтому
        # последовательность остановок сопоставляется с ориентирами «в целом» (RANSAC по парам)
        self.rs_enable = bool(cfg.get('scale_ransac', True))
        self.rs_min_inliers = int(_f(cfg, 'scale_ransac_min_inliers', 4))
        self.rs_min_rate = _f(cfg, 'scale_ransac_min_rate', 0.0035)
        self.rs_tol = _f(cfg, 'scale_ransac_tolerance', 1.2)
        self.rs_min_span = _f(cfg, 'scale_ransac_min_span', 500.0)   # м пути между крайними остановками
        self.rs_long_span = _f(cfg, 'scale_ransac_long_span', 2000.0)  # при таком размахе хватает на 1 меньше
        self.stop_creep_dist = _f(cfg, 'stop_landmark_creep_dist', 20.0)   # м — «подтягивание»
        self.stop_creep_gate = _f(cfg, 'stop_landmark_creep_gate', 1.0)    # м
        self._path_q = [self.q_rec if getattr(p, 'source', '') in rec else 0.0
                        for p in self.map.paths]
        self._rec_pids = {i for i, p in enumerate(self.map.paths) if getattr(p, 'source', '') in rec}
        self.landmarks = self._load_landmarks(cfg.get('stop_landmarks_file'))
        # ветки, которых нет в графе (средний путь «Таллинской»): после стрелки положение выдаётся
        # взвешенно по вероятности ветки из обучающих записей — оценка с минимумом среднего квадрата
        # ошибки, когда без GNSS и IMU нельзя узнать, куда ушёл трамвай; разброс гипотез — в дисперсию
        self.branch_blend = bool(cfg.get('branch_blend', True))
        self.branch_stand_tau = _f(cfg, 'branch_stand_tau_sec', 15.0)  # на стоянке вес ветки убывает (0 — не убывает)
        self.branches = self._load_branches(cfg.get('branch_map_files') or [],
                                            cfg.get('branch_probability'))
        self.reset()

    def _load_branches(self, files, p_override=None):
        out = []
        names = {p.name: i for i, p in enumerate(self.map.paths)}
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    meta = json.load(fh).get('meta') or {}
                alt = TrackMap([f])
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                continue
            main = names.get(str(meta.get('switch_path', '')))
            if main is None or not alt.paths:
                continue
            try:
                prob = float(p_override if p_override is not None else meta.get('probability', 0.15))
            except (TypeError, ValueError):
                prob = 0.15
            if not 0.0 < prob < 1.0:
                continue
            L = alt.length(0)
            out.append({'main': main, 's_sw': float(meta.get('switch_s', 0.0)), 'alt': alt,
                        'len': L, 'max_dist': L + 10.0, 'p': prob, 'name': alt.paths[0].name})
        return out

    def _load_landmarks(self, path):
        lm = {}
        if not path or self.map.empty:
            return lm
        try:
            import json
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return lm
        names = {p.name: i for i, p in enumerate(self.map.paths)}
        for d in data if isinstance(data, list) else []:
            try:
                pid = names.get(str(d['track']))
                if pid is None or int(d.get('n', 0)) < int(self.cfg.get('stop_landmark_min_n', 4)):
                    continue
                # погрешность места остановки: по MAD (ядро распределения) или по СКО (с учётом
                # редких остановок в 1–2 м от обычного места — «хвосты»)
                spread = float(d.get('std', d.get('mad', 0.5))) if self.stop_sigma_std \
                    else 1.4826 * float(d.get('mad', 0.5))
                sig = max(self.stop_sigma_min, spread)
                lm.setdefault(pid, []).append((float(d['s']), sig, int(d.get('n', 0))))
            except (KeyError, TypeError, ValueError):
                continue
        for v in lm.values():
            v.sort()
        # «вторичные» места: рядом (< 10 м) более частое место остановки. Трамвай почти всегда
        # сначала встаёт у основного (стоп-линия) и лишь иногда подтягивается ко вторичному,
        # поэтому для первой остановки у платформы вторичное место маловероятно.
        for v in lm.values():
            for i, (s_lm, sig, n) in enumerate(v):
                sec = any(abs(s2 - s_lm) < 10.0 and n2 > n for s2, _, n2 in v)
                v[i] = (s_lm, sig, n * (self.stop_secondary_w if sec else 1.0))
        return lm

    # ------------------------------------------------------------------ жизненный цикл
    def reset(self):
        self.t = None
        self.t_v = None
        self._gap_t0 = None
        self.v_out = None
        self.t_out = None
        self.v = 0.0
        self.a = 0.0
        self._a_ref = None            # (t, v) — опорная точка для наклона скорости
        self.D = 0.0
        self.hist = _Hist()
        self.front = None             # (t, v_mps)
        self.rear = None
        self.cmds = []                # (t, cmd) — для запаздывания отклика тяги
        self.cmd = 0
        self.slip = False
        self.slip_front = False
        self.slip_rear = False
        self.slip_ratio = 0.0
        self.mu = math.nan
        self.mu_t = None
        self.vel_valid = False
        self.agree_since = None
        self.n_meas = 0
        self._sin_slope = 0.0
        self._slope_D = -1e9
        self._vbuf = []          # (t, v) оценки на моменты измерений — для проверки по модели
        self.slip_mode = 0       # 0 — нет; +1 — боксование (тяга); -1 — юз (торможение)
        self.k_mod = {1: 1.0, -1: 1.0}   # масштаб модели: тяга / торможение
        self.P_mod = {1: 0.05, -1: 0.05}
        self._cmd_change_t = None
        self._still_t0 = None
        self.n_adapt = 0
        self._w_prev = None      # (t, среднее колёс) — ускорение колёс для детектора
        self._a_w = 0.0
        self._sa_sigma = 0.2
        self._sa_since = None    # (t, v, D, знак) — начало подозрения
        self.sa_mode = 0
        self.sa_t0 = None
        self.sa_v = 0.0
        self.sa_ok_since = None
        self.n_sa_events = 0
        self.n_sa_undo = 0
        self.sa_D0 = 0.0
        self.sa_dD = 0.0
        self.sa_gap_max = 0.0
        self.sa_mode_last = 0
        self._sa_block = None
        self._sa_block_u = None
        self.slip_t0 = None
        self.slip_vm = 0.0
        self.slip_ok_since = None
        self.n_slip_events = 0
        self.mode = 'NO_FIX'
        self.dir = 1
        self.captured = False
        self._capture_D = -1e9
        self.pid = None
        self.s_ref = None
        self.D_ref = 0.0
        self.xy_ref = None
        self.anchors = []
        self.xy_anchors = []
        self.last_fix = {}
        self.gnss_closed = False
        self.n_fix = {'master': 0, 'rover': 0}
        self.n_corr = 0
        self._last_corr = None
        self._reanchor_votes = []
        self.still_since = None
        self.stop_done = False
        self.scale = self.scale0
        self.bogie_rho = self.bogie_rho0 if self.bogie_calib else 0.0
        self.n_bogie_upd = 0
        self.var_along = 0.25
        self.P_de = 0.0               # ковариация [ошибка пути, ошибка масштаба]
        self.P_ee = 0.0               # дисперсия ошибки масштаба (0 до окончания GNSS-старта)
        self.D_var = 0.0
        self.n_lm = 0
        self.last_lm = None
        self._lm_D = 0.0
        self._prev_stop = None        # (D, pid, s, s_ориентира или None) — прошлая остановка
        self._lm_hist = []            # принятые поправки у ориентиров: (D, на выданном пути, pid)
        self._rates = []              # «скорость ухода» пути между ориентирами, 1/м
        self._suspect_rate = 0.0
        self._slip_sec = 0.0            # время с проскальзыванием с прошлого ориентира
        self.n_scale_fix = 0
        self._rs_init = None          # (pid, s, D) — привязка, от которой считается чистое счисление
        self._rs_stops = []           # (A, [s_ориентира - s_счисления, ...]) по остановкам
        self._D_init = 0.0

    # ------------------------------------------------------------------ входы
    def on_front_velocity(self, t, v_raw):
        self._bogie('front', t, v_raw)

    def on_rear_velocity(self, t, v_raw):
        self._bogie('rear', t, v_raw)

    def on_driver_cmd(self, t, position):
        try:
            p = int(position)
        except (TypeError, ValueError):
            return
        if not -15 <= p <= 15:
            return
        if p != self.cmd or self._cmd_change_t is None:
            self._cmd_change_t = t
        self.cmd = p
        if not self.cmds or t >= self.cmds[-1][0]:
            self.cmds.append((t, p))
            if len(self.cmds) > 200:
                del self.cmds[:100]
        self._advance(t)

    def _bogie(self, which, t, v_raw):
        try:
            v = float(v_raw) * self.scale
        except (TypeError, ValueError):
            return
        if self.bogie_rho:
            v *= math.exp(-0.5 * self.bogie_rho if which == 'front' else 0.5 * self.bogie_rho)
        if not math.isfinite(v) or v < -1.0 or v > self.v_max * 1.5:
            return
        setattr(self, which, (t, max(v, 0.0)))
        self._advance(t, measure=True)

    # ------------------------------------------------------------------ модель
    def _cmd_at(self, t):
        c = self.cmds
        if not c:
            return self.cmd
        tq = t - self.traction_lag
        if tq >= c[-1][0]:
            return c[-1][1]
        i = bisect.bisect_right(c, (tq, 99)) - 1
        return c[i][1] if i >= 0 else c[0][1]

    def _model_accel(self, t, v):
        u = self._cmd_at(t)
        a = self.model.accel(u, v)
        if self.adapt and u != 0:
            a *= self.k_mod[1 if u > 0 else -1]
        if self.use_grade:
            self._update_slope()
            a -= G * self._sin_slope
            if v <= 0.05 and a < 0.0 and self._cmd_at(t) <= 0:
                a = 0.0   # стоящий на тормозе трамвай под уклон не катится
        return a

    def _update_slope(self):
        """sin(уклона) по ходу движения — по хорде между тележками на текущем месте карты."""
        if self.mode != 'MAP' or self.pid is None:
            self._sin_slope = 0.0
            return
        if abs(self.D - self._slope_D) < 1.0:
            return
        self._slope_D = self.D
        s = self.s_ref + self.dir * (self.D - self.D_ref)
        pid, s, _ = self.map.advance(self.pid, s, 0.0)
        p1 = self.map.pose(pid, s)
        pid0, s0, _ = self.map.advance(pid, s, -self.dir * self.bogie_dist)
        p0 = self.map.pose(pid0, s0)
        d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        sl = (p1[2] - p0[2]) / max(d, 1.0)
        self._sin_slope = max(-0.08, min(0.08, sl))

    def _propagate(self, v, D, t0, t1):
        """Движение по модели тяги от t0 до t1 (без колёсных данных)."""
        t = t0
        a = 0.0
        while t < t1 - 1e-9:
            h = min(0.1, t1 - t)
            a = self._model_accel(t, v)
            v2 = min(max(v + a * h, 0.0), self.v_max)
            D += 0.5 * (v + v2) * h
            v = v2
            t += h
        return v, D, a

    # ------------------------------------------------------------------ ядро
    def _fresh(self, meas, t):
        return meas is not None and (t - meas[0]) <= self.stale

    def _advance(self, t, measure=False):
        """Шаг оценщика по входу с временем t.

        Часы интегрирования пути (self.t) — максимум времени входов. Скорость обновляется по
        времени самого измерения тележки (self.t_v); измерение, пришедшее «из прошлого»
        (другой топик доставлен позже), учитывается ретроспективной поправкой пути.
        """
        if self.t is None:
            self.t = t
            self.t_v = t
            f = self.front if self._fresh(self.front, t) else None
            r = self.rear if self._fresh(self.rear, t) else None
            vals = [m[1] for m in (f, r) if m]
            if vals:
                self.v = sum(vals) / len(vals)
                self.vel_valid = True
            self.hist.push(t, self.D, self.v, 0.0)
            return
        if t > self.t:
            dt = t - self.t
            tq = self.t + dt
            if not (self._fresh(self.front, tq) or self._fresh(self.rear, tq)):
                # нет колёсных данных — движение по модели тяги
                if self._gap_t0 is None:
                    self._gap_t0 = self.t
                self.v, self.D, self.a = self._propagate(self.v, self.D, self.t, t)
                self.slip = False
                self.t_v = t
            else:
                self.D += self.v * dt
            self.t = t
        if measure:
            f = self.front if self._fresh(self.front, t) else None
            r = self.rear if self._fresh(self.rear, t) else None
            if f is not None or r is not None:
                dt_v = max(t - self.t_v, 0.0)
                v_old = self.v
                if self._gap_t0 is not None:
                    # колёса вернулись после пропуска: им доверяем сразу, а путь за пропуск
                    # поправляем — ошибка скорости модели нарастала ~линейно от начала пропуска
                    T = t - self._gap_t0
                    self._gap_t0 = None
                    vals = [m[1] for m in (f, r) if m]
                    v_w = min(vals) if len(vals) == 2 and abs(vals[0] - vals[1]) > 0.5 \
                        else sum(vals) / len(vals)
                    if 0.2 < T < 60.0:
                        self.D -= 0.5 * (v_old - v_w) * T
                    self.v = v_w
                    self.vel_valid = True
                    self._a_ref = None
                    self.t_v = t
                    self.hist.push(self.t, self.D, self.v, self.a)
                    return
                self._fuse(t, dt_v, f, r)
                # новая скорость действует с середины интервала между измерениями
                t_switch = t - 0.5 * min(dt_v, 0.3)
                lag = self.t - t_switch
                if 0.0 < lag < 1.0:
                    self.D += (self.v - v_old) * lag
                self.t_v = max(self.t_v, t)
                self.n_meas += 1
                if self.mode == 'OFF_MAP' and self.gnss_closed:
                    self._try_capture()
                if self.landmarks and self.mode == 'MAP' and self.gnss_closed:
                    self._stop_landmark(t)
                # сглаженная скорость для выхода (эталон — отфильтрованная оценка, запаздывает)
                if self.vel_tau > 0.0:
                    if self.v_out is None:
                        self.v_out = self.v
                    else:
                        k = 1.0 - math.exp(-max(dt_v, 0.0) / self.vel_tau)
                        self.v_out += (self.v - self.v_out) * k
        self.hist.push(self.t, self.D, self.v, self.a)

    def _fuse(self, t, dt, f, r):
        a_meas = self.a
        a_mod = self._model_accel(t, self.v)
        # прогноз: измеренное ускорение (если данные были недавно), иначе модель. Измеренное
        # ограничено моделью ± gate_model_margin: разгон/замедление сверх того, что даёт позиция
        # ручки, — признак проскальзывания, строб не должен «уезжать» вслед за колесом
        a_pred = a_meas if self.vel_valid else a_mod
        if self.vel_valid and self.gate_model_margin > 0.0:
            a_pred = min(max(a_pred, a_mod - self.gate_model_margin), a_mod + self.gate_model_margin)
        a_pred = min(max(a_pred, -self.a_brake_max), self.a_trac_max)
        v_pred = max(0.0, self.v + a_pred * dt)
        gate = self.gate0 + self.gate_k * dt
        cands = [('front', f[1]) if f else None, ('rear', r[1]) if r else None]
        cands = [c for c in cands if c is not None]
        ok = [c for c in cands if abs(c[1] - v_pred) <= gate]
        self.slip_front = f is not None and all(c[0] != 'front' for c in ok)
        self.slip_rear = r is not None and all(c[0] != 'rear' for c in ok)
        both_agree = f is not None and r is not None and \
            abs(f[1] - r[1]) <= max(self.slip_abs, self.slip_rel * max(f[1], r[1]))
        if not self.vel_valid:
            vm = sum(c[1] for c in cands) / len(cands)
            slip = False
        elif len(ok) == len(cands):
            vm = sum(c[1] for c in ok) / len(ok)
            slip = False
        elif ok:
            vm = ok[0][1]
            slip = True
        else:
            # обе вне строба
            closest = min(cands, key=lambda c: abs(c[1] - v_pred))[1]
            if both_agree:
                if self.agree_since is None:
                    self.agree_since = t
                if t - self.agree_since >= self.resync:
                    vm = 0.5 * (f[1] + r[1])   # тележки согласны долго — доверяем им
                    slip = False
                    self.agree_since = None
                else:
                    vm = min(max(closest, self.v - self.a_brake_max * dt),
                             self.v + self.a_trac_max * dt)
                    slip = True
            else:
                vm = min(max(closest, self.v - self.a_brake_max * dt),
                         self.v + self.a_trac_max * dt)
                slip = True
        if ok:
            self.agree_since = None
        if self.slip_check and self.vel_valid:
            vm, slip2 = self._model_slip_check(t, dt, vm, f, r)
            slip = slip or slip2
        if self.slip_acc and self.vel_valid:
            vm, slip3 = self._accel_slip_check(t, dt, vm, f, r)
            slip = slip or slip3
        self.v = min(max(vm, 0.0), self.v_max)
        if self.slip_check:
            self._vbuf.append((t, self.v))
            if len(self._vbuf) > 80:
                del self._vbuf[:len(self._vbuf) - 60]
        # ускорение — наклон скорости по времени самих измерений. Половина вызовов приходит с
        # dt = 0 (измерение второй тележки старше последнего), раньше их изменения скорости в
        # ускорение не попадали и оно выходило ~0.5 от истинного. Теперь приращение считается от
        # опорной точки (t, v), сдвигаемой не чаще accel_min_dt, — ничего не теряется.
        ms = [m for m in (f, r) if m is not None]
        t_m = sum(m[0] for m in ms) / len(ms)
        if not self.vel_valid or self._a_ref is None or t_m < self._a_ref[0] - 0.5:
            self._a_ref = (t_m, self.v)
        else:
            h = t_m - self._a_ref[0]
            if h >= self.accel_min_dt:
                a_new = (self.v - self._a_ref[1]) / h
                k = min(1.0, h / self.accel_tau)
                self.a += (a_new - self.a) * k
                self.a = min(max(self.a, -self.a_brake_max), self.a_trac_max)
                self._a_ref = (t_m, self.v)
        self.vel_valid = True
        if self.adapt:
            self._adapt_model(t, dt, slip, both_agree)
        # относительное проскальзывание и оценка сцепления
        wheels = [c[1] for c in cands]
        if self.v > 0.5 and wheels:
            dev = max(wheels, key=lambda w: abs(w - self.v))
            self.slip_ratio = (dev - self.v) / self.v
        else:
            self.slip_ratio = 0.0
        if slip:
            self._slip_sec += min(max(dt, 0.0), 0.2)
        elif self.bogie_calib and f is not None and r is not None and both_agree and \
                min(f[1], r[1]) > 3.0 and abs(f[0] - r[0]) < 0.02 and abs(self.a) < 0.5:
            # пара показаний одного момента (тележки опрашиваются синхронно, 10 Гц); они уже
            # поправлены на текущее ρ -> невязка ln(f/r) — поправка к ρ, каждая пара с весом 0.1 с / τ
            k = min(1.0, 0.1 / self.bogie_tau)
            self.bogie_rho += k * math.log(f[1] / r[1])
            self.bogie_rho = max(-self.bogie_max, min(self.bogie_max, self.bogie_rho))
            self.n_bogie_upd += 1
        if slip and not self.slip:
            # в момент срыва реализуемое сцепление ~ |a|/g (ускорение по модели для этой позиции)
            self.mu = max(0.03, min(0.5, max(abs(self.a), abs(a_mod)) / G))
            self.mu_t = t
        self.slip = slip

    def _adapt_model(self, t, dt, slip, both_agree):
        """Рекурсивный МНК с забыванием: масштаб модели тяги/торможения по измеренному ускорению.

        Обновляется только на «чистых» участках: тележки согласны, нет проскальзывания, скорость
        > 2 м/с, позиция контроллера держится >= model_adapt_hold_sec (отклик привода установился),
        модель даёт заметное ускорение. После долгой стоянки уверенность снижается (другая загрузка).
        """
        if self.v < 0.05:
            if self._still_t0 is None:
                self._still_t0 = t
            elif t - self._still_t0 > self.adapt_reset_stop:
                for key in (1, -1):
                    self.P_mod[key] = max(self.P_mod[key], 0.05)
            return
        self._still_t0 = None
        if slip or not both_agree or self.v < 2.0 or self.mode != 'MAP' or dt <= 0.0:
            return
        u = self._cmd_at(t)
        if u == 0 or self._cmd_change_t is None or \
                t - self._cmd_change_t < self.adapt_hold + self.traction_lag:
            return
        phi = self.model.accel(u, self.v)
        if abs(phi) < self.adapt_min_phi:
            return
        self._update_slope()
        y = self.a + G * self._sin_slope          # ускорение «на ровном»
        key = 1 if u > 0 else -1
        lam = math.exp(-dt / self.adapt_tau)
        P = self.P_mod[key] / lam
        K = P * phi / (self.adapt_noise + phi * P * phi)
        k = self.k_mod[key] + K * (y - self.k_mod[key] * phi)
        self.k_mod[key] = min(max(k, self.adapt_lo), self.adapt_hi)
        self.P_mod[key] = min((1.0 - K * phi) * P, 1.0)
        self.n_adapt += 1

    def _accel_slip_check(self, t, dt, vm, f, r):
        """Боксование/юз обеих тележек: ускорение колёс расходится с моделью по ручке.

        Третий «голос» против двух согласных тележек. Невязка «ускорение колёс − модель»
        сравнивается с адаптивным порогом max(порог_мин, k·σ, коридор ячейки + запас): σ — разброс
        невязки на обычной езде (оценивается на ходу), коридор — traction_spread_table (разброс
        отклика на эту позицию ручки при этой скорости по обучающим записям). Подозрение должно
        держаться slip_accel_persist_sec. При срабатывании скорость ведётся по модели от начала
        подозрения, но с физическими ограничениями: при тяге не выше самой медленной тележки, при
        торможении не ниже самой быстрой; путь за окно подозрения поправляется. Выход — колёса
        снова согласны с оценкой 0.3 с или прошло slip_accel_max_sec. Ложная тревога (разрыв
        «модель − колёса» не закрывается slip_accel_confirm_sec или колёса встали) — откат к
        колёсам с ретроспективной поправкой пути и блокировка повтора на той же позиции.
        """
        wheels = [m[1] for m in (f, r) if m is not None]
        if not wheels:
            return vm, False
        wmean = sum(wheels) / len(wheels)
        if self._w_prev is None:
            self._w_prev = (t, wmean)
        elif t - self._w_prev[0] >= 0.02:
            h = t - self._w_prev[0]
            a_raw = (wmean - self._w_prev[1]) / h
            self._a_w += (a_raw - self._a_w) * min(1.0, h / self.sa_tau)
            self._w_prev = (t, wmean)
        if self.sa_mode:
            self.sa_v, _, _ = self._propagate(self.sa_v, 0.0, t - dt, t)
            v_new = min(self.sa_v, min(wheels)) if self.sa_mode > 0 else max(self.sa_v, max(wheels))
            gap = abs(v_new - wmean)
            self.sa_gap_max = max(self.sa_gap_max, gap)
            if abs(wmean - v_new) < max(0.25, 0.04 * v_new):
                if self.sa_ok_since is None:
                    self.sa_ok_since = t
            else:
                self.sa_ok_since = None
            # ложная тревога: настоящий срыв сцепления проходит (противобоксовочная/противоюзная
            # защита возвращает колёса к скорости трамвая за 1–3 с), и разрыв «модель − колёса»
            # начинает закрываться. Если за sa_confirm_sec он всё ещё у максимума (или колёса
            # встали) — ошиблась модель (например, экстренное торможение рельсовым тормозом):
            # возвращаемся к колёсам и ретроспективно убираем набранную разницу пути.
            stopped = max(wheels) < 0.3 and gap > 0.5
            if (t - self.sa_t0 >= self.sa_confirm and gap >= 0.85 * self.sa_gap_max) or \
                    (stopped and t - self.sa_t0 >= 1.0):
                self.D -= self.sa_dD
                self.D += self.sa_D0
                self.sa_mode = 0
                self._sa_since = None
                self._sa_block = (self.sa_mode_last, t + self.sa_block_sec)
                self.n_sa_undo += 1
                return wmean, False
            if (self.sa_ok_since is not None and t - self.sa_ok_since >= 0.3) or \
                    t - self.sa_t0 > self.sa_max:
                self.sa_mode = 0
                self._sa_since = None
                return vm, False
            self.sa_dD += (v_new - wmean) * max(dt, 0.0)
            self.sa_v = v_new
            return v_new, True
        u = self._cmd_at(t)
        res = self._a_w - self._model_accel(t, self.v)
        spread = self.spread.lookup(u, self.v) if self.spread is not None else self.sa_spread_default
        thr = max(self.sa_thr_min, self.sa_k * self._sa_sigma, spread + self.sa_spread_margin)
        susp = 0
        if max(self.v, wmean) >= self.sa_min_speed:
            if u > 0 and res > thr:
                susp = 1
            elif u < 0 and res < -thr:
                susp = -1
        if not susp:
            if dt > 0.0 and self.v >= self.sa_min_speed:
                self._sa_sigma += (1.25 * min(abs(res), 1.0) - self._sa_sigma) * min(1.0, dt / 20.0)
                self._sa_sigma = min(max(self._sa_sigma, 0.08), 0.5)
            self._sa_since = None
            return vm, False
        if self._sa_block is not None and self._sa_block[0] == susp and t < self._sa_block[1] and \
                u == self._sa_block_u:
            return vm, False   # после ложной тревоги — не повторять её на том же торможении/разгоне
        if self._sa_since is None or self._sa_since[3] != susp:
            self._sa_since = (t, self.v, self.D, susp)
            return vm, False
        if t - self._sa_since[0] < self.sa_persist:
            return vm, False
        t0, v0, _, mode = self._sa_since
        v_pred, _, _ = self._propagate(v0, 0.0, t0, t)
        e = vm - v_pred
        self.sa_D0 = 0.5 * e * (t - t0)
        self.D -= self.sa_D0                  # путь за окно подозрения шёл за колёсами
        self.sa_mode = mode
        self.sa_mode_last = mode
        self._sa_block_u = u
        self.sa_t0 = t
        self.sa_ok_since = None
        self.sa_dD = 0.0
        self.sa_gap_max = 0.0
        self.n_sa_events += 1
        v_new = min(v_pred, min(wheels)) if mode > 0 else max(v_pred, max(wheels))
        self.sa_v = v_new
        return v_new, True

    def _model_slip_check(self, t, dt, vm, f, r):
        """Боксование/юз обеих тележек: колёса расходятся с моделью тяги за последнюю секунду.

        Прогноз по модели a(позиция, v) от оценки секундной давности: при тяге колёса не
        могут разгоняться заметно быстрее модели, при торможении — замедляться заметно быстрее.
        При срабатывании оценка идёт по модели, ограниченная колёсами (при боксовании истинная
        скорость не выше самой медленной тележки, при юзе — не ниже самой быстрой), путь за
        последнюю секунду ретроспективно поправляется. Выход — когда колёса снова согласны с
        оценкой 0.5 с или через slip_model_max_sec.
        """
        wheels = [m[1] for m in (f, r) if m is not None]
        if not wheels:
            return vm, False
        u = self._cmd_at(t)
        if self.slip_mode:
            self.slip_vm, _, _ = self._propagate(self.slip_vm, 0.0, t - dt, t)
            if self.slip_mode > 0:
                v_new = min(self.slip_vm, min(wheels))
            else:
                v_new = max(self.slip_vm, max(wheels))
            wmean = sum(wheels) / len(wheels)
            if abs(wmean - v_new) < 0.3:
                if self.slip_ok_since is None:
                    self.slip_ok_since = t
            else:
                self.slip_ok_since = None
            if (self.slip_ok_since is not None and t - self.slip_ok_since >= 0.3) or \
                    t - self.slip_t0 > self.slip_max_sec:
                self.slip_mode = 0
                self._vbuf = []
                return vm, False
            self.slip_vm = v_new
            return v_new, True
        # поиск оценки ~slip_window назад
        ref = None
        for tb, vb in reversed(self._vbuf):
            if t - tb >= self.slip_window:
                ref = (tb, vb)
                break
        if ref is None or t - ref[0] > 2.0 * self.slip_window:
            return vm, False
        v_pred, _, _ = self._propagate(ref[1], 0.0, ref[0], t)
        e = vm - v_pred
        if vm < self.slip_min_speed:
            return vm, False
        if u >= self.slip_min_notch and self.slip_thr_trac > 0 and e > self.slip_thr_trac:
            mode = 1
        elif u < 0 and self.slip_thr_brake > 0 and e < -self.slip_thr_brake:
            mode = -1
        else:
            return vm, False
        self.slip_mode = mode
        self.slip_t0 = t
        self.slip_ok_since = None
        self.slip_vm = v_pred
        self.n_slip_events += 1
        # ретроспективно: за окно оценка шла за колёсами, ошибка нарастала ~линейно
        self.D -= 0.5 * e * (t - ref[0])
        v_new = min(v_pred, min(wheels)) if mode > 0 else max(v_pred, max(wheels))
        return v_new, True

    # ------------------------------------------------------------------ ориентиры-остановки
    def _stop_landmark(self, t):
        """Остановка >= stop_min_sec у известного места остановки — продольная коррекция.

        Водители останавливаются у стоп-линий и платформ очень повторяемо (по обучающим
        записям разброс 0.1–0.3 м), а колёсный путь «плывёт» ~0.1 % от пройденного.
        Коррекция — шаг фильтра Калмана по продольной координате, строб по текущей
        неопределённости (не больше stop_landmark_gate_max).
        """
        # рост продольной неопределённости с пройденным путём
        dD = abs(self.D - self.D_var)
        if dD > 0.0:
            q_path = 0.0
            if self.q_rec > 0.0 and self._path_q:
                pid_now, _, _ = self.map.advance(self.pid, self.s_ref + self.dir * (self.D - self.D_ref), 0.0)
                q_path = self._path_q[pid_now] * dD
            self.var_along += q_path
            if self.scale_learn:
                # ошибка пути нарастает как (ошибка масштаба) * путь + случайная составляющая
                self.var_along += 2.0 * dD * self.P_de + dD * dD * self.P_ee + self.q_along * dD
                self.P_de += dD * self.P_ee
                self.P_ee += self.q_scale * dD
            else:
                rate = max(self.drift_rate, 1.5 * self._suspect_rate)
                self.var_along = (math.sqrt(self.var_along) + rate * dD) ** 2
            self.D_var = self.D
        f = self.front if self._fresh(self.front, t) else None
        r = self.rear if self._fresh(self.rear, t) else None
        vals = [m[1] for m in (f, r) if m]
        still = self.v < 0.05 and vals and max(vals) < 0.05
        if not still:
            self.still_since = None
            self.stop_done = False
            return
        if self.still_since is None:
            self.still_since = t
            return
        if self.stop_done or t - self.still_since < self.stop_min_sec:
            return
        self.stop_done = True
        if self.rs_enable and self._scale_ransac():
            return   # положение и масштаб только что пересчитаны по серии остановок
        s_est = self.s_ref + self.dir * (self.D - self.D_ref)
        pid, s_e, off_end = self.map.advance(self.pid, s_est, 0.0)
        prev, self._prev_stop = self._prev_stop, (self.D, pid, s_e, None)
        if off_end or self.dir < 0:
            return
        lms = self.landmarks.get(pid)
        if not lms:
            return
        sig_est = math.sqrt(self.var_along)
        gate = min(self.stop_gate_max, max(2.5, 3.0 * sig_est))
        creep = prev is not None and prev[1] == pid and 0.3 < self.D - prev[0] < self.stop_creep_dist
        best = None
        if creep and prev[3] is not None:
            # подтянулись после остановки у известного ориентира: смещение точно известно по
            # колёсам — ищем другой ориентир рядом с ожидаемым местом (строб ~1 м)
            c = [(abs(L[0] - s_e), L) for L in lms if abs(L[0] - prev[3]) > 0.5]
            if c:
                dist, L = min(c)
                if dist <= self.stop_creep_gate:
                    best = (L[0] - s_e, L[1], L[0])
        elif creep:
            # прошлая остановка не сопоставлена (неоднозначно): сопоставляем пару остановок с
            # парой ориентиров по смещению между ними
            delta = self.D - prev[0]
            pairs = []
            for L1 in lms:
                if abs(L1[0] - prev[2]) > gate:
                    continue
                for L2 in lms:
                    if L2[0] > L1[0] and abs((L2[0] - L1[0]) - delta) < self.stop_creep_gate:
                        pairs.append(L2)
            if len(pairs) == 1:
                L = pairs[0]
                best = (L[0] - s_e, L[1], L[0])
        else:
            # первая остановка в этом месте: правдоподобие ориентиров в стробе с учётом частоты
            # остановок у них; выбранный должен быть намного вероятнее остальных
            cands = []
            for s_lm, sig_lm, n in lms:
                d = s_lm - s_e
                if abs(d) <= gate:
                    S = self.var_along + sig_lm ** 2
                    cands.append((max(n, 0.01) * math.exp(-0.5 * d * d / S) / math.sqrt(S),
                                  d, sig_lm, s_lm))
            if cands:
                cands.sort(reverse=True)
                rest = sum(c[0] for c in cands[1:])
                if rest <= 0.0 or cands[0][0] >= self.stop_ambiguity * rest:
                    best = cands[0][1:]
        if best is None:
            return
        d, sig_lm, s_lm = best
        self._prev_stop = (self.D, pid, s_e + d, s_lm)
        S = self.var_along + sig_lm ** 2
        K = self.var_along / S
        self.s_ref += self.dir * K * d
        if self.scale_learn and self.P_ee > 0.0:
            # поправка масштаба колёс: трамвай «недоехал» по счислению -> масштаб занижен
            Ke = self.P_de / S
            new_scale = self.scale * (1.0 + Ke * d)
            lo = self.scale0 * (1.0 - self.scale_max_dev)
            hi = self.scale0 * (1.0 + self.scale_max_dev)
            self.scale = min(max(new_scale, lo), hi)
            self.P_ee = max(self.P_ee - Ke * self.P_de, 1e-10)
            self.P_de = (1.0 - K) * self.P_de
        self.var_along = (1.0 - K) * self.var_along
        self.n_lm += 1
        self.last_lm = (t, self.map.paths[pid].name, s_lm, d, K)
        if self.scale_detect and not self.scale_learn:
            self._scale_anomaly(pid, d)
        self._lm_D = self.D

    def _scale_ransac(self):
        """Грубая ошибка масштаба колёс по серии остановок (RANSAC по парам «остановка–ориентир»).

        Чистое счисление от стартовой привязки: s_raw = s_start + A, A — колёсный путь. Для каждой
        остановки берутся ориентиры в окне ±max(15 м, 3.5 % A) и смещения d = s_ориентира − s_raw.
        Модель: d = d0 + r·A (d0 — ошибка стартовой привязки, r — ошибка масштаба). Гипотезы — по
        парам остановок, поддержка — число остановок у ориентира (±1.2 м). Срабатывает, если лучшая
        гипотеза объясняет ≥ 4 остановок, |r| > 0.35 % и она заметно лучше гипотез с r ≈ 0.
        """
        if self._rs_init is None or self.dir < 0 or self.mode != 'MAP':
            return False
        pid0, s0, D0 = self._rs_init
        A = self.D - D0
        if A < 200.0:
            return False
        pid, s_raw, off = self.map.advance(pid0, s0, A)
        if off or pid in self._rec_pids:
            return False   # в петлях у конечных остановки в очереди и неточная геометрия — не используем
        lms = self.landmarks.get(pid)
        if not lms:
            return False
        win = max(15.0, 0.035 * A)
        cands = [L[0] - s_raw for L in lms if abs(L[0] - s_raw) <= win]
        if not cands:
            return False
        if self._rs_stops and A - self._rs_stops[-1][0] < 25.0:
            return False   # подтягивание к следующему месту той же платформы — не новое свидетельство
        self._rs_stops.append((A, cands))
        self._rs_stops = self._rs_stops[-12:]
        S = self._rs_stops
        tol = self.rs_tol
        best = (0, 0.0, 0.0, 0.0)
        best0 = 0
        for i in range(len(S)):
            Ai, ci = S[i]
            for j in range(i + 1, len(S)):
                Aj, cj = S[j]
                if Aj - Ai < 150.0:
                    continue
                for di in ci:
                    for dj in cj:
                        r = (dj - di) / (Aj - Ai)
                        if abs(r) > 0.03:
                            continue
                        d0 = di - r * Ai
                        if abs(d0) > 15.0:
                            continue
                        n = 0
                        a_min = a_max = None
                        for Ak, ck in S:
                            e = d0 + r * Ak
                            for dk in ck:
                                if abs(dk - e) < tol:
                                    n += 1
                                    a_min = Ak if a_min is None else a_min
                                    a_max = Ak
                                    break
                        if n > best[0] or (n == best[0] and abs(r) < abs(best[1])):
                            best = (n, r, d0, (a_max - a_min) if n else 0.0)
                        if abs(r) < 0.002 and n > best0:
                            best0 = n
        n, r, d0, span = best
        enough = (n >= self.rs_min_inliers and span >= self.rs_min_span) or \
            (n >= self.rs_min_inliers - 1 and span >= self.rs_long_span)
        if not enough or abs(r) <= self.rs_min_rate or n < best0 + 2:
            return False
        lo = self.scale0 * (1.0 - self.scale_max_dev)
        hi = self.scale0 * (1.0 + self.scale_max_dev)
        self.scale = min(max(self.scale * (1.0 + r), lo), hi)
        pid_t, s_t, _ = self.map.advance(pid0, s0, A * (1.0 + r) + d0)
        self.pid, self.s_ref, self.D_ref = pid_t, s_t, self.D
        self.var_along = 0.5
        self.D_var = self.D
        self._prev_stop = None
        self.last_lm = None
        self._lm_hist = []
        self._rates = []
        self._suspect_rate = 0.0
        self._rs_init = (pid_t, s_t, self.D)
        self._rs_stops = []
        self.n_scale_fix += 1
        return True

    def _scale_anomaly(self, pid, d):
        """Детектор грубой ошибки масштаба по поправкам у ориентиров (см. __init__)."""
        main = self._path_q[pid] == 0.0 if self._path_q else True
        prev = self._lm_hist[-1] if self._lm_hist else None
        self._lm_hist.append((self.D, main, pid))
        self._lm_hist = self._lm_hist[-2:]
        slipped = self._slip_sec > self.scale_detect_max_slip
        self._slip_sec = 0.0
        if not main or prev is None or not prev[1] or slipped:
            # на интервале было проскальзывание — уход пути объясняется им, а не масштабом колёс
            self._rates = []
            self._suspect_rate = 0.0 if slipped else self._suspect_rate
            return
        dD = self.D - prev[0]
        if dD < self.scale_detect_min_dD:
            return
        r = d / dD
        # подозрение на грубую ошибку масштаба: неопределённость пути растёт быстрее, чтобы
        # следующий ориентир попал в строб и подтвердил (или опроверг) подозрение
        self._suspect_rate = abs(r) if abs(r) > self.scale_detect_thr else 0.0
        self._rates.append(r)
        self._rates = self._rates[-2:]
        if len(self._rates) == 2:
            r1, r2 = self._rates
            if r1 * r2 > 0.0 and min(abs(r1), abs(r2)) > self.scale_detect_thr:
                corr = 0.5 * (r1 + r2)
                lo = self.scale0 * (1.0 - self.scale_max_dev)
                hi = self.scale0 * (1.0 + self.scale_max_dev)
                self.scale = min(max(self.scale * (1.0 + corr), lo), hi)
                self.n_scale_fix += 1
                self._rates = []
                self._suspect_rate = 0.0

    # ------------------------------------------------------------------ GNSS (только старт)
    def on_gnss_fix(self, t, antenna, lat, lon, alt, status=0, covariance=None):
        if self.gnss_closed or antenna not in self.ant:
            return
        try:
            st = int(status)
        except (TypeError, ValueError):
            st = 0
        if st < 0:
            return
        p = self.proj.to_map(lat, lon, alt)
        if p is None:
            return
        self.last_fix[antenna] = (t, p[0], p[1], p[2], st)
        self.n_fix[antenna] += 1
        w = self.ant_w.get(antenna, 0.3) * self.status_w.get(st, 0.05)
        self._anchor_from_fix(antenna, t, w)

    def on_gnss_vel(self, t, antenna, vx, vy, vz):
        return  # базовый оценщик скорость GNSS не использует

    def on_gnss_correction(self, t, antenna, lat, lon, alt, status=0, covariance=None):
        """Редкие фиксы в середине маршрута (ТОЛЬКО при gnss_midroute_corrections=true).

        Сдвигает продольное положение к RTK-фиксу антенны master (коэффициент 0.3 на фикс —
        за ~1-секундную пачку из 10 фиксов поправка сходится) и подстраивает масштаб колёс
        по накопленной ошибке между пачками.
        """
        try:
            st_ = int(status)
        except (TypeError, ValueError):
            return
        if st_ != 2 or antenna != 'master' or self.mode != 'MAP' or self.pid is None:
            return
        p = self.proj.to_map(lat, lon, alt)
        if p is None:
            return
        off = self.ant[antenna]
        D_t = self._D_at(t)
        s_est = self.s_ref + self.dir * (D_t - self.D_ref)
        pid_e, s_e, _ = self.map.advance(self.pid, s_est, 0.0)
        pid_a, s_a, _ = self.map.advance(pid_e, s_e, self.dir * off[0])
        xa, ya, _, yaw, _ = self.map.pose(pid_a, s_a)
        if self.dir < 0:
            yaw += math.pi
        dx, dy = p[0] - xa, p[1] - ya
        along = dx * math.cos(yaw) + dy * math.sin(yaw)
        cross = -dx * math.sin(yaw) + dy * math.cos(yaw)
        if abs(cross) > 3.0 or abs(along) > 60.0:
            # фикс не на нашем пути (другая ветка у конечной?) — перепривязка по карте
            best = self._match(p[0], p[1], yaw, 3.0)
            if best is not None:
                self._reanchor_votes.append((t, best))
                self._reanchor_votes = [v for v in self._reanchor_votes if t - v[0] < 3.0]
                if len(self._reanchor_votes) >= 3:
                    _, pid, s_ant, dr = best
                    pid2, s_bl, _ = self.map.advance(pid, s_ant, -dr * off[0])
                    self.pid, self.dir = pid2, dr
                    self.s_ref = s_bl - dr * D_t
                    self.D_ref = 0.0
                    self._reanchor_votes = []
                    self.n_corr += 1
            return
        # масштаб колёс по пачкам GNSS не подстраиваем: между пачками путь уже поправлен
        # ориентирами-остановками, и «ошибка/путь» перестаёт быть ошибкой масштаба (грубую
        # ошибку масштаба ловит детектор по ориентирам)
        self._last_corr = (t, D_t)
        self.s_ref += self.dir * 0.3 * along
        self.n_corr += 1

    def on_gnss_window_closed(self, t):
        self.gnss_closed = True
        # неопределённость продольной координаты после GNSS-старта (RTK master ~0.2 м)
        good = sum(1 for a in self.anchors if a[2] >= 0.5)   # фиксы master со status 0/2
        fair = sum(1 for a in self.anchors if a[2] >= 0.1)   # + rover status 0/2
        if good >= 5:
            self.var_along = 0.05
        elif fair >= 5:
            self.var_along = 1.0
        else:
            self.var_along = 64.0   # только SBAS/единичные фиксы: первая остановка у ориентира поправит
        self.P_de = 0.0
        self.P_ee = self.scale_sigma0 ** 2 if self.scale_learn else 0.0
        self.D_var = self.D
        self._D_init = self.D
        if self.mode == 'MAP' and self.pid is not None and self.dir > 0:
            pid, s, _ = self.map.advance(self.pid, self.s_ref + self.dir * (self.D - self.D_ref), 0.0)
            self._rs_init = (pid, s, self.D)
            self._rs_stops = []

    def _heading_from_antennas(self):
        m = self.last_fix.get('master')
        r = self.last_fix.get('rover')
        if not m or not r or abs(m[0] - r[0]) > 0.3:
            return None
        base = self.ant['rover'][0] - self.ant['master'][0]
        dx, dy = r[1] - m[1], r[2] - m[2]
        d = math.hypot(dx, dy)
        if base <= 0 or abs(d - base) > 3.0:
            return None
        return math.atan2(dy, dx)

    def _D_at(self, t):
        h = self.hist.at(t)
        if h is not None:
            return h[0]
        if self.t is not None and t > self.t:
            return self.D + self.v * (t - self.t)
        return self.D

    def _match(self, x, y, yaw_body, max_dist):
        """Лучший путь для точки (x, y): (dist, pid, s, dir) или None.

        dir = +1 — трамвай едет по направлению пути, -1 — против (например, стоит на
        двухстороннем пути или на встречном пути разворотного кольца). Попутные пути
        предпочтительнее: встречный берётся, только если он заметно ближе (> 2 м).
        """
        cands = self.map.project(x, y, max_dist=max_dist) if not self.map.empty else []
        fwd = rev = None
        for d, pid, s_, yaw_path, _side in cands:
            if yaw_body is None:
                dh = 0.0
            else:
                dh = abs(_wrap(yaw_path - yaw_body))
            if dh <= math.radians(60):
                if fwd is None or d < fwd[0]:
                    fwd = (d, pid, s_, 1)
            elif dh >= math.radians(120):
                if rev is None or d < rev[0]:
                    rev = (d, pid, s_, -1)
        if fwd is not None and (rev is None or fwd[0] <= rev[0] + 2.0):
            return fwd
        return rev

    def _anchor_from_fix(self, antenna, t, w):
        fx = self.last_fix[antenna]
        off = self.ant[antenna]
        yaw_ant = self._heading_from_antennas()
        D_fix = self._D_at(t)
        best = self._match(fx[1], fx[2], yaw_ant, self.map_match_max)
        if best is not None:
            _, pid, s_ant, dr = best
            # base_link на -off[0] м впереди антенны по ходу трамвая
            pid2, s_bl2, _ = self.map.advance(pid, s_ant, -dr * off[0])
            self.anchors.append((pid2, s_bl2 - dr * D_fix, w, t, dr))
            if len(self.anchors) > 400:
                del self.anchors[:-400]
            self._resolve_on_map()
        else:
            yaw = yaw_ant
            if yaw is None:
                cands = self.map.project(fx[1], fx[2], max_dist=200.0)
                yaw = cands[0][3] if cands else (self.xy_ref[3] if self.xy_ref else 0.0)
            c, s_ = math.cos(yaw), math.sin(yaw)
            x = fx[1] - off[0] * c + off[1] * s_
            y = fx[2] - off[0] * s_ - off[1] * c
            z = fx[3] - off[2]
            self.xy_anchors.append((x, y, z, yaw, D_fix, w))
            if len(self.xy_anchors) > 400:
                del self.xy_anchors[:-400]
            if not self.anchors:
                self._resolve_off_map()

    @staticmethod
    def _wmedian(vals):
        vals = sorted(vals)
        tot = sum(w for _, w in vals)
        acc = 0.0
        for v, w in vals:
            acc += w
            if acc >= 0.5 * tot:
                return v
        return vals[-1][0]

    def _resolve_on_map(self):
        pid = self.anchors[-1][0]
        dr = self.anchors[-1][4]
        same = [(a[1], a[2]) for a in self.anchors if a[0] == pid and a[4] == dr]
        self.pid = pid
        self.dir = dr
        self.s_ref = self._wmedian(same)
        self.D_ref = 0.0
        self.mode = 'MAP'

    def _resolve_off_map(self):
        a = self.xy_anchors
        last = a[-1]
        recent = [q for q in a if abs(q[4] - last[4]) < 0.5]
        xs = self._wmedian([(q[0], q[5]) for q in recent])
        ys = self._wmedian([(q[1], q[5]) for q in recent])
        zs = self._wmedian([(q[2], q[5]) for q in recent])
        self.xy_ref = (xs, ys, zs, last[3])
        self.D_ref = last[4]
        self.mode = 'OFF_MAP'

    def _try_capture(self):
        """Вне карты: как только счисление подошло к пути с подходящим направлением — на карту."""
        if self.mode != 'OFF_MAP' or self.xy_ref is None:
            return
        if abs(self.D - self._capture_D) < 2.0:
            return
        self._capture_D = self.D
        x0, y0, z0, yaw = self.xy_ref
        dD = self.D - self.D_ref
        x, y = x0 + dD * math.cos(yaw), y0 + dD * math.sin(yaw)
        best = self._match(x, y, yaw, self.capture_dist)
        if best is None:
            return
        _, pid, s_, dr = best
        self.pid, self.dir = pid, dr
        self.s_ref = s_ - dr * self.D
        self.D_ref = 0.0
        self.anchors = []
        self.mode = 'MAP'
        self.captured = True

    # ------------------------------------------------------------------ выход
    def _body_pose(self, pid, s, dr=1):
        """Положение base_link и курс кузова (хорда задняя -> передняя тележка)."""
        x, y, z, yaw_t, slope = self.map.pose(pid, s)
        if dr < 0:
            yaw_t = _wrap(yaw_t + math.pi)
        L = self.bogie_dist
        if L > 0:
            pr, sr, _ = self.map.advance(pid, s, -dr * L)
            xr, yr, _, _, _ = self.map.pose(pr, sr)
            if math.hypot(x - xr, y - yr) > 0.5 * L:
                return x, y, z, math.atan2(y - yr, x - xr), slope
        return x, y, z, yaw_t, slope

    def _blend_branch(self, st, pid, s):
        """После стрелки на ветку, которой нет в графе: взвешенное положение «основной путь / ветка»."""
        for br in self.branches:
            if pid != br['main'] or s <= br['s_sw']:
                continue
            ds = s - br['s_sw']
            if ds > br['max_dist']:
                continue            # проехали дальше конца ветки — трамвай на основном пути
            xa, ya, za, _, _ = br['alt'].pose(0, min(ds, br['len']))
            p = br['p']
            if self.branch_stand_tau > 0.0 and self.still_since is not None and self.t is not None:
                p *= math.exp(-max(0.0, self.t - self.still_since) / self.branch_stand_tau)
            d2 = (st.x - xa) ** 2 + (st.y - ya) ** 2
            st.x += p * (xa - st.x)
            st.y += p * (ya - st.y)
            st.z += p * (za - st.z)
            st.var_position += p * (1.0 - p) * d2
            return br['name'], p
        return None

    def get_state(self, t):
        st = EstimatorState()
        st.t = t
        te = t - self.delay
        h = self.hist.at(te) if self.t is not None else None
        if h is not None:
            D, v, a = h
        elif self.t is not None:
            dt = min(max(te - self.t, 0.0), self.extrap_max)
            if dt > 0.0:
                v, D, a = self._propagate(self.v, self.D, self.t, self.t + dt)
            else:
                D, v, a = self.D, self.v, self.a
        else:
            D, v, a = 0.0, 0.0, 0.0
        if self.vel_tau > 0.0 and self.v_out is not None and h is None:
            v = self.v_out
        if self.vel_delay > 0.0 and self.t is not None:
            hv = self.hist.at(te - self.vel_delay)
            if hv is not None:
                v = hv[1]
        st.velocity = v
        st.acceleration = a
        st.velocity_valid = self.vel_valid
        st.distance = D
        st.slip = bool(self.slip)
        st.slip_ratio = float(self.slip_ratio)
        mu = self.mu if math.isfinite(self.mu) else self.mu_nominal
        if self.mu_t is not None and self.t is not None:
            k = math.exp(-max(0.0, self.t - self.mu_t) / 60.0)
            mu = self.mu_nominal + (self.mu - self.mu_nominal) * k
        st.adhesion = mu
        st.mode = self.mode
        branch = None
        if self.mode == 'MAP' and self.pid is not None:
            s = self.s_ref + self.dir * (D - self.D_ref)
            pid, s, off_end = self.map.advance(self.pid, s, 0.0)
            x, y, z, yaw, slope = self._body_pose(pid, s, self.dir)
            if self.dir < 0:
                slope = -slope
            st.x, st.y, st.z, st.yaw, st.pitch = x, y, z, yaw, -slope
            st.track = self.map.paths[pid].name
            st.s = s
            st.position_valid = True
            st.var_position = max(0.04, self.var_along)
            if self.branch_blend and self.branches and self.dir > 0:
                branch = self._blend_branch(st, pid, s)
            if off_end:
                st.mode = 'MAP_END'
        elif self.mode == 'OFF_MAP' and self.xy_ref is not None:
            x0, y0, z0, yaw = self.xy_ref
            dD = D - self.D_ref
            st.x = x0 + dD * math.cos(yaw)
            st.y = y0 + dD * math.sin(yaw)
            st.z = z0
            st.yaw = yaw
            st.position_valid = True
            st.var_position = 1.0 + (0.05 * abs(dD)) ** 2
        st.var_velocity = 0.05 ** 2 if not self.slip else 0.5 ** 2
        st.extra = {'cmd': self.cmd, 'fixes_master': self.n_fix['master'],
                    'fixes_rover': self.n_fix['rover'], 'slip_front': self.slip_front,
                    'slip_rear': self.slip_rear, 'stop_landmark_corrections': self.n_lm,
                    'slip_events': self.n_slip_events + self.n_sa_events,
                    'slip_false_alarms_undone': self.n_sa_undo,
                    'model_scale_traction': round(self.k_mod[1], 3),
                    'model_scale_brake': round(self.k_mod[-1], 3),
                    'slip_residual_sigma': round(self._sa_sigma, 3),
                    'wheel_scale_correction_pct': round(100.0 * (self.scale / self.scale0 - 1.0), 3),
                    'wheel_scale_fixes': self.n_scale_fix,
                    'bogie_ratio_front_rear': round(math.exp(self.bogie_rho), 5)}
        if branch is not None:
            st.extra['branch'], st.extra['branch_probability'] = branch
        return st
