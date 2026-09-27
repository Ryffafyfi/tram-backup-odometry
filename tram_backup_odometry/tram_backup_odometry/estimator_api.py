"""Контракт между ROS-нодой и оценщиком (класс Estimator).

Нода (node.py -> core.py) ничего не знает о математике: она вызывает методы оценщика
и публикует то, что вернул ``get_state(t)``. Любой класс с этими методами можно подключить
параметром ``estimator_class`` (формат ``модуль:Класс``) — без правки ноды.

Время ``t`` — float, секунды от начала сессии (первое сообщение записи), вычисленные
из ``header.stamp`` входного сообщения (не системные часы!). Сообщения разных топиков
могут приходить с перестановкой по времени на десятки мс — это нормально.

Обязательные методы::

    Estimator(config: dict)
        config — все параметры ноды из YAML (пути к картам уже абсолютные)
        + содержимое файла коэффициентов (config/params.yaml) в config['coefficients'].
        (Оценщик с интерфейсом on_notch/on_bogie/state — Estimator(params, track_map), on_notch, on_bogie, state —
        подключается автоматически через переходник agreed_api.py.)

    on_front_velocity(t: float, v_raw: float) -> None   # сырое значение из топика
    on_rear_velocity(t: float, v_raw: float) -> None    # (в записях это км/ч!)
    on_driver_cmd(t: float, position: int) -> None      # -15..+15

    get_state(t: float) -> EstimatorState | dict
        Оценка на момент t (t может быть чуть раньше/позже последнего входа).

Необязательные методы (нода вызывает, если они есть)::

    on_gnss_fix(t, antenna: str, lat, lon, alt, status: int, covariance: list) -> None
    on_gnss_vel(t, antenna: str, vx, vy, vz) -> None
        Вызываются ТОЛЬКО в стартовом окне (gnss_init_duration_sec). antenna: 'master'|'rover'.
    on_gnss_window_closed(t) -> None
        Окно GNSS закрыто, подписки удалены — дальше только колёса и контроллер.
    reset() -> None
        Новая сессия (обнаружен скачок времени назад: запись запущена заново).

Требования: каждый вызов < 1 мс в среднем; не бросать исключений на NaN/пустых данных
(нода всё равно защищена: при исключении или NaN она переключится на резервный оценщик).
"""

import math


class EstimatorState:
    """Выход оценщика. Все поля имеют безопасные значения по умолчанию."""

    __slots__ = (
        't', 'velocity', 'acceleration', 'x', 'y', 'z', 'yaw', 'pitch',
        'position_valid', 'velocity_valid', 'var_position', 'var_velocity',
        'slip', 'adhesion', 'slip_ratio', 'mode', 'track', 's', 'distance', 'extra',
    )

    def __init__(self, **kw):
        self.t = 0.0
        self.velocity = 0.0          # продольная скорость base_link, м/с
        self.acceleration = 0.0      # м/с^2
        self.x = math.nan            # положение base_link в системе карты, м
        self.y = math.nan
        self.z = math.nan
        self.yaw = 0.0               # курс, рад (0 — вдоль оси X карты, против часовой)
        self.pitch = 0.0             # тангаж, рад (положительный — нос вниз по ROS REP-103)
        self.position_valid = False  # есть ли абсолютная привязка (после GNSS-старта)
        self.velocity_valid = False
        self.var_position = 1e4      # дисперсия положения, м^2
        self.var_velocity = 1.0      # дисперсия скорости, (м/с)^2
        self.slip = False            # колёса проскальзывают / юз
        self.adhesion = math.nan     # оценка коэффициента сцепления (mu)
        self.slip_ratio = 0.0        # относительное проскальзывание (v_колеса - v)/v
        self.mode = 'INIT'
        self.track = ''
        self.s = 0.0
        self.distance = 0.0
        self.extra = {}
        for k, v in kw.items():
            setattr(self, k, v)

    @classmethod
    def from_any(cls, obj):
        """Привести dict/объект с похожими полями к EstimatorState."""
        if isinstance(obj, cls):
            return obj
        st = cls()
        if obj is None:
            return st
        get = obj.get if isinstance(obj, dict) else (lambda k, d=None: getattr(obj, k, d))
        for k in cls.__slots__:
            v = get(k, None)
            if v is not None:
                setattr(st, k, v)
        # распространённые синонимы
        for alias, k in (('v', 'velocity'), ('speed', 'velocity'), ('heading', 'yaw'),
                         ('valid', 'position_valid'), ('adhesion_estimate', 'adhesion'),
                         ('slip_detected', 'slip')):
            v = get(alias, None)
            if v is not None and get(k, None) is None:
                setattr(st, k, v)
        return st

    def is_finite(self):
        vals = [self.velocity]
        if self.position_valid:
            vals += [self.x, self.y, self.z, self.yaw]
        try:
            return all(math.isfinite(float(v)) for v in vals)
        except (TypeError, ValueError):
            return False
