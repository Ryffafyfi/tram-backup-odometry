"""Переходник: оценщик с интерфейсом on_notch / on_bogie / state -> интерфейс ноды.

Интерфейс::

    class Estimator:
        __init__(params, track_map)
        on_notch(t, position)                  # позиция ручки, -15..+15
        on_bogie(t, which, velocity)           # which = "front" | "rear", скорость в м/с
        on_gnss_fix(t, antenna, lat, lon, alt) # только первые секунды
        on_gnss_vel(t, antenna, vx, vy)        # только первые секунды
        state(t)                               # dict: скорость, s, x, y, z, курс, погрешности, флаги

Нода вызывает свой контракт (estimator_api.py): on_front_velocity / on_rear_velocity (сырые км/ч),
on_driver_cmd, get_state, время — секунды от начала записи. Переходник приводит одно к другому:

* скорость тележек: сырое значение × ``agreed_api_velocity_scale`` (по умолчанию 1/3.6 — как в
  CSV выгрузки tools/export_runs.py, где скорость уже в м/с);
* время: абсолютные секунды из ``header.stamp`` (как столбец ``t`` в CSV выгрузки) — нода
  сообщает начало сессии через ``set_time_origin``;
* ``params`` — параметры ноды (геометрия, перевод GNSS, пути к картам) + содержимое
  ``config/params.yaml`` (коэффициенты модели поверх);
* ``track_map`` — ``{направление: TrackLine}`` из ``maps/track_*.csv`` (только
  numpy), см. track_csv.py; ``agreed_api_track_map: graph`` — граф путей ноды (JSON + петли);
* ``state(t)`` -> EstimatorState: понимает распространённые имена ключей (v/speed/velocity,
  heading/yaw/course, sigma_*/std_*/var_* ...). Нет курса — берётся по направлению движения;
  нет z — из карты.

Сигнатуры on_gnss_fix/on_gnss_vel проверяются: status/covariance/vz передаются, только если метод
их принимает. Если у оценщика нет reset(), при новой сессии он создаётся заново.

Конструктор может принимать третий аргумент — перевод GNSS в координаты карты (``to_utm`` / ``to_map``
-> x, y, z карты), как в team_estimator.py. У ``track_map`` есть и интерфейс «одной
текущей линии»: ``select_direction``, ``project_to_path``, ``get_pose_at``, ``get_special_points``.

Проверка правдоподобия ответа: положение вне габаритов карты (±200 м) — например, нули до
GNSS-старта — не публикуется (берётся у резервного оценщика); z вне диапазона высот карты
заменяется высотой карты; скорость вне 0..max_speed не публикуется.
"""

import inspect
import math

from .estimator_api import EstimatorState

NATIVE_METHODS = ('on_front_velocity', 'on_rear_velocity', 'on_driver_cmd')
AGREED_METHODS = ('on_bogie', 'on_notch')


def _has(obj, name):
    return callable(getattr(obj, name, None))


def detect_api(cls_or_obj):
    """'native' — контракт ноды, 'agreed' — интерфейс on_notch/on_bogie/state, None — ни то ни другое."""
    if _has(cls_or_obj, 'get_state') and any(_has(cls_or_obj, m) for m in NATIVE_METHODS):
        return 'native'
    if _has(cls_or_obj, 'state') and any(_has(cls_or_obj, m) for m in AGREED_METHODS):
        return 'agreed'
    return None


def make_estimator(cls, config, log=None):
    """Создать оценщик: свой контракт — напрямую, интерфейс on_notch/on_bogie/state — через переходник."""
    api = detect_api(cls)
    name = getattr(cls, '__name__', str(cls))
    if api == 'native':
        return cls(config)
    if api == 'agreed':
        return AgreedApiAdapter(cls, config, log)
    raise TypeError(
        'estimator %s: unknown interface. Need either the node contract (get_state + '
        'on_front_velocity/on_rear_velocity/on_driver_cmd, see README) or the '
        'on_notch/on_bogie/state interface. Falling back to the baseline estimator.' % name)


def _positional_capacity(fn):
    """Сколько позиционных аргументов принимает fn (без self); None — сколько угодно (*args)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    n = 0
    for p in sig.parameters.values():
        if p.kind == p.VAR_POSITIONAL:
            return None
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
    return n


def _call_flex(fn, required, optional):
    """Вызвать fn с обязательными аргументами и теми необязательными, что он принимает."""
    cap = _positional_capacity(fn)
    args = list(required) + list(optional)
    if cap is not None:
        args = args[:max(cap, len(required))]
    return fn(*args)


def _scalar(v):
    """numpy-скаляр или массив из одного элемента (например, из track.at(s)) -> число."""
    try:
        if hasattr(v, '__len__') and not isinstance(v, (str, bytes, dict)):
            if len(v) != 1:
                return v
            v = v[0]
        if hasattr(v, 'item'):
            v = v.item()
    except Exception:  # noqa: BLE001
        pass
    return v


def _finite(*vals):
    try:
        return all(v is not None and math.isfinite(float(_scalar(v))) for v in vals)
    except (TypeError, ValueError):
        return False


# ключи state(t) -> поле EstimatorState (первый найденный)
_KEYS = {
    'velocity': ('velocity', 'v', 'speed', 'vel', 'v_mps', 'speed_mps'),
    'acceleration': ('acceleration', 'accel', 'a'),
    'x': ('x',), 'y': ('y',), 'z': ('z',),
    'yaw': ('yaw', 'heading', 'course', 'psi', 'kurs'),
    'pitch': ('pitch',),
    's': ('s', 's_m', 'along', 'distance_along', 'arc'),
    'distance': ('distance', 'odometer', 'dist', 'path'),
    'adhesion': ('adhesion', 'mu', 'adhesion_estimate'),
    'slip_ratio': ('slip_ratio',),
    'mode': ('mode', 'status'),
    'track': ('track', 'direction', 'direction_key', 'line'),
}
_VAR_KEYS = {
    'var_position': (('var_position', 'var_pos', 'pos_var', 'var_s', 'var_xy', 'p_s', 'cov_s',
                      'cov_pos'),
                     ('std_position', 'sigma_position', 'pos_std', 'std_pos', 'sigma_pos',
                      'std_s', 'sigma_s')),
    'var_velocity': (('var_velocity', 'var_v', 'vel_var', 'p_v', 'cov_v', 'cov_vel'),
                     ('std_velocity', 'sigma_velocity', 'vel_std', 'std_v', 'sigma_v')),
}
_SLIP_KEYS = ('slip', 'slip_detected', 'slip_flag', 'slip_active', 'wheel_slip', 'skid', 'slide',
              'boxing')
_POS_VALID_KEYS = ('position_valid', 'pos_valid', 'valid', 'initialized', 'localized')
_VEL_VALID_KEYS = ('velocity_valid', 'v_valid', 'vel_valid')


def _as_dict(raw):
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, EstimatorState):
        return {k: getattr(raw, k) for k in EstimatorState.__slots__}
    d = getattr(raw, '__dict__', None)
    if isinstance(d, dict):
        return d
    if hasattr(raw, '_asdict'):  # namedtuple
        return dict(raw._asdict())
    return None


class GnssToMap:
    """GNSS (широта, долгота, высота) -> x, y, z в системе карты (UTM 37N минус 300000/6100000).

    ``to_utm`` — имя метода, которое ждёт team_estimator.py; возвращает сразу координаты карты (в них и карта).
    """

    def __init__(self, config):
        from .geo import make_projection
        self._proj = make_projection(config)

    def to_map(self, lat, lon, alt=0.0):
        r = self._proj.to_map(lat, lon, alt)
        if r is None:
            return math.nan, math.nan, math.nan
        return float(r[0]), float(r[1]), float(r[2])

    to_utm = to_map


class AgreedApiAdapter:
    """Оценщик с интерфейсом on_notch / on_bogie / state, обёрнутый в интерфейс ноды."""

    def __init__(self, cls, config, log=None):
        self._cls = cls
        self._cfg = dict(config or {})
        self._log = log or (lambda level, msg: None)
        self.name = getattr(cls, '__name__', 'Estimator')
        self.v_scale = float(self._cfg.get('agreed_api_velocity_scale', 1.0 / 3.6))
        self.t0 = 0.0
        self.params = self._build_params()
        self.track_map = self._build_track_map()
        self.gnss = GnssToMap(self._cfg)
        self.v_max = float(self._cfg.get('max_speed', 30.0) or 30.0)
        self._box = None
        if self.track_map is not None and hasattr(self.track_map, 'bounds'):
            try:
                self._box = self.track_map.bounds()
            except Exception:  # noqa: BLE001
                self._box = None
        self._warned = set()
        self._last_xy = None
        self._last_yaw = 0.0
        self.obj = None
        self._create()
        self._log('info', 'estimator %s: team contract (on_notch/on_bogie/state) via adapter: '
                          'bogie velocity x%.6f (km/h -> m/s), t = header.stamp seconds, '
                          'track_map = %s' % (self.name, self.v_scale,
                                              type(self.track_map).__name__))

    # ------------------------------------------------------------------ создание
    def _build_params(self):
        params = {k: v for k, v in self._cfg.items() if k not in ('coefficients',
                                                                  'baseline_coefficients')}
        coef = self._cfg.get('coefficients')
        if isinstance(coef, dict):
            params.update(coef)  # коэффициенты физической модели важнее параметров ноды
        params.setdefault('velocity_units', 'm/s')
        params.setdefault('time_units', 's (header.stamp)')
        return params

    def _build_track_map(self):
        kind = str(self._cfg.get('agreed_api_track_map', 'csv') or 'csv').lower()
        if kind in ('none', 'off', ''):
            return None
        if kind == 'graph':
            try:
                from .track_map import TrackMap
                return TrackMap(self._cfg.get('map_files') or [])
            except Exception as e:  # noqa: BLE001
                self._log('error', 'track_map (graph) load failed: %r' % (e,))
                return None
        from .track_csv import load_track_map
        return load_track_map(self._cfg.get('track_csv_files') or [],
                              self._cfg.get('map_files') or [], self._log,
                              self._cfg.get('stop_landmarks_file') or None)

    def _create(self):
        if self._cls.__init__ is object.__init__:
            cap = 0  # у класса нет своего __init__
        else:
            cap = _positional_capacity(self._cls.__init__)
            if cap is not None:
                cap -= 1  # self
        args = (self.params, self.track_map, self.gnss)
        n = 2 if cap is None else min(cap, len(args))
        self.obj = self._cls(*args[:n])
        self._last_xy = None
        self._last_yaw = 0.0

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            self._log('warn', 'estimator %s: %s' % (self.name, msg))

    # ------------------------------------------------------------------ время и сессии
    def set_time_origin(self, t0_sec):
        self.t0 = float(t0_sec)

    def _abs(self, t):
        return self.t0 + float(t)

    def reset(self):
        fn = getattr(self.obj, 'reset', None)
        if callable(fn):
            fn()
            self._last_xy = None
        else:
            self._create()

    # ------------------------------------------------------------------ входы
    def on_front_velocity(self, t, v_raw):
        fn = getattr(self.obj, 'on_bogie', None)
        if fn is not None:
            fn(self._abs(t), 'front', float(v_raw) * self.v_scale)

    def on_rear_velocity(self, t, v_raw):
        fn = getattr(self.obj, 'on_bogie', None)
        if fn is not None:
            fn(self._abs(t), 'rear', float(v_raw) * self.v_scale)

    def on_driver_cmd(self, t, position):
        fn = getattr(self.obj, 'on_notch', None)
        if fn is not None:
            fn(self._abs(t), int(position))

    def on_gnss_fix(self, t, antenna, lat, lon, alt, status=0, covariance=None):
        fn = getattr(self.obj, 'on_gnss_fix', None)
        if fn is not None:
            _call_flex(fn, (self._abs(t), antenna, lat, lon, alt),
                       (int(status), list(covariance or [])))

    def on_gnss_vel(self, t, antenna, vx, vy, vz=0.0):
        fn = getattr(self.obj, 'on_gnss_vel', None)
        if fn is not None:
            _call_flex(fn, (self._abs(t), antenna, vx, vy), (vz,))

    def on_gnss_window_closed(self, t):
        fn = getattr(self.obj, 'on_gnss_window_closed', None)
        if fn is not None:
            _call_flex(fn, (), (self._abs(t),))

    # ------------------------------------------------------------------ выход
    def get_state(self, t):
        fn = getattr(self.obj, 'state', None)
        if not callable(fn):
            # в объекте атрибут self.state (данные) закрыл метод state(t) — вызываем метод класса
            cls_fn = getattr(type(self.obj), 'state', None)
            if not callable(cls_fn):
                return None
            self._warn_once('state_attr', 'instance attribute "state" shadows method state(t) -> '
                                          'calling the class method (better rename the attribute)')
            raw = cls_fn(self.obj, self._abs(t))
        else:
            raw = fn(self._abs(t))
        d = _as_dict(raw)
        if d is None:
            if raw is not None:
                self._warn_once('type', 'state() returned %s, expected dict' % type(raw).__name__)
            return None
        return self._convert(d)

    def _convert(self, d):
        low = {str(k).lower(): v for k, v in d.items()}
        flags = low.get('flags') if isinstance(low.get('flags'), dict) else {}
        low_flags = {str(k).lower(): v for k, v in flags.items()}
        st = EstimatorState()
        used = set()
        given = set()

        def pick(names, source=low):
            for n in names:
                if n in source and source[n] is not None:
                    used.add(n)
                    return source[n]
            return None

        for field, names in _KEYS.items():
            v = pick(names)
            if v is not None:
                setattr(st, field, _scalar(v))
                given.add(field)
        # позиция одним полем: position / xyz = (x, y, z)
        if not ({'x', 'y'} <= given):
            pos = pick(('position', 'xyz', 'pos'))
            if hasattr(pos, '__len__') and not isinstance(pos, (str, dict)) and len(pos) >= 2:
                st.x, st.y = _scalar(pos[0]), _scalar(pos[1])
                given.update(('x', 'y'))
                if len(pos) >= 3:
                    st.z = _scalar(pos[2])
                    given.add('z')
        for field, (var_names, std_names) in _VAR_KEYS.items():
            v = pick(var_names)
            if _finite(v):
                setattr(st, field, float(_scalar(v)))
            else:
                s = pick(std_names)
                if _finite(s):
                    setattr(st, field, float(_scalar(s)) ** 2)
        slip = pick(_SLIP_KEYS)
        if slip is None:
            slip = pick(_SLIP_KEYS, low_flags)
        st.slip = bool(slip) if slip is not None else False
        # скорость: валидна, если дана и конечна (или явный флаг)
        vv = pick(_VEL_VALID_KEYS)
        has_v = 'velocity' in given and _finite(st.velocity)
        st.velocity_valid = (bool(vv) and has_v) if vv is not None else has_v
        if not has_v:
            st.velocity = 0.0
        # положение: нет z — из карты, нет курса — по направлению движения
        if _finite(st.x, st.y) and not ('z' in given and _finite(st.z)):
            z = self._z_from_map(st.x, st.y)
            if z is not None:
                st.z = z
                self._warn_once('z', 'state() has no z -> z taken from the track map')
        xy_ok = _finite(st.x, st.y) and self._plausible_xy(st.x, st.y)
        # z вне диапазона высот карты (например, 0) — высота из карты
        if xy_ok and self._box is not None and _finite(st.z) and _finite(self._box[4]):
            if not (self._box[4] - 15.0 <= float(st.z) <= self._box[5] + 15.0):
                z = self._z_from_map(st.x, st.y)
                if z is not None:
                    self._warn_once('zrange', 'state() z=%.1f is outside map heights -> z taken '
                                              'from the track map' % float(st.z))
                    st.z = z
        pv = pick(_POS_VALID_KEYS)
        has_pos = xy_ok and _finite(st.z)
        st.position_valid = (bool(pv) and has_pos) if pv is not None else has_pos
        if st.velocity_valid and not (-1.0 <= float(st.velocity) <= self.v_max):
            self._warn_once('vrange', 'state() velocity %.2f m/s is implausible -> not published'
                            % float(st.velocity))
            st.velocity_valid = False
        if st.position_valid:
            xy = (float(st.x), float(st.y))
            if 'yaw' in given and _finite(st.yaw):
                self._last_yaw = float(st.yaw)
                self._last_xy = xy
            else:
                self._warn_once('yaw', 'state() has no heading -> heading from motion')
                if self._last_xy is None:
                    self._last_xy = xy
                else:
                    dx, dy = xy[0] - self._last_xy[0], xy[1] - self._last_xy[1]
                    if dx * dx + dy * dy > 0.09:
                        self._last_yaw = math.atan2(dy, dx)
                        self._last_xy = xy
                st.yaw = self._last_yaw
        if not _finite(st.pitch):
            st.pitch = 0.0
        st.mode = str(st.mode) if 'mode' in given else 'TEAM'
        st.track = str(st.track) if 'track' in given else ''
        extra = {}
        for k, v in low.items():
            if k in used or k in ('flags',) or k in _POS_VALID_KEYS or k in _VEL_VALID_KEYS:
                continue
            if isinstance(v, (int, float, str, bool)):
                extra[k] = v
        for k, v in low_flags.items():
            if isinstance(v, (int, float, str, bool)):
                extra['flag_%s' % k] = v
        if 'yaw_rate' in low and _finite(low['yaw_rate']):
            extra['yaw_rate'] = float(low['yaw_rate'])
        st.extra = extra
        return st

    def _plausible_xy(self, x, y):
        """Точка в габаритах карты (±200 м); без карты — хотя бы не (0, 0)."""
        x, y = float(x), float(y)
        if self._box is None:
            ok = abs(x) > 1.0 or abs(y) > 1.0
        else:
            ok = (self._box[0] - 200.0 <= x <= self._box[1] + 200.0 and
                  self._box[2] - 200.0 <= y <= self._box[3] + 200.0)
        if not ok:
            self._warn_once('xy', 'state() position (%.1f, %.1f) is outside the map -> not '
                                  'published (e.g. zeros before GNSS start)' % (x, y))
        return ok

    def _z_from_map(self, x, y):
        tm = self.track_map
        if not tm or not hasattr(tm, 'nearest'):
            return None
        try:
            k, s, _c = tm.nearest(float(x), float(y))
            z = float(tm[k].at(s)[2][0])
            return z if math.isfinite(z) else None
        except Exception:  # noqa: BLE001
            return None
