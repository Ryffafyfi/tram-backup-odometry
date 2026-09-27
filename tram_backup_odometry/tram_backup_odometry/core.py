"""Логика ноды без ROS: проверка входов, окно GNSS, работа с оценщиком, прогноз в пропусках.

node.py — тонкая ROS-обёртка над OdometryCore. Такое разделение позволяет прогонять
ровно ту же логику офлайн по bag-файлам (tools/offline_eval.py) и в юнит-тестах.

Время:
  * stamp_ns — header.stamp входа в наносекундах (int, без потери точности);
  * t (float) — секунды от начала сессии, передаётся оценщику (оценщику с интерфейсом on_notch/on_bogie/state
    переходник agreed_api передаёт абсолютные секунды header.stamp: начало сессии сообщается
    через set_time_origin);
  * wall — монотонные часы процесса (только для прогноза в пропусках и замеров).
"""

import importlib
import math
import time
import traceback
from collections import deque

from .agreed_api import detect_api, make_estimator
from .estimator_api import EstimatorState

NS = 1_000_000_000


def load_class(spec):
    """'пакет.модуль:Класс' -> класс."""
    mod_name, _, cls_name = str(spec).partition(':')
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name or 'Estimator')


def quat_from_yaw_pitch(yaw, pitch):
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    return (-sy * sp, cy * sp, sy * cp, cy * cp)  # x, y, z, w (roll = 0)


class Output:
    """Готовый ответ для публикации (один на каждый принятый вход)."""

    __slots__ = ('stamp_ns', 'velocity', 'publish_velocity', 'publish_position',
                 'x', 'y', 'z', 'q', 'var_position', 'var_velocity', 'yaw_rate',
                 'state', 'source', 'fallback')

    def __init__(self):
        self.stamp_ns = 0
        self.velocity = 0.0
        self.publish_velocity = False
        self.publish_position = False
        self.x = self.y = self.z = 0.0
        self.q = (0.0, 0.0, 0.0, 1.0)
        self.var_position = 1e4
        self.var_velocity = 1.0
        self.yaw_rate = 0.0
        self.state = None
        self.source = 'input'
        self.fallback = False


class LatencyStats:
    """Скользящая статистика задержек (мс) без накопления памяти."""

    def __init__(self, keep=4000):
        self.samples = deque(maxlen=keep)
        self.count = 0
        self.max_all = 0.0
        self.sum_all = 0.0

    def add(self, ms):
        self.samples.append(ms)
        self.count += 1
        self.sum_all += ms
        if ms > self.max_all:
            self.max_all = ms

    def summary(self):
        if not self.samples:
            return {'n': self.count, 'mean': math.nan, 'p50': math.nan, 'p95': math.nan,
                    'p99': math.nan, 'max': math.nan, 'max_all': self.max_all}
        s = sorted(self.samples)
        n = len(s)

        def pct(p):
            return s[min(n - 1, int(p * (n - 1) + 0.5))]
        return {'n': self.count, 'mean': sum(s) / n, 'p50': pct(0.5), 'p95': pct(0.95),
                'p99': pct(0.99), 'max': s[-1], 'max_all': self.max_all}


class _SafeEstimator:
    """Обёртка: любые исключения оценщика перехватываются и считаются."""

    def __init__(self, obj, name, log):
        self.obj = obj
        self.name = name
        self.log = log
        self.errors = 0
        self.last_error_wall = -1e9
        self._last_logged = -1e9

    def call(self, method, *args):
        fn = getattr(self.obj, method, None)
        if fn is None:
            return None
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 — нода не должна падать ни при каком оценщике
            self.errors += 1
            now = time.monotonic()
            self.last_error_wall = now
            if now - self._last_logged > 5.0:
                self._last_logged = now
                self.log('error', '%s.%s failed (%d errors so far): %r\n%s' % (
                    self.name, method, self.errors, e, traceback.format_exc(limit=3)))
            return None

    def healthy(self, window=2.0):
        return time.monotonic() - self.last_error_wall > window


VEHICLE_TOPICS = ('front', 'rear', 'cmd')


class OdometryCore:
    def __init__(self, cfg, log=None):
        self.cfg = cfg
        self._log = log or (lambda level, msg: None)
        self.gnss_duration = float(cfg.get('gnss_init_duration_sec', 5.0))
        self.gnss_max_wait = max(self.gnss_duration, float(cfg.get('gnss_init_max_wait_sec', 20.0)))
        self.gnss_min_fixes = int(cfg.get('gnss_init_min_fixes', 10))
        # Выключено по умолчанию: GNSS используется только на старте. Опция для экспериментов —
        # редкие фиксы в середине маршрута как коррекция.
        self.midroute = bool(cfg.get('gnss_midroute_corrections', False))
        self.reset_back_ns = int(float(cfg.get('time_jump_reset_sec', 3.0)) * NS)
        self.jump_fwd_ns = int(float(cfg.get('time_jump_forward_sec', 30.0)) * NS)
        self.topic_jump_ns = int(float(cfg.get('topic_jump_sec', 0.6)) * NS)
        self.topic_ahead_tol_ns = int(float(cfg.get('topic_ahead_tolerance_sec', 0.3)) * NS)
        self.accept_escape_sec = float(cfg.get('time_glitch_escape_sec', 1.0))
        self.first_msg_tol_ns = int(float(cfg.get('first_message_tolerance_sec', 5.0)) * NS)
        self.gap_timeout = float(cfg.get('gap_timeout_sec', 0.07))
        self.gap_period = float(cfg.get('gap_publish_period_sec', 0.05))
        self.max_gap_fill = float(cfg.get('max_gap_fill_sec', 30.0))
        self.fixed_rate = float(cfg.get('playback_rate', 0.0))
        self.publish_before_init = bool(cfg.get('publish_position_before_init', False))

        main_spec = cfg.get('estimator_class', 'tram_backup_odometry.estimator:Estimator')
        fb_spec = cfg.get('fallback_estimator_class',
                          'tram_backup_odometry.baseline_estimator:BaselineEstimator')
        self._main_cls = None
        self._fb_cls = None
        try:
            self._main_cls = load_class(main_spec)
        except Exception as e:  # noqa: BLE001
            self._log('error', 'cannot load estimator %s: %r -> using fallback' % (main_spec, e))
        if cfg.get('fallback_on_error', True):
            try:
                self._fb_cls = load_class(fb_spec)
            except Exception as e:  # noqa: BLE001
                self._log('error', 'cannot load fallback estimator %s: %r' % (fb_spec, e))
        if self._main_cls is None:
            self._main_cls, self._fb_cls = self._fb_cls, None
        if self._fb_cls is not None and self._fb_cls is self._main_cls:
            self._fb_cls = None  # заглушка и так основная — второй экземпляр не нужен
        self.main_name = getattr(self._main_cls, '__name__', '?')
        if self._main_cls is not None and detect_api(self._main_cls) == 'agreed':
            self.main_name += ' [team API via adapter]'
        self.fb_name = getattr(self._fb_cls, '__name__', '-') if self._fb_cls else '-'

        # статистика за всё время работы
        self.counters = {}
        self.latency = LatencyStats()
        self.n_out_vel = 0
        self.n_out_pos = 0
        self.n_out_pred = 0
        self.n_out_fallback = 0
        self.sessions = 0
        self.gnss_events = []
        self._new_session_state()
        self.main = None
        self.fb = None
        self._make_estimators()

    # ------------------------------------------------------------------ служебное
    def _count(self, key, n=1):
        self.counters[key] = self.counters.get(key, 0) + n

    def _make_estimators(self):
        self.main = None
        self.fb = None
        try:
            self.main = _SafeEstimator(make_estimator(self._main_cls, self.cfg, self._log),
                                       self.main_name, self._log)
        except Exception as e:  # noqa: BLE001
            self._log('error', 'estimator %s init failed: %r\n%s' % (
                self.main_name, e, traceback.format_exc(limit=3)))
        if self._fb_cls is not None or self.main is None:
            cls = self._fb_cls or load_class(
                'tram_backup_odometry.baseline_estimator:BaselineEstimator')
            try:
                self.fb = _SafeEstimator(make_estimator(cls, self.cfg, self._log),
                                         getattr(cls, '__name__', 'fallback'), self._log)
            except Exception as e:  # noqa: BLE001
                self._log('error', 'fallback estimator init failed: %r' % (e,))
        if self.main is None and self.fb is not None:
            # основной не создан — работает резервный; имя в логах/диагностике — честное
            self._log('error', 'MAIN ESTIMATOR %s NOT RUNNING -> all answers from %s' % (
                self.main_name, self.fb.name))
            self.main_name = '%s (main %s failed)' % (self.fb.name, self.main_name)
            self.fb_name = '-'
            self.main, self.fb = self.fb, None

    def _new_session_state(self):
        self.epoch_ns = None
        self.t_latest_ns = None
        self.wall_at_latest = None
        self.last_input_wall = None
        self.last_output_wall = None
        self.last_stamp = {}
        self.pending_jump = []
        self.suspects = []
        self.last_accept_wall = None
        self.gnss_open = True
        self.gnss_closed_info = None
        self.rate_samples = deque(maxlen=400)
        self.last_pred_ns = None
        self.last_out_stamp_ns = None
        self.session_start_wall = None
        self.gnss_counts = {'master_fix': 0, 'rover_fix': 0, 'master_vel': 0, 'rover_vel': 0}

    def _reset_session(self, reason):
        self.sessions += 1
        self._log('warn', 'NEW SESSION (%s): estimator reset, GNSS start window reopened' % reason)
        self._new_session_state()
        for est in (self.main, self.fb):
            if est is None:
                continue
            if hasattr(est.obj, 'reset'):
                est.call('reset')
            else:
                try:
                    est.obj = type(est.obj)(self.cfg)
                except Exception as e:  # noqa: BLE001
                    self._log('error', 'estimator re-create failed: %r' % (e,))

    def rel(self, stamp_ns):
        return (stamp_ns - self.epoch_ns) / NS

    def _set_epoch(self, stamp_ns, wall):
        """Начало сессии (t = 0). Оценщикам, которым нужно абсолютное время, сообщается отдельно."""
        self.epoch_ns = stamp_ns
        self.session_start_wall = wall
        for est in (self.main, self.fb):
            if est is not None:
                est.call('set_time_origin', stamp_ns / NS)

    # ------------------------------------------------------------------ проверка времени
    def expected_now_ns(self, wall):
        """Оценка текущего времени записи по «стенным» часам (для проверки скачков и прогноза)."""
        if self.t_latest_ns is None or self.wall_at_latest is None:
            return None
        return self.t_latest_ns + int((wall - self.wall_at_latest) * self.playback_rate() * NS)

    def _consistent_with_clock(self, stamp_ns, wall):
        """stamp не «убегает» вперёд от времени записи, оценённого по стенным часам.

        Одиночные/чередующиеся сбойные stamp (в обучающих записях встречаются серии с +1 c)
        отбрасываются. Если же нормальных сообщений нет уже accept_escape_sec, а «убежавшие»
        stamp согласованы между собой — шкала времени действительно сдвинулась, принимаем.
        """
        exp = self.expected_now_ns(wall)
        if exp is None or stamp_ns - exp <= self.topic_ahead_tol_ns:
            return True
        self.suspects.append((stamp_ns, wall))
        if len(self.suspects) > 50:
            del self.suspects[:-50]
        if self.last_accept_wall is None or wall - self.last_accept_wall < self.accept_escape_sec:
            return False
        recent = [p for p in self.suspects if wall - p[1] <= self.accept_escape_sec + 0.5]
        if len(recent) < 3:
            return False
        ss = [p[0] for p in recent]
        return max(ss) - min(ss) <= int((self.accept_escape_sec + 1.0) * self.playback_rate() * NS)

    def _accept_stamp(self, topic, stamp_ns, wall):
        """True — вход принят; False — пропущен (посчитано в counters).

        Правила:
          * нулевой/отрицательный stamp — пропуск;
          * повтор stamp в топике — пропуск; stamp меньше предыдущего в топике — пропуск;
          * скачок вперёд внутри топика > topic_jump_sec, не подтверждённый остальными топиками
            (stamp «убежал» от текущего времени записи) — пропуск, пока следующее сообщение
            этого топика не подтвердит новую шкалу (так отсекаются одиночные сбойные stamp);
          * скачок всей шкалы назад > time_jump_reset_sec (или вперёд > time_jump_forward_sec),
            подтверждённый >= 3 сообщениями от >= 2 топиков, — новая сессия (запись запущена заново).
        """
        if not isinstance(stamp_ns, int) or stamp_ns <= 0:
            self._count('%s_bad_stamp' % topic)
            return False
        if self.t_latest_ns is None:
            if self.epoch_ns is None:
                self._set_epoch(stamp_ns, wall)
            if self.sessions == 0:
                self.sessions = 1
            self.t_latest_ns = stamp_ns
            self.wall_at_latest = wall
            self.last_accept_wall = wall
            self.last_stamp[topic] = stamp_ns
            return True
        delta = stamp_ns - self.t_latest_ns
        if delta < -self.reset_back_ns or delta > self.jump_fwd_ns:
            # большой скачок всей шкалы: принимаем только если подтверждён несколькими топиками
            self.pending_jump = [p for p in self.pending_jump
                                 if wall - p[1] < 1.5 and abs(p[0] - stamp_ns) < 2 * NS]
            self.pending_jump.append((stamp_ns, wall, topic))
            self._count('%s_time_jump' % topic)
            topics = {p[2] for p in self.pending_jump}
            active = len(self.last_stamp)
            if len(self.pending_jump) >= 3 and (len(topics) >= 2 or
                                               (active == 1 and topic in self.last_stamp)):
                back = delta < 0
                pend = list(self.pending_jump)
                self._reset_session('time jump %+.1f s' % (delta / NS))
                self._set_epoch(min(p[0] for p in pend) if back else stamp_ns, wall)
                self.t_latest_ns = stamp_ns
                self.wall_at_latest = wall
                self.last_accept_wall = wall
                self.last_stamp = {topic: stamp_ns}
                return True
            return False
        last = self.last_stamp.get(topic)
        if last is not None:
            if stamp_ns == last:
                self._count('%s_duplicate' % topic)
                return False
            if stamp_ns < last:
                self._count('%s_out_of_order' % topic)
                return False
            jump = stamp_ns - last > self.topic_jump_ns
        else:
            # первое сообщение топика (в начале записи бывает пачка задержанных сообщений,
            # поэтому по часам не проверяем — только грубо относительно других топиков)
            jump = stamp_ns - self.t_latest_ns > self.first_msg_tol_ns
        if jump and not self._consistent_with_clock(stamp_ns, wall):
            self._count('%s_time_glitch' % topic)
            return False
        self.suspects = []
        self.last_accept_wall = wall
        self.last_stamp[topic] = stamp_ns
        if stamp_ns > self.t_latest_ns:
            self.t_latest_ns = stamp_ns
            self.wall_at_latest = wall
        return True

    # ------------------------------------------------------------------ входы
    def on_input(self, topic, stamp_ns, value, wall=None, proc_start=None):
        """topic: 'front'|'rear'|'cmd'. Возвращает Output или None (вход пропущен)."""
        wall = time.monotonic() if wall is None else wall
        self._count('%s_rx' % topic)
        # значение
        if topic == 'cmd':
            try:
                iv = int(value)
            except (TypeError, ValueError):
                self._count('cmd_bad_value')
                return None
            if iv < -15 or iv > 15:
                self._count('cmd_bad_value')
                return None
            val = iv
        else:
            try:
                val = float(value)
            except (TypeError, ValueError):
                self._count('%s_bad_value' % topic)
                return None
            if not math.isfinite(val):
                self._count('%s_bad_value' % topic)
                return None
        if not self._accept_stamp(topic, stamp_ns, wall):
            return None
        self._count('%s_ok' % topic)
        t = self.rel(stamp_ns)
        method = {'front': 'on_front_velocity', 'rear': 'on_rear_velocity',
                  'cmd': 'on_driver_cmd'}[topic]
        for est in (self.main, self.fb):
            if est is not None:
                est.call(method, t, val)
        self.last_input_wall = wall
        self.rate_samples.append((wall, stamp_ns))
        out = self._make_output(stamp_ns, t, 'input')
        self._after_output(out, wall)
        return out

    def on_gnss_fix(self, antenna, stamp_ns, lat, lon, alt, status, cov, wall=None):
        wall = time.monotonic() if wall is None else wall
        key = '%s_fix' % antenna
        self._count('gnss_%s_rx' % key)
        midroute = False
        if not self.gnss_open:
            if not self.midroute:
                self._count('gnss_after_close_ignored')
                return
            midroute = True
        if not isinstance(stamp_ns, int) or stamp_ns <= 0:
            self._count('gnss_bad_stamp')
            return
        if self.epoch_ns is None:
            self._set_epoch(stamp_ns, wall)
        elif self.t_latest_ns is not None and abs(stamp_ns - self.t_latest_ns) > 30 * NS:
            self._count('gnss_bad_stamp')
            return
        try:
            vals = [float(lat), float(lon), float(alt)]
        except (TypeError, ValueError):
            self._count('gnss_bad_value')
            return
        if not all(math.isfinite(v) for v in vals[:2]):
            self._count('gnss_bad_value')
            return
        t = self.rel(stamp_ns)
        if midroute:
            self._count('gnss_midroute_fix')
            for est in (self.main, self.fb):
                if est is not None:
                    est.call('on_gnss_correction', t, antenna, vals[0], vals[1], vals[2],
                             int(status), list(cov) if cov is not None else [])
            return
        self.gnss_counts[key] += 1
        for est in (self.main, self.fb):
            if est is not None:
                est.call('on_gnss_fix', t, antenna, vals[0], vals[1], vals[2], int(status),
                         list(cov) if cov is not None else [])

    def on_gnss_vel(self, antenna, stamp_ns, vx, vy, vz, wall=None):
        key = '%s_vel' % antenna
        self._count('gnss_%s_rx' % key)
        if not self.gnss_open or self.epoch_ns is None:
            return
        if not isinstance(stamp_ns, int) or stamp_ns <= 0:
            return
        try:
            v = [float(vx), float(vy), float(vz)]
        except (TypeError, ValueError):
            return
        if not all(math.isfinite(a) for a in v):
            return
        self.gnss_counts[key] += 1
        t = self.rel(stamp_ns)
        for est in (self.main, self.fb):
            if est is not None:
                est.call('on_gnss_vel', t, antenna, v[0], v[1], v[2])

    # ------------------------------------------------------------------ окно GNSS
    def gnss_should_close(self):
        if not self.gnss_open or self.epoch_ns is None or self.t_latest_ns is None:
            return False
        elapsed = self.rel(self.t_latest_ns)
        if elapsed < self.gnss_duration:
            return False
        if elapsed >= self.gnss_max_wait:
            return True
        n_fix = self.gnss_counts['master_fix'] + self.gnss_counts['rover_fix']
        if n_fix < self.gnss_min_fixes:
            return False
        st = self._state_of(self.main) or self._state_of(self.fb)
        return bool(st and st.position_valid)

    def close_gnss_window(self, wall=None):
        wall = time.monotonic() if wall is None else wall
        self.gnss_open = False
        t = self.rel(self.t_latest_ns) if self.t_latest_ns is not None else 0.0
        for est in (self.main, self.fb):
            if est is not None:
                est.call('on_gnss_window_closed', t)
        st = self._state_of(self.main) or self._state_of(self.fb)
        info = {
            'bag_time_sec': (self.t_latest_ns or 0) / NS,
            'since_start_sec': t,
            'wall_since_start_sec': (wall - self.session_start_wall)
            if self.session_start_wall is not None else math.nan,
            'counts': dict(self.gnss_counts),
            'position_valid': bool(st and st.position_valid),
            'mode': st.mode if st else '?',
        }
        self.gnss_closed_info = info
        self.gnss_events.append(info)
        return info

    # ------------------------------------------------------------------ выходы
    def _state_of(self, est):
        if est is None or self.epoch_ns is None or self.t_latest_ns is None:
            return None
        raw = est.call('get_state', self.rel(self.t_latest_ns))
        if raw is None:
            return None
        try:
            return EstimatorState.from_any(raw)
        except Exception:  # noqa: BLE001
            return None

    def _get_state(self, est, t):
        if est is None:
            return None
        raw = est.call('get_state', t)
        if raw is None:
            return None
        try:
            st = EstimatorState.from_any(raw)
        except Exception:  # noqa: BLE001
            return None
        return st

    def _make_output(self, stamp_ns, t, source):
        out = Output()
        out.stamp_ns = stamp_ns
        out.source = source
        st = self._get_state(self.main, t)
        fb = self._get_state(self.fb, t) if self.fb is not None else None
        main_ok = self.main is not None and self.main.healthy()
        use = st
        # скорость
        v = None
        if st is not None and main_ok and _finite(st.velocity) and st.velocity_valid:
            v = float(st.velocity)
            var_v = st.var_velocity
        elif fb is not None and _finite(fb.velocity) and fb.velocity_valid:
            v = float(fb.velocity)
            var_v = fb.var_velocity
            out.fallback = True
            use = fb if use is None else use
        if v is not None:
            out.velocity = v
            out.var_velocity = float(var_v) if _finite(var_v) else 1.0
            out.publish_velocity = True
        # положение
        pos = None
        if st is not None and main_ok and st.position_valid and _finite(st.x, st.y, st.z, st.yaw):
            pos = st
        elif fb is not None and fb.position_valid and _finite(fb.x, fb.y, fb.z, fb.yaw):
            pos = fb
            out.fallback = True
        if pos is not None:
            out.x, out.y, out.z = float(pos.x), float(pos.y), float(pos.z)
            pitch = float(pos.pitch) if _finite(pos.pitch) else 0.0
            out.q = quat_from_yaw_pitch(float(pos.yaw), pitch)
            out.var_position = float(pos.var_position) if _finite(pos.var_position) else 100.0
            yr = (pos.extra or {}).get('yaw_rate') if isinstance(pos.extra, dict) else None
            out.yaw_rate = float(yr) if _finite(yr) else 0.0
            out.publish_position = True
        out.state = use if use is not None else (pos or fb)
        return out

    def _after_output(self, out, wall):
        if out.publish_velocity:
            self.n_out_vel += 1
        if out.publish_position:
            self.n_out_pos += 1
        if out.fallback:
            self.n_out_fallback += 1
        if out.publish_velocity or out.publish_position:
            self.last_output_wall = wall
            if self.last_out_stamp_ns is None or out.stamp_ns > self.last_out_stamp_ns:
                self.last_out_stamp_ns = out.stamp_ns

    def playback_rate(self):
        if self.fixed_rate > 0:
            return self.fixed_rate
        rs = self.rate_samples
        if len(rs) < 20:
            return 1.0
        w0, s0 = rs[0]
        w1, s1 = rs[-1]
        if w1 - w0 < 3.0:
            return 1.0
        r = (s1 - s0) / NS / (w1 - w0)
        return min(max(r, 0.05), 20.0) if math.isfinite(r) else 1.0

    def gap_fill(self, wall=None):
        """Вызывается таймером. Если входы пропали — прогноз оценщика с оценённым временем."""
        wall = time.monotonic() if wall is None else wall
        if self.t_latest_ns is None or self.last_input_wall is None:
            return None
        idle = wall - self.last_input_wall
        if idle < self.gap_timeout or idle > self.max_gap_fill:
            return None
        if self.last_output_wall is not None and wall - self.last_output_wall < self.gap_period:
            return None
        dt = (wall - self.wall_at_latest) * self.playback_rate()
        stamp = self.t_latest_ns + int(dt * NS)
        if self.last_out_stamp_ns is not None and stamp <= self.last_out_stamp_ns:
            return None
        out = self._make_output(stamp, self.rel(stamp), 'predict')
        if out.publish_velocity or out.publish_position:
            self.n_out_pred += 1
            self._after_output(out, wall)
            return out
        return None

    def stats(self):
        return {
            'counters': dict(self.counters),
            'latency': self.latency.summary(),
            'out_velocity': self.n_out_vel,
            'out_position': self.n_out_pos,
            'out_predicted': self.n_out_pred,
            'out_fallback': self.n_out_fallback,
            'main_errors': self.main.errors if self.main else -1,
            'fallback_errors': self.fb.errors if self.fb else 0,
            'estimator': self.main_name,
            'fallback': self.fb_name,
            'gnss_open': self.gnss_open,
            'gnss_counts': dict(self.gnss_counts),
            'sessions': self.sessions,
        }


def _finite(*vals):
    try:
        return all(v is not None and math.isfinite(float(v)) for v in vals)
    except (TypeError, ValueError):
        return False
