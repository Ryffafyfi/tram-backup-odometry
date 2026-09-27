#!/usr/bin/env python3
"""ROS 2 нода резервной одометрии трамвая (tram_backup_odometry).

Входы:  /vehicle/front_bogie_velocity, /vehicle/rear_bogie_velocity (VelocitySensor),
        /vehicle/driver_position_cmd (DriverControllerCommand)
        /sensing/gnss/{master,rover}/{fix,vel} — ТОЛЬКО первые gnss_init_duration_sec секунд,
        после чего подписки удаляются (в лог пишется момент удаления).
Выходы: /result/velocity (VelocitySensor), /result/position (nav_msgs/Odometry),
        /diagnostics (флаг проскальзывания, оценка сцепления, задержки, ресурсы).

header.stamp каждого ответа = header.stamp входного сообщения, из которого он посчитан.
"""

import gc
import json
import math
import os
import resource
import time

import rclpy
from rclpy.clock import Clock, ClockType
try:
    from rclpy.executors import ExternalShutdownException
except ImportError:  # pragma: no cover
    class ExternalShutdownException(Exception):
        pass
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor

from builtin_interfaces.msg import Time as TimeMsg
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float64

from tram_vehicle_msgs.msg import VelocitySensor
try:  # в части окружений (check-code) пакет сообщений содержит только VelocitySensor
    from tram_vehicle_msgs.msg import DriverControllerCommand
except ImportError:  # pragma: no cover
    DriverControllerCommand = None

from .config_util import finalize_config
from .core import NS, OdometryCore

PKG = 'tram_backup_odometry'

DEFAULTS = {
    # топики
    'front_topic': '/vehicle/front_bogie_velocity',
    'rear_topic': '/vehicle/rear_bogie_velocity',
    'cmd_topic': '/vehicle/driver_position_cmd',
    'velocity_out_topic': '/result/velocity',
    'position_out_topic': '/result/position',
    'diagnostics_topic': '/diagnostics',
    'gnss_master_fix_topic': '/sensing/gnss/master/fix',
    'gnss_master_vel_topic': '/sensing/gnss/master/vel',
    'gnss_rover_fix_topic': '/sensing/gnss/rover/fix',
    'gnss_rover_vel_topic': '/sensing/gnss/rover/vel',
    'input_qos_reliable': False,
    # кадры
    'map_frame': 'map',
    'base_frame': 'base_link',
    # карта и перевод GNSS -> карта
    'map_files': ['maps/shchukinskaya_tallinskaya.json', 'maps/tallinskaya_shchukinskaya.json',
                  'maps/terminal_loops.json'],
    # ветки, которых нет в графе путей (средний путь «Таллинской»): выход — взвешенно по вероятности
    'branch_map_files': ['maps/tallinskaya_middle.json'],
    # линии пути в формате track_*.csv (track_map для оценщика с интерфейсом on_notch/on_bogie/state)
    'track_csv_files': ['maps/track_tallinskaya_shchukinskaya.csv',
                        'maps/track_shchukinskaya_tallinskaya.csv'],
    'gnss_tm_lon0_deg': 39.0,
    'gnss_tm_k0': 0.9996,
    'gnss_tm_false_easting': 500000.0,
    'gnss_tm_false_northing': 0.0,
    'gnss_map_offset_x': -300000.0,
    'gnss_map_offset_y': -6100000.0,
    'gnss_map_offset_z': 0.0,
    # GNSS-старт
    'gnss_init_duration_sec': 5.0,
    'gnss_init_max_wait_sec': 20.0,
    'gnss_init_min_fixes': 10,
    'gnss_midroute_corrections': False,
    # геометрия трамвая (base_link — ось поворота передней тележки, высота — головка рельса)
    'antenna_master_xyz': [-9.873, 0.0, 3.0],
    'antenna_rover_xyz': [2.563, 0.0, 3.0],
    'bogie_distance': 7.55,
    # оценщик
    'estimator_class': 'tram_backup_odometry.estimator:Estimator',
    'fallback_estimator_class': 'tram_backup_odometry.baseline_estimator:BaselineEstimator',
    'fallback_on_error': True,
    'coefficients_file': 'config/params.yaml',
    'baseline_coefficients_file': 'config/baseline_params.yaml',
    # переходник для оценщика с интерфейсом on_notch/on_bogie/state (on_notch / on_bogie / state)
    'agreed_api_velocity_scale': 1.0 / 3.6,
    'agreed_api_track_map': 'csv',
    'stop_landmarks_file': 'maps/stop_landmarks.json',
    # пропуски / прогноз
    'gap_timeout_sec': 0.07,
    'gap_publish_period_sec': 0.05,
    'max_gap_fill_sec': 30.0,
    'playback_rate': 0.0,
    # живучесть
    'time_jump_reset_sec': 3.0,
    'time_jump_forward_sec': 30.0,
    'topic_jump_sec': 0.6,
    'topic_ahead_tolerance_sec': 0.3,
    'publish_position_before_init': False,
    # диагностика и замеры
    'diagnostics_period_sec': 1.0,
    'stats_log_period_sec': 10.0,
    'metrics_dir': '',
    'hardware_id': 'tram',
    'vehicle_id': '30618',
}


def stamp_to_ns(stamp):
    return int(stamp.sec) * NS + int(stamp.nanosec)


def ns_to_stamp(ns):
    m = TimeMsg()
    m.sec = int(ns // NS)
    m.nanosec = int(ns % NS)
    return m


def _read_mem_kb():
    rss = hwm = -1
    try:
        with open('/proc/self/status', 'r') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    rss = int(line.split()[1])
                elif line.startswith('VmHWM:'):
                    hwm = int(line.split()[1])
    except OSError:
        hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss, hwm


class TramBackupOdometryNode(Node):

    def __init__(self):
        super().__init__(PKG)
        self.cfg = self._load_params()
        self.core = OdometryCore(self.cfg, log=self._log)
        self.get_logger().info('estimator: %s, fallback: %s' % (self.core.main_name,
                                                                self.core.fb_name))
        rel = ReliabilityPolicy.RELIABLE if self.cfg['input_qos_reliable'] \
            else ReliabilityPolicy.BEST_EFFORT
        self.qos_in = QoSProfile(reliability=rel, history=HistoryPolicy.KEEP_LAST, depth=200,
                                 durability=DurabilityPolicy.VOLATILE)
        qos_out = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=50,
                             durability=DurabilityPolicy.VOLATILE)
        self.pub_vel = self.create_publisher(VelocitySensor, self.cfg['velocity_out_topic'], qos_out)
        self.pub_pos = self.create_publisher(Odometry, self.cfg['position_out_topic'], qos_out)
        self.pub_diag = self.create_publisher(DiagnosticArray, self.cfg['diagnostics_topic'], 10)
        self.pub_slip = self.create_publisher(Bool, '~/slip_detected', 10)
        self.pub_adh = self.create_publisher(Float64, '~/adhesion_estimate', 10)

        self.create_subscription(VelocitySensor, self.cfg['front_topic'],
                                 lambda m: self._on_bogie('front', m), self.qos_in)
        self.create_subscription(VelocitySensor, self.cfg['rear_topic'],
                                 lambda m: self._on_bogie('rear', m), self.qos_in)
        if DriverControllerCommand is not None:
            self.create_subscription(DriverControllerCommand, self.cfg['cmd_topic'],
                                     self._on_cmd, self.qos_in)
        else:
            self.get_logger().warn(
                'tram_vehicle_msgs has no DriverControllerCommand: %s is not used '
                '(estimator works from bogie velocities only)' % self.cfg['cmd_topic'])

        self.gnss_subs = []
        self._gnss_kept = False
        self._open_gnss()

        # шаблоны ковариаций
        self._pose_cov = [0.0] * 36
        self._twist_cov = [0.0] * 36
        self.last_slip = None
        self._diag_pending = False
        self.last_state = None
        self.last_out_stamp = None
        self._out_times = []
        self._last_pub_wall = None
        self._max_pub_gap = 0.0          # максимальный интервал между ответами за всё время, с
        self._max_pub_gap_period = 0.0   # то же за последний период статистики
        # таймеры на монотонных часах: работают и при use_sim_time без /clock
        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.02, self._on_gap_timer, clock=steady)
        self.create_timer(max(0.1, float(self.cfg['diagnostics_period_sec'])),
                          self._publish_diagnostics, clock=steady)
        self.create_timer(max(1.0, float(self.cfg['stats_log_period_sec'])),
                          self._log_stats, clock=steady)
        self._cpu_prev = (time.monotonic(), time.process_time())
        self._cpu_pct = 0.0
        self._start_wall = time.monotonic()
        self._metrics_path = self._init_metrics_path()
        self.get_logger().info(
            'ready: in [%s, %s, %s] -> out [%s, %s]; GNSS start window %.1f s' % (
                self.cfg['front_topic'], self.cfg['rear_topic'], self.cfg['cmd_topic'],
                self.cfg['velocity_out_topic'], self.cfg['position_out_topic'],
                self.core.gnss_duration))

    # ------------------------------------------------------------------ параметры
    def _load_params(self):
        desc = ParameterDescriptor(dynamic_typing=True)
        cfg = {}
        for k, v in DEFAULTS.items():
            try:
                self.declare_parameter(k, v, desc)
                cfg[k] = self.get_parameter(k).value
            except Exception as e:  # noqa: BLE001
                self.get_logger().error('parameter %s: %r -> default %r' % (k, e, v))
                cfg[k] = v
            if cfg[k] is None:
                cfg[k] = v
        for k, v in DEFAULTS.items():  # приведение типов (YAML может дать int вместо float)
            try:
                if isinstance(v, bool):
                    cfg[k] = bool(cfg[k])
                elif isinstance(v, float):
                    cfg[k] = float(cfg[k])
                elif isinstance(v, list):
                    cfg[k] = list(cfg[k])
                elif isinstance(v, str):
                    cfg[k] = str(cfg[k])
            except (TypeError, ValueError):
                cfg[k] = v
        # пути относительно share пакета, коэффициенты физической модели и базового оценщика
        return finalize_config(cfg, self._share_dir(), self._log)

    def _share_dir(self):
        try:
            from ament_index_python.packages import get_package_share_directory
            return get_package_share_directory(PKG)
        except Exception:  # noqa: BLE001
            return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _log(self, level, msg):
        lg = self.get_logger()
        {'error': lg.error, 'warn': lg.warn, 'info': lg.info}.get(level, lg.info)(msg)

    # ------------------------------------------------------------------ GNSS (только старт)
    def _open_gnss(self):
        if self.gnss_subs:
            return
        c = self.cfg
        self.gnss_subs = [
            self.create_subscription(NavSatFix, c['gnss_master_fix_topic'],
                                     lambda m: self._on_fix('master', m), self.qos_in),
            self.create_subscription(NavSatFix, c['gnss_rover_fix_topic'],
                                     lambda m: self._on_fix('rover', m), self.qos_in),
            self.create_subscription(TwistStamped, c['gnss_master_vel_topic'],
                                     lambda m: self._on_gvel('master', m), self.qos_in),
            self.create_subscription(TwistStamped, c['gnss_rover_vel_topic'],
                                     lambda m: self._on_gvel('rover', m), self.qos_in),
        ]
        self.get_logger().info('GNSS start window opened (%.1f s of bag time): subscribed to '
                               'GNSS topics for initial alignment only' % self.core.gnss_duration)

    def _close_gnss(self):
        info = self.core.close_gnss_window()
        if self.core.midroute:
            self.get_logger().warn(
                'GNSS start window closed at bag time %.3f (%.2f s after start); fixes used: '
                'master %d, rover %d. NON-DEFAULT: gnss_midroute_corrections=true — GNSS '
                'subscriptions are KEPT for sparse mid-route corrections.' % (
                    info['bag_time_sec'], info['since_start_sec'],
                    info['counts']['master_fix'], info['counts']['rover_fix']))
            self._gnss_kept = True
            return
        for s in self.gnss_subs:
            try:
                self.destroy_subscription(s)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error('destroy_subscription failed: %r' % (e,))
        self.gnss_subs = []
        self.get_logger().warn(
            'GNSS SUBSCRIPTIONS REMOVED at bag time %.3f (%.2f s after start, wall %.2f s); '
            'fixes used: master %d, rover %d; vel: master %d, rover %d; position_valid=%s '
            'mode=%s. From now on only wheel velocities and driver controller are used.' % (
                info['bag_time_sec'], info['since_start_sec'], info['wall_since_start_sec'],
                info['counts']['master_fix'], info['counts']['rover_fix'],
                info['counts']['master_vel'], info['counts']['rover_vel'],
                info['position_valid'], info['mode']))
        self._write_metrics()

    def _on_fix(self, antenna, msg):
        if not self.gnss_subs:
            return
        try:
            self.core.on_gnss_fix(antenna, stamp_to_ns(msg.header.stamp), msg.latitude,
                                  msg.longitude, msg.altitude, msg.status.status,
                                  msg.position_covariance, time.monotonic())
        except Exception as e:  # noqa: BLE001
            self.get_logger().error('gnss fix handler: %r' % (e,), throttle_duration_sec=5.0)
        self._check_gnss()

    def _on_gvel(self, antenna, msg):
        if not self.gnss_subs:
            return
        try:
            v = msg.twist.linear
            self.core.on_gnss_vel(antenna, stamp_to_ns(msg.header.stamp), v.x, v.y, v.z,
                                  time.monotonic())
        except Exception as e:  # noqa: BLE001
            self.get_logger().error('gnss vel handler: %r' % (e,), throttle_duration_sec=5.0)

    def _check_gnss(self):
        try:
            if self._gnss_kept and self.core.gnss_open is False:
                return
            if self.core.gnss_open:
                if not self.gnss_subs:
                    self._open_gnss()  # новая сессия (запись запущена заново)
                elif self.core.gnss_should_close():
                    self._close_gnss()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error('gnss window check: %r' % (e,), throttle_duration_sec=5.0)

    # ------------------------------------------------------------------ основной контур
    def _on_bogie(self, which, msg):
        t0 = time.perf_counter()
        try:
            out = self.core.on_input(which, stamp_to_ns(msg.header.stamp), msg.velocity,
                                     time.monotonic())
            if out is not None:
                self._publish(out, msg.header.stamp)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error('%s handler: %r' % (which, e), throttle_duration_sec=5.0)
        self.core.latency.add((time.perf_counter() - t0) * 1000.0)
        self._check_gnss()

    def _on_cmd(self, msg):
        t0 = time.perf_counter()
        try:
            out = self.core.on_input('cmd', stamp_to_ns(msg.header.stamp), msg.position,
                                     time.monotonic())
            if out is not None:
                self._publish(out, msg.header.stamp)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error('cmd handler: %r' % (e,), throttle_duration_sec=5.0)
        self.core.latency.add((time.perf_counter() - t0) * 1000.0)
        self._check_gnss()

    def _on_gap_timer(self):
        try:
            out = self.core.gap_fill(time.monotonic())
            if out is not None:
                self._publish(out, None)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error('gap timer: %r' % (e,), throttle_duration_sec=5.0)
        if self._diag_pending:
            self._diag_pending = False
            self._publish_diagnostics()
        self._check_gnss()

    def _publish(self, out, stamp_msg):
        stamp = stamp_msg if stamp_msg is not None else ns_to_stamp(out.stamp_ns)
        if out.publish_velocity:
            m = VelocitySensor()
            m.header.stamp = stamp
            m.header.frame_id = self.cfg['base_frame']
            m.velocity = float(out.velocity)
            self.pub_vel.publish(m)
        if out.publish_position:
            o = Odometry()
            o.header.stamp = stamp
            o.header.frame_id = self.cfg['map_frame']
            o.child_frame_id = self.cfg['base_frame']
            p = o.pose.pose.position
            p.x, p.y, p.z = out.x, out.y, out.z
            q = o.pose.pose.orientation
            q.x, q.y, q.z, q.w = out.q
            vp = max(out.var_position, 1e-4)
            pc = self._pose_cov
            pc[0] = pc[7] = vp
            pc[14] = 0.25
            pc[21] = pc[28] = 0.01
            pc[35] = 0.01
            o.pose.covariance = pc
            o.twist.twist.linear.x = float(out.velocity)
            o.twist.twist.angular.z = float(out.yaw_rate)
            tc = self._twist_cov
            tc[0] = max(out.var_velocity, 1e-6)
            tc[7] = tc[14] = 1e-4
            tc[21] = tc[28] = tc[35] = 1e-2
            o.twist.covariance = tc
            self.pub_pos.publish(o)
        st = out.state
        if st is not None:
            self.last_state = st
            slip = bool(st.slip)
            if slip != self.last_slip:
                self.last_slip = slip
                self._diag_pending = True   # опубликует таймер (не нагружаем колбэк входа)
        self.last_out_stamp = out.stamp_ns
        now = time.monotonic()
        if self._last_pub_wall is not None:
            gap = now - self._last_pub_wall
            if gap > self._max_pub_gap:
                self._max_pub_gap = gap
            if gap > self._max_pub_gap_period:
                self._max_pub_gap_period = gap
        self._last_pub_wall = now
        self._out_times.append(now)

    # ------------------------------------------------------------------ диагностика/замеры
    def _resources(self):
        now = time.monotonic()
        cpu = time.process_time()
        w0, c0 = self._cpu_prev
        if now - w0 >= 0.5:
            self._cpu_pct = 100.0 * (cpu - c0) / (now - w0)
            self._cpu_prev = (now, cpu)
        rss, hwm = _read_mem_kb()
        return self._cpu_pct, rss / 1024.0, hwm / 1024.0

    def _out_rate(self):
        now = time.monotonic()
        self._out_times = [t for t in self._out_times if now - t <= 5.0]
        return len(self._out_times) / 5.0

    def _publish_diagnostics(self):
        try:
            arr = DiagnosticArray()
            arr.header.stamp = ns_to_stamp(self.last_out_stamp) if self.last_out_stamp \
                else self.get_clock().now().to_msg()
            st = self.last_state
            hw = str(self.cfg['hardware_id'])
            # 1) проскальзывание / сцепление
            s1 = DiagnosticStatus()
            s1.name = '%s: wheel slip' % PKG
            s1.hardware_id = hw
            slip = bool(st.slip) if st is not None else False
            s1.level = DiagnosticStatus.WARN if slip else DiagnosticStatus.OK
            s1.message = 'wheel slip/slide detected' if slip else 'no slip'
            adh = float(st.adhesion) if st is not None and _fin(st.adhesion) else float('nan')
            s1.values = [
                KeyValue(key='slip_detected', value=str(slip).lower()),
                KeyValue(key='adhesion_estimate', value='%.3f' % adh),
                KeyValue(key='slip_ratio', value='%.3f' % (float(st.slip_ratio)
                                                          if st is not None else 0.0)),
                KeyValue(key='velocity_mps', value='%.3f' % (float(st.velocity)
                                                            if st is not None else 0.0)),
                KeyValue(key='acceleration_mps2', value='%.3f' % (float(st.acceleration)
                                                                 if st is not None else 0.0)),
            ]
            # 2) оценщик
            s2 = DiagnosticStatus()
            s2.name = '%s: estimator' % PKG
            s2.hardware_id = hw
            valid = bool(st.position_valid) if st is not None else False
            s2.level = DiagnosticStatus.OK if valid else DiagnosticStatus.WARN
            s2.message = (st.mode if st is not None else 'no data')
            stats = self.core.stats()
            s2.values = [
                KeyValue(key='estimator', value=stats['estimator']),
                KeyValue(key='fallback', value=stats['fallback']),
                KeyValue(key='estimator_errors', value=str(stats['main_errors'])),
                KeyValue(key='fallback_outputs', value=str(stats['out_fallback'])),
                KeyValue(key='position_valid', value=str(valid).lower()),
                KeyValue(key='track', value=str(st.track) if st is not None else ''),
                KeyValue(key='s_m', value='%.2f' % (float(st.s) if st is not None else 0.0)),
                KeyValue(key='distance_m', value='%.1f' % (float(st.distance)
                                                          if st is not None else 0.0)),
                KeyValue(key='gnss_window_open', value=str(self.core.gnss_open).lower()),
                KeyValue(key='gnss_fixes', value=json.dumps(stats['gnss_counts'])),
                KeyValue(key='sessions', value=str(stats['sessions'])),
            ]
            # доп. сведения оценщика (число поправок у ориентиров, поправка масштаба колёс и т.п.)
            extra = getattr(st, 'extra', None) if st is not None else None
            if isinstance(extra, dict):
                for k, v in list(extra.items())[:20]:
                    if isinstance(v, (bool, int, float, str)):
                        s2.values.append(KeyValue(key=str(k), value=str(v)))
            # 3) реальное время и ресурсы
            s3 = DiagnosticStatus()
            s3.name = '%s: performance' % PKG
            s3.hardware_id = hw
            lat = stats['latency']
            cpu, rss, hwm = self._resources()
            rate = self._out_rate()
            bad = (_fin(lat['p99']) and lat['p99'] > 100.0) or rss > 450.0
            s3.level = DiagnosticStatus.WARN if bad else DiagnosticStatus.OK
            s3.message = 'latency p99 %.2f ms, rate %.1f Hz, cpu %.1f%%, rss %.0f MB' % (
                lat['p99'] if _fin(lat['p99']) else -1, rate, cpu, rss)
            c = stats['counters']
            s3.values = [
                KeyValue(key='latency_ms_p50', value='%.3f' % lat['p50']),
                KeyValue(key='latency_ms_p99', value='%.3f' % lat['p99']),
                KeyValue(key='latency_ms_max', value='%.3f' % lat['max_all']),
                KeyValue(key='output_rate_hz', value='%.1f' % rate),
                KeyValue(key='max_output_gap_sec', value='%.3f' % self._max_pub_gap),
                KeyValue(key='cpu_percent', value='%.1f' % cpu),
                KeyValue(key='rss_mb', value='%.1f' % rss),
                KeyValue(key='rss_peak_mb', value='%.1f' % hwm),
                KeyValue(key='out_velocity', value=str(stats['out_velocity'])),
                KeyValue(key='out_position', value=str(stats['out_position'])),
                KeyValue(key='out_predicted', value=str(stats['out_predicted'])),
                KeyValue(key='rejected_inputs', value=json.dumps(
                    {k: v for k, v in c.items()
                     if k.endswith(('_bad_stamp', '_bad_value', '_duplicate', '_out_of_order',
                                    '_time_jump', '_time_glitch'))})),
            ]
            arr.status = [s1, s2, s3]
            self.pub_diag.publish(arr)
            self.pub_slip.publish(Bool(data=slip))
            if _fin(adh):
                self.pub_adh.publish(Float64(data=adh))
        except Exception as e:  # noqa: BLE001
            self.get_logger().error('diagnostics: %r' % (e,), throttle_duration_sec=10.0)

    def _init_metrics_path(self):
        d = str(self.cfg.get('metrics_dir') or '')
        if not d:
            d = os.path.join(os.environ.get('ROS_LOG_DIR') or
                             os.path.join(os.path.expanduser('~'), '.ros', 'log'), PKG)
        try:
            os.makedirs(d, exist_ok=True)
            return os.path.join(d, 'metrics_%s.json' % time.strftime('%Y%m%d_%H%M%S'))
        except OSError:
            return None

    def _write_metrics(self):
        if not self._metrics_path:
            return
        try:
            cpu, rss, hwm = self._resources()
            stats = self.core.stats()
            stats.update({'cpu_percent': cpu, 'rss_mb': rss, 'rss_peak_mb': hwm,
                          'output_rate_hz_5s': self._out_rate(),
                          'max_output_gap_sec': self._max_pub_gap,
                          'uptime_sec': time.monotonic() - self._start_wall,
                          'gnss_windows': self.core.gnss_events})
            tmp = self._metrics_path + '.tmp'
            with open(tmp, 'w') as fh:
                json.dump(stats, fh, indent=1, default=str)
            os.replace(tmp, self._metrics_path)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn('metrics write failed: %r' % (e,), throttle_duration_sec=60.0)

    def _log_stats(self):
        try:
            stats = self.core.stats()
            lat = stats['latency']
            cpu, rss, hwm = self._resources()
            c = stats['counters']
            rej = sum(v for k, v in c.items() if k.endswith(
                ('_bad_stamp', '_bad_value', '_duplicate', '_out_of_order', '_time_jump',
                 '_time_glitch')))
            st = self.last_state
            self.get_logger().info(
                'stats: out vel=%d pos=%d pred=%d fallback=%d rate=%.1fHz maxgap=%.3fs | '
                'latency ms p50=%.3f p99=%.3f max=%.3f | cpu=%.1f%% rss=%.1fMB peak=%.1fMB | '
                'rejected=%d | %s v=%.2f slip=%s' % (
                    stats['out_velocity'], stats['out_position'], stats['out_predicted'],
                    stats['out_fallback'], self._out_rate(), self._max_pub_gap_period,
                    lat['p50'], lat['p99'],
                    lat['max_all'], cpu, rss, hwm, rej, st.mode if st is not None else '-',
                    float(st.velocity) if st is not None else 0.0,
                    st.slip if st is not None else '-'))
            self._max_pub_gap_period = 0.0
            self._write_metrics()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error('stats: %r' % (e,), throttle_duration_sec=30.0)

    def shutdown_report(self):
        self._log_stats()
        if self._metrics_path:
            self.get_logger().info('metrics saved to %s' % self._metrics_path)


def _fin(v):
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TramBackupOdometryNode()
        # карта и прочие долгоживущие объекты — в «вечное» поколение GC: сборщик мусора их
        # больше не обходит, паузы GC в колбэках становятся короче
        gc.collect()
        gc.freeze()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as e:  # noqa: BLE001
        if node is not None:
            node.get_logger().fatal('unhandled: %r' % (e,))
        raise
    finally:
        if node is not None:
            try:
                node.shutdown_report()
            except Exception:  # noqa: BLE001
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
