#!/usr/bin/env python3
"""Замер быстродействия решения со стороны (как видит судья).

    ros2 run tram_backup_odometry perf_probe --ros-args -p output_file:=/tmp/perf.json

Подписывается на входы и выходы, сопоставляет ответ с входом по header.stamp и считает:
  * задержку «вход -> ответ» (время приёма ответа минус время приёма входа с тем же stamp);
  * частоту /result/velocity и /result/position, максимальный разрыв между ответами;
  * CPU и память процесса ноды (по /proc, ищется процесс с 'tram_backup_odometry' в cmdline).
Сводка печатается каждые 5 с и при завершении (Ctrl-C), по желанию пишется в JSON.
"""

import json
import os
import time
from collections import deque

import rclpy
from rclpy.node import Node
try:
    from rclpy.executors import ExternalShutdownException
except ImportError:  # pragma: no cover
    class ExternalShutdownException(Exception):
        pass
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from tram_vehicle_msgs.msg import VelocitySensor
try:
    from tram_vehicle_msgs.msg import DriverControllerCommand
except ImportError:  # pragma: no cover
    DriverControllerCommand = None

NS = 1_000_000_000


def _pct(vals, p):
    if not vals:
        return float('nan')
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]


def _find_pid(pattern, own=os.getpid()):
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) == own:
            continue
        try:
            with open('/proc/%s/cmdline' % d, 'rb') as fh:
                cmd = fh.read().replace(b'\0', b' ').decode(errors='ignore')
        except OSError:
            continue
        if pattern in cmd and 'perf_probe' not in cmd and 'launch' not in cmd:
            return int(d), cmd.strip()
    return None, ''


def _proc_cpu_mem(pid):
    try:
        with open('/proc/%d/stat' % pid) as fh:
            parts = fh.read().rsplit(')', 1)[1].split()
        ticks = int(parts[11]) + int(parts[12])  # utime + stime
        rss = hwm = 0
        with open('/proc/%d/status' % pid) as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    rss = int(line.split()[1])
                elif line.startswith('VmHWM:'):
                    hwm = int(line.split()[1])
        return ticks / os.sysconf('SC_CLK_TCK'), rss / 1024.0, hwm / 1024.0
    except (OSError, ValueError, IndexError):
        return None


class PerfProbe(Node):
    def __init__(self):
        super().__init__('tram_perf_probe')
        self.declare_parameter('output_file', '')
        self.declare_parameter('process_pattern', 'tram_backup_odometry')
        self.declare_parameter('report_period_sec', 5.0)
        self.out_file = self.get_parameter('output_file').value
        self.pattern = self.get_parameter('process_pattern').value
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=200)
        self.in_recv = {}          # stamp_ns -> wall
        self.in_order = deque()
        self.pending_out = {}      # stamp_ns -> [(kind, wall)] ответ пришёл раньше входа (гонка доставки)
        self.lat = {'velocity': [], 'position': []}
        self.lat_series = []   # (wall_since_start, kind, latency_ms)
        self.unmatched = {'velocity': 0, 'position': 0}
        self.race = {'velocity': 0, 'position': 0}
        self.times = {'velocity': [], 'position': []}
        self.stamps = {'velocity': [], 'position': []}
        self.n_in = 0
        self.create_subscription(VelocitySensor, '/vehicle/front_bogie_velocity', self._in, qos)
        self.create_subscription(VelocitySensor, '/vehicle/rear_bogie_velocity', self._in, qos)
        if DriverControllerCommand is not None:
            self.create_subscription(DriverControllerCommand, '/vehicle/driver_position_cmd',
                                     self._in, qos)
        self.create_subscription(VelocitySensor, '/result/velocity',
                                 lambda m: self._out('velocity', m), qos)
        self.create_subscription(Odometry, '/result/position',
                                 lambda m: self._out('position', m), qos)
        self.pid = None
        self.cpu0 = None
        self.cpu_samples = []
        self.rss_max = 0.0
        self.t0 = time.monotonic()
        self.create_timer(float(self.get_parameter('report_period_sec').value), self.report)
        self.create_timer(1.0, self._sample_proc)

    def _in(self, msg):
        now = time.monotonic()
        s = msg.header.stamp.sec * NS + msg.header.stamp.nanosec
        self.n_in += 1
        if s not in self.in_recv:
            self.in_recv[s] = now
            self.in_order.append(s)
            for kind, t_out in self.pending_out.pop(s, ()):
                # ответ доставлен пробе раньше входа: задержка ноды не больше нуля по часам пробы
                self.lat[kind].append(max(0.0, (t_out - now) * 1000.0))
                self.unmatched[kind] -= 1
                self.race[kind] += 1
        while len(self.in_order) > 20000:
            self.in_recv.pop(self.in_order.popleft(), None)
        if len(self.pending_out) > 2000:
            for k in list(self.pending_out)[:1000]:
                del self.pending_out[k]

    def _out(self, kind, msg):
        now = time.monotonic()
        s = msg.header.stamp.sec * NS + msg.header.stamp.nanosec
        self.times[kind].append(now)
        self.stamps[kind].append(s)
        t_in = self.in_recv.get(s)
        if t_in is None:
            self.unmatched[kind] += 1   # прогноз в пропуске или вход ещё не доставлен пробе
            self.pending_out.setdefault(s, []).append((kind, now))
        else:
            self.lat[kind].append((now - t_in) * 1000.0)
            if len(self.lat_series) < 200000:
                self.lat_series.append((round(now - self.t0, 4), kind[0], round((now - t_in) * 1000.0, 3)))

    def _sample_proc(self):
        if self.pid is None:
            self.pid, cmd = _find_pid(self.pattern)
            if self.pid:
                self.get_logger().info('measuring process %d: %s' % (self.pid, cmd[:120]))
        if self.pid is None:
            return
        r = _proc_cpu_mem(self.pid)
        if r is None:
            self.pid = None
            return
        cpu, rss, hwm = r
        now = time.monotonic()
        if self.cpu0 is not None:
            dt = now - self.cpu0[0]
            if dt > 0:
                self.cpu_samples.append(100.0 * (cpu - self.cpu0[1]) / dt)
        self.cpu0 = (now, cpu)
        self.rss_max = max(self.rss_max, hwm, rss)
        self.rss_now = rss

    def summary(self):
        res = {'inputs_seen': self.n_in, 'duration_sec': time.monotonic() - self.t0}
        for k in ('velocity', 'position'):
            t = self.times[k]
            lat = self.lat[k]
            gaps = [b - a for a, b in zip(t, t[1:])]
            st = sorted(self.stamps[k])
            sgaps = [(b - a) / NS for a, b in zip(st, st[1:])]
            span = (t[-1] - t[0]) if len(t) > 1 else 0.0
            res[k] = {
                'count': len(t),
                'rate_hz': (len(t) - 1) / span if span > 0 else 0.0,
                'latency_ms_mean': sum(lat) / len(lat) if lat else float('nan'),
                'latency_ms_p50': _pct(lat, 0.5), 'latency_ms_p95': _pct(lat, 0.95),
                'latency_ms_p99': _pct(lat, 0.99), 'latency_ms_max': max(lat) if lat else float('nan'),
                'matched': len(lat), 'unmatched': self.unmatched[k],
                'output_before_input_race': self.race[k],
                'max_gap_wall_sec': max(gaps) if gaps else float('nan'),
                'max_gap_stamp_sec': max(sgaps) if sgaps else float('nan'),
                'share_gaps_over_100ms': (sum(1 for g in gaps if g > 0.1) / len(gaps)) if gaps else 0,
            }
        cs = self.cpu_samples
        res['node_process'] = {
            'pid': self.pid,
            'cpu_percent_mean': sum(cs) / len(cs) if cs else float('nan'),
            'cpu_percent_max': max(cs) if cs else float('nan'),
            'rss_mb_peak': self.rss_max,
        }
        return res

    def report(self):
        s = self.summary()
        v, p, n = s['velocity'], s['position'], s['node_process']
        self.get_logger().info(
            'velocity: %d msgs %.1f Hz lat p50/p99/max %.2f/%.2f/%.2f ms maxgap %.3fs | '
            'position: %d msgs %.1f Hz lat p99 %.2f ms maxgap %.3fs | node cpu %.1f%% (max %.1f) '
            'rss peak %.1f MB' % (
                v['count'], v['rate_hz'], v['latency_ms_p50'], v['latency_ms_p99'],
                v['latency_ms_max'], v['max_gap_wall_sec'], p['count'], p['rate_hz'],
                p['latency_ms_p99'], p['max_gap_wall_sec'], n['cpu_percent_mean'],
                n['cpu_percent_max'], n['rss_mb_peak']))
        if self.out_file:
            try:
                with open(self.out_file, 'w') as fh:
                    json.dump(s, fh, indent=1)
                with open(self.out_file + '.series.csv', 'w') as fh:
                    fh.write('t,kind,latency_ms\n')
                    for row in self.lat_series:
                        fh.write('%s,%s,%s\n' % row)
            except OSError as e:
                self.get_logger().warn('cannot write %s: %r' % (self.out_file, e))


def main(args=None):
    rclpy.init(args=args)
    node = PerfProbe()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
