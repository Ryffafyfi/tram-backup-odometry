#!/usr/bin/env python3
"""Офлайн-прогон bag-файла через ту же логику, что и нода (OdometryCore), + метрики «как у судьи».

Без ROS: нужен только pip-пакет rosbags (pip install rosbags). Воспроизводит порядок
сообщений по времени записи (как ros2 bag play), окно GNSS, прогноз в пропусках и
сопоставление с эталоном через эмуляцию message_filters.ApproximateTimeSynchronizer
(slop 0.05 c, очередь 100) — ровно как hackathon_solution_checker.

Эталон: /localization/kinematic_state, если есть в записи (проверочная запись), иначе
псевдо-эталон из GNSS master+rover (base_link), только для отладки.

    python3 tools/offline_eval.py <bag_dir> [<bag_dir> ...] [--set key=value ...] [--csv out.csv]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

from tram_backup_odometry.config_util import finalize_config  # noqa: E402
from tram_backup_odometry.core import NS, OdometryCore  # noqa: E402
from tram_backup_odometry.geo import make_projection  # noqa: E402

VS = 'std_msgs/Header header\nfloat64 velocity\n'
DCC = 'std_msgs/Header header\nint8 position\n'

TOPICS = {
    '/vehicle/front_bogie_velocity': 'front',
    '/vehicle/rear_bogie_velocity': 'rear',
    '/vehicle/driver_position_cmd': 'cmd',
    '/sensing/gnss/master/fix': 'master_fix',
    '/sensing/gnss/rover/fix': 'rover_fix',
    '/sensing/gnss/master/vel': 'master_vel',
    '/sensing/gnss/rover/vel': 'rover_vel',
    '/localization/kinematic_state': 'ref',
}


def default_cfg():
    """Конфигурация как у ноды: YAML ноды + пути от корня пакета + оба файла коэффициентов."""
    import yaml
    with open(PKG_ROOT / 'config' / 'tram_backup_odometry.yaml') as fh:
        cfg = yaml.safe_load(fh)['tram_backup_odometry']['ros__parameters']
    return finalize_config(cfg, PKG_ROOT)


def set_value(cfg, key, val):
    """--set key=value: параметр ноды или коэффициент (в оба файла коэффициентов)."""
    if key in cfg and key not in ('coefficients', 'baseline_coefficients'):
        cfg[key] = val
        return
    cfg.setdefault('coefficients', {})[key] = val
    if isinstance(cfg.get('baseline_coefficients'), dict):
        cfg['baseline_coefficients'][key] = val


def read_bag(path):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    ts = get_typestore(Stores.ROS2_HUMBLE)
    add = {}
    add.update(get_types_from_msg(VS, 'tram_vehicle_msgs/msg/VelocitySensor'))
    add.update(get_types_from_msg(DCC, 'tram_vehicle_msgs/msg/DriverControllerCommand'))
    ts.register(add)
    out = []
    with AnyReader([Path(path)], default_typestore=ts) as r:
        conns = [c for c in r.connections if c.topic in TOPICS]
        for c, t, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            out.append((t, TOPICS[c.topic], m))
    out.sort(key=lambda x: x[0])
    return out


def sns(h):
    return int(h.stamp.sec) * NS + int(h.stamp.nanosec)


class ATS:
    """Эмуляция message_filters.ApproximateTimeSynchronizer для двух очередей."""

    def __init__(self, queue_size=100, slop=0.05):
        self.qs = queue_size
        self.slop = int(slop * NS)
        self.q = [{}, {}]
        self.pairs = []

    def add(self, idx, stamp, msg):
        my = self.q[idx]
        my[stamp] = msg
        while len(my) > self.qs:
            del my[min(my)]
        other = self.q[1 - idx]
        cands = sorted((abs(s - stamp), s) for s in other if abs(s - stamp) <= self.slop)
        for d, s in cands:
            if d < self.slop and stamp in my and s in other:
                a, b = (my[stamp], other[s]) if idx == 0 else (other[s], my[stamp])
                self.pairs.append((a, b))
                del my[stamp]
                del other[s]
                break


def pseudo_reference(msgs, cfg):
    """Эталон из GNSS для записей без /localization/kinematic_state (только отладка)."""
    proj = make_projection(cfg)
    m_off = cfg['antenna_master_xyz']
    last = {}
    ref = []
    for t, kind, m in msgs:
        if kind not in ('master_fix', 'rover_fix'):
            continue
        if m.status.status != 2:      # только RTK — иначе псевдо-эталон сам ошибается на метры
            continue
        p = proj.to_map(m.latitude, m.longitude, m.altitude)
        if p is None:
            continue
        last[kind] = (sns(m.header), p)
        if kind == 'master_fix' and 'rover_fix' in last:
            ts_r, pr = last['rover_fix']
            ts_m, pm = last['master_fix']
            if abs(ts_r - ts_m) > 0.15 * NS:
                continue
            dx, dy = pr[0] - pm[0], pr[1] - pm[1]
            d = math.hypot(dx, dy)
            if abs(d - 12.436) > 1.5:
                continue
            c, s = dx / d, dy / d
            x = pm[0] - m_off[0] * c
            y = pm[1] - m_off[0] * s
            z = pm[2] - m_off[2]
            ref.append((t, ts_m, x, y, z, None))
    return ref


class SlipInjector:
    """Синтетическое проскальзывание: при тяге/торможении показания тележек искажаются.

    Событие: длительность 2–5 с, скольжение нарастает до s_max (боксование +10…40 % при тяге,
    юз −10…40 % при торможении) и спадает; с вероятностью 50 % — обе тележки, иначе одна.
    """

    def __init__(self, seed=1, every=45.0):
        import random
        self.rnd = random.Random(seed)
        self.every = every
        self.next_t = None
        self.ev = None   # (t0, dur, smax, which)
        self.cmd = 0
        self.n = 0

    def cmd_update(self, c):
        self.cmd = c

    def apply(self, t, kind, v_raw):
        if self.next_t is None:
            self.next_t = t + 30.0
        if self.ev is None and t >= self.next_t and self.cmd != 0 and v_raw > 5.0:
            dur = self.rnd.uniform(2.0, 5.0)
            smax = self.rnd.uniform(0.1, 0.4) * (1 if self.cmd > 0 else -1)
            which = self.rnd.choice(['both', 'front', 'rear'])
            self.ev = (t, dur, smax, which)
            self.n += 1
        if self.ev is not None:
            t0, dur, smax, which = self.ev
            u = (t - t0) / dur
            if u >= 1.0:
                self.ev = None
                self.next_t = t + self.every * (0.5 + self.rnd.random())
            elif which in ('both', kind):
                shape = min(1.0, 3 * u, 3 * (1 - u))
                return v_raw * (1.0 + smax * shape)
        return v_raw


def run(bag, cfg, verbose=False, slip=None, gaps=None):
    """gaps=(каждые_с, длительность_с): выпадение колёсных датчиков (контроллер идёт)."""
    msgs = read_bag(bag)
    logs = []
    core = OdometryCore(cfg, log=lambda lvl, msg: logs.append((lvl, msg)))
    outs = []           # (arrival_ns, stamp_ns, Output)
    refs = []           # (arrival_ns, stamp_ns, x, y, z, v)
    gnss_closed_at = None
    t_first = msgs[0][0] if msgs else 0
    next_timer = t_first
    for t, kind, m in msgs:
        wall = t / NS
        # таймер ноды (прогноз в пропусках) — каждые 20 мс по «стенным» часам
        while next_timer < t:
            o = core.gap_fill(next_timer / NS)
            if o is not None:
                outs.append((next_timer, o.stamp_ns, o))
            if core.gnss_open and core.gnss_should_close():
                core.close_gnss_window(next_timer / NS)
                gnss_closed_at = next_timer
            next_timer += 20_000_000
        if gaps is not None and kind in ('front', 'rear'):
            rel = (t - t_first) / NS
            if rel > 30.0 and (rel % gaps[0]) < gaps[1]:
                continue
        if kind in ('front', 'rear'):
            v_raw = m.velocity if slip is None else slip.apply(t / NS, kind, m.velocity)
            o = core.on_input(kind, sns(m.header), v_raw, wall)
        elif kind == 'cmd':
            if slip is not None:
                slip.cmd_update(m.position)
            o = core.on_input('cmd', sns(m.header), m.position, wall)
        elif kind.endswith('_fix'):
            o = None
            if core.gnss_open or core.midroute:
                core.on_gnss_fix(kind[:-4], sns(m.header), m.latitude, m.longitude, m.altitude,
                                 m.status.status, m.position_covariance, wall)
        elif kind.endswith('_vel'):
            o = None
            if core.gnss_open:
                v = m.twist.linear
                core.on_gnss_vel(kind[:-4], sns(m.header), v.x, v.y, v.z, wall)
        elif kind == 'ref':
            p = m.pose.pose.position
            refs.append((t, sns(m.header), p.x, p.y, p.z, m.twist.twist.linear.x))
            continue
        else:
            continue
        if o is not None:
            outs.append((t + 200_000, o.stamp_ns, o))  # ~0.2 мс обработка
        if core.gnss_open and core.gnss_should_close():
            core.close_gnss_window(wall)
            gnss_closed_at = t
    ref_kind = 'kinematic_state'
    if not refs:
        ref_kind = 'gnss_pseudo'
        refs = pseudo_reference(msgs, cfg)
    return core, outs, refs, ref_kind, logs, gnss_closed_at, msgs


def metrics(outs, refs, ref_kind):
    # поток событий в порядке прихода к судье
    ev = [(a, 0, s, ('ref', x, y, z, v)) for a, s, x, y, z, v in refs]
    ev += [(a, 1, s, o) for a, s, o in outs]
    ev.sort(key=lambda e: (e[0], e[1]))
    vel = ATS()
    pos = ATS()
    for a, idx, s, payload in ev:
        if idx == 0:
            vel.add(0, s, (s, payload))
            pos.add(0, s, (s, payload))
        else:
            o = payload
            if o.publish_velocity:
                vel.add(1, s, (s, o))
            if o.publish_position:
                pos.add(1, s, (s, o))
    res = {'ref_kind': ref_kind}
    ev_ = [(o.velocity - r[4]) for (sr, r), (so, o) in vel.pairs if r[4] is not None]
    if ev_:
        res['vel_rmse'] = math.sqrt(sum(e * e for e in ev_) / len(ev_))
        res['vel_max'] = max(abs(e) for e in ev_)
        res['vel_bias'] = sum(ev_) / len(ev_)
        res['vel_n'] = len(ev_)
    dx, dy, dz, dd = [], [], [], []
    series = []
    for (sr, r), (so, o) in pos.pairs:
        ex, ey, ez = o.x - r[1], o.y - r[2], o.z - r[3]
        dx.append(ex)
        dy.append(ey)
        dz.append(ez)
        dd.append(math.sqrt(ex * ex + ey * ey + ez * ez))
        series.append((sr, dd[-1], ex, ey, ez))
    if dd:
        def rm(a):
            return math.sqrt(sum(e * e for e in a) / len(a))
        res.update({'pos_rmse_x': rm(dx), 'pos_rmse_y': rm(dy), 'pos_rmse_z': rm(dz),
                    'pos_rmse_3d': rm(dd), 'pos_max_3d': max(dd), 'pos_mean_3d': sum(dd) / len(dd),
                    'pos_final_3d': dd[-1], 'pos_n': len(dd)})
    res['n_ref'] = len(refs)
    res['n_out_vel'] = sum(1 for _, _, o in outs if o.publish_velocity)
    res['n_out_pos'] = sum(1 for _, _, o in outs if o.publish_position)
    res['n_pred'] = sum(1 for _, _, o in outs if o.source == 'predict')
    return res, series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--set', action='append', default=[], help='key=value (в config или coefficients)')
    ap.add_argument('--series', default='', help='csv с ошибкой положения во времени')
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('--slip', type=int, default=0, help='синтетическое проскальзывание (seed>0)')
    ap.add_argument('--gaps', default='', help='выпадение колёсных датчиков: каждые,длительность (с)')
    args = ap.parse_args()
    cfg = default_cfg()
    for kv in args.set:
        k, _, v = kv.partition('=')
        try:
            val = json.loads(v)
        except ValueError:
            val = v
        set_value(cfg, k, val)
    allres = []
    for b in args.bags:
        inj = SlipInjector(args.slip) if args.slip else None
        gp = tuple(float(x) for x in args.gaps.split(',')) if args.gaps else None
        core, outs, refs, kind, logs, gclose, msgs = run(b, cfg, args.verbose, slip=inj, gaps=gp)
        res, series = metrics(outs, refs, kind)
        res['bag'] = os.path.basename(os.path.normpath(b))
        if inj is not None:
            res['slip_events'] = inj.n
        res['gnss_closed_after_sec'] = core.gnss_closed_info['since_start_sec'] \
            if core.gnss_closed_info else None
        res['rejected'] = {k: v for k, v in core.counters.items()
                           if not k.endswith(('_rx', '_ok'))}
        allres.append(res)
        print(json.dumps(res, ensure_ascii=False))
        if args.verbose:
            for lvl, msg in logs[:30]:
                print('  [%s] %s' % (lvl, msg))
        if args.series:
            with open(args.series, 'w') as fh:
                fh.write('stamp,err3d,ex,ey,ez\n')
                for row in series:
                    fh.write('%d,%.3f,%.3f,%.3f,%.3f\n' % row)
    return allres


if __name__ == '__main__':
    main()
