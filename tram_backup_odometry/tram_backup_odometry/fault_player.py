#!/usr/bin/env python3
"""Воспроизведение bag с внесёнными сбоями — проверка живучести ноды (критерий 3).

    ros2 run tram_backup_odometry fault_player <bag_dir> [--rate 1.0] [--seed 1] [--profile hard]

Публикует все топики записи в реальном времени (как ros2 bag play), но во входные топики
/vehicle/* вносит сбои:
  * NaN / inf / отрицательные / огромные значения скорости;
  * нулевой header.stamp, повторы сообщений, перестановка соседних сообщений;
  * сбойные stamp (+1.2 c вперёд), выбросы скорости (x3 — «боксование» одного датчика);
  * «залипание» задней тележки на 0 (отказ датчика) на 5 с;
  * пропуски всех входов на 0.5–3 с (нода должна прогнозировать, без дыр для судьи);
  * положение контроллера вне диапазона.
Эталон /localization/kinematic_state и GNSS не трогаются — судья считает метрики как обычно.
В конце печатает, сколько сбоев каждого вида внесено.
"""

import argparse
import copy
import random
import time
from collections import Counter

import rclpy
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message

PROFILES = {
    # вероятности на сообщение и параметры окон
    'soft': dict(nan=0.002, zero_stamp=0.002, dup=0.005, swap=0.005, glitch=0.001, spike=0.002,
                 cmd_bad=0.002, gap_every=120.0, gap_len=(0.3, 1.0), stuck_every=0.0),
    'hard': dict(nan=0.01, zero_stamp=0.005, dup=0.01, swap=0.01, glitch=0.003, spike=0.005,
                 cmd_bad=0.005, gap_every=60.0, gap_len=(0.5, 3.0), stuck_every=180.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--rate', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--profile', default='hard', choices=sorted(PROFILES))
    ap.add_argument('--start-delay', type=float, default=1.0)
    args, ros_args = ap.parse_known_args()
    prof = PROFILES[args.profile]
    rnd = random.Random(args.seed)

    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message

    rclpy.init(args=ros_args)
    node = rclpy.create_node('fault_player')
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                     depth=100, durability=DurabilityPolicy.VOLATILE)
    pubs, classes = {}, {}
    for name, typ in types.items():
        try:
            classes[name] = get_message(typ)
        except (AttributeError, ModuleNotFoundError, ValueError) as e:
            node.get_logger().warn('skip %s (%s): %r' % (name, typ, e))
            continue
        pubs[name] = node.create_publisher(classes[name], name, qos)
    time.sleep(args.start_delay)

    stats = Counter()
    t_bag0 = None
    t_wall0 = None
    gap_until = -1.0
    next_gap = prof['gap_every'] * (0.5 + rnd.random()) if prof['gap_every'] else 1e18
    stuck_until = -1.0
    next_stuck = prof['stuck_every'] * (0.5 + rnd.random()) if prof['stuck_every'] else 1e18
    held = None  # сообщение, отложенное для перестановки

    def publish(topic, msg):
        pubs[topic].publish(msg)
        stats['published'] += 1

    while reader.has_next() and rclpy.ok():
        topic, data, t_ns = reader.read_next()
        if topic not in pubs:
            continue
        t_bag = t_ns * 1e-9
        if t_bag0 is None:
            t_bag0, t_wall0 = t_bag, time.monotonic()
        rel = t_bag - t_bag0
        # темп как у ros2 bag play
        delay = t_wall0 + rel / args.rate - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        msg = deserialize_message(data, classes[topic])
        if not topic.startswith('/vehicle/'):
            publish(topic, msg)
            continue
        # --- окна отказов (после 20 c — чтобы GNSS-старт прошёл нормально) ---
        if rel > 20.0 and rel >= next_gap:
            gap_until = rel + rnd.uniform(*prof['gap_len'])
            next_gap = rel + prof['gap_every'] * (0.5 + rnd.random())
            stats['gap_windows'] += 1
        if rel < gap_until:
            stats['dropped_in_gap'] += 1
            continue
        if rel > 20.0 and rel >= next_stuck:
            stuck_until = rel + 5.0
            next_stuck = rel + prof['stuck_every'] * (0.5 + rnd.random())
            stats['stuck_windows'] += 1
        is_vel = topic.endswith('_velocity')
        if is_vel and topic.endswith('rear_bogie_velocity') and rel < stuck_until:
            msg.velocity = 0.0
            stats['stuck_rear_zero'] += 1
        r = rnd.random()
        p = 0.0
        if is_vel:
            for kind in ('nan', 'spike'):
                p += prof[kind]
                if r < p:
                    if kind == 'nan':
                        msg.velocity = rnd.choice([float('nan'), float('inf'), -float('inf'),
                                                   -5.0, 1e6])
                        stats['bad_value'] += 1
                    else:
                        msg.velocity = msg.velocity * 3.0 + 10.0
                        stats['spike'] += 1
                    break
        elif rnd.random() < prof['cmd_bad']:
            msg.position = rnd.choice([-128, 127, 42, -99])
            stats['cmd_out_of_range'] += 1
        r = rnd.random()
        if r < prof['zero_stamp']:
            msg.header.stamp.sec = 0
            msg.header.stamp.nanosec = 0
            stats['zero_stamp'] += 1
        elif r < prof['zero_stamp'] + prof['glitch']:
            msg.header.stamp.sec += 1
            msg.header.stamp.nanosec = (msg.header.stamp.nanosec + 200_000_000) % 1_000_000_000
            stats['stamp_glitch'] += 1
        # перестановка соседних
        if held is not None:
            publish(*held)
            held = None
        if rnd.random() < prof['swap']:
            held = (topic, msg)
            stats['swapped'] += 1
            continue
        publish(topic, msg)
        if rnd.random() < prof['dup']:
            publish(topic, copy.deepcopy(msg))
            stats['duplicate'] += 1
    if held is not None:
        publish(*held)
    node.get_logger().info('fault injection summary: %s' % dict(stats))
    print('FAULTS', dict(stats), flush=True)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
