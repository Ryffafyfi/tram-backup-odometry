#!/usr/bin/env python3
"""Проверка записанных ответов: stamp каждого ответа = stamp одного из входов (или прогноз).

    ros2 bag record -o out /result/velocity /result/position     # во время прогона
    python3 tools/check_stamps.py <входная_запись> out

Печатает: сколько ответов, какая доля совпала со stamp входов, сколько прогнозов (stamp не
совпал ни с одним входом — публикуется только при пропаже входов), частоту, максимальный
разрыв по времени приёма, frame_id и child_frame_id.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from offline_eval import read_bag, sns  # noqa: E402


def read_out(path):
    from rosbags.highlevel import AnyReader
    from offline_eval import VS, DCC
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    ts = get_typestore(Stores.ROS2_HUMBLE)
    add = {}
    add.update(get_types_from_msg(VS, 'tram_vehicle_msgs/msg/VelocitySensor'))
    add.update(get_types_from_msg(DCC, 'tram_vehicle_msgs/msg/DriverControllerCommand'))
    ts.register(add)
    out = {}
    with AnyReader([Path(path)], default_typestore=ts) as r:
        for c, t, raw in r.messages(connections=r.connections):
            m = r.deserialize(raw, c.msgtype)
            out.setdefault(c.topic, []).append((t, m))
    return out


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    stamps = {sns(m.header) for _, k, m in read_bag(inp) if k in ('front', 'rear', 'cmd')}
    res = read_out(outp)
    for topic in ('/result/velocity', '/result/position'):
        lst = res.get(topic, [])
        if not lst:
            print(topic, 'НЕТ СООБЩЕНИЙ')
            continue
        st = [sns(m.header) for _, m in lst]
        match = sum(1 for s in st if s in stamps)
        tr = [t for t, _ in lst]
        gaps = [(b - a) * 1e-9 for a, b in zip(tr, tr[1:])]
        span = (tr[-1] - tr[0]) * 1e-9 if len(tr) > 1 else 0
        m0 = lst[0][1]
        frames = m0.header.frame_id + ('/' + m0.child_frame_id if hasattr(m0, 'child_frame_id') else '')
        zero = sum(1 for s in st if s == 0)
        print('%s: %d msgs, stamp = stamp входа: %d (%.1f%%), прогнозов: %d, нулевых stamp: %d, '
              '%.1f Гц, макс. разрыв %.3f c, frame %s' % (
                  topic, len(lst), match, 100.0 * match / len(lst), len(lst) - match, zero,
                  (len(lst) - 1) / span if span > 0 else 0, max(gaps) if gaps else 0, frames))


if __name__ == '__main__':
    main()
