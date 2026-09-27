#!/usr/bin/env python3
"""Ответы ноды по записям (та же логика, офлайн) в формате tools/evaluate.py: t, v, x, y, z.

    python3 tools/export_answers.py <папка_ответов> <запись> [<запись> ...] [--set ключ=значение]

Для каждой записи пишется <папка_ответов>/<имя_записи>.csv (t — header.stamp ответа, с; v — скорость,
м/с; x, y, z — base_link в системе карты; до GNSS-привязки x, y, z пустые). Дальше подсчёт ошибок
по эталону ref.csv (из корня репозитория):  python3 tools/evaluate.py <папка_ответов>
Для записей трамвая 30639 автоматически выбирается его калибровка колёс (vehicle_id).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import offline_eval as oe  # noqa: E402

NS = 1_000_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--set', action='append', default=[], help='ключ=значение (config или coefficients)')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for bag in a.bags:
        name = os.path.basename(os.path.normpath(bag))
        cfg = oe.default_cfg()
        if name.startswith('30639'):
            cfg['vehicle_id'] = '30639'
        for kv in a.set:
            k, _, v = kv.partition('=')
            try:
                val = json.loads(v)
            except ValueError:
                val = v
            oe.set_value(cfg, k, val)
        core, outs, refs, kind, logs, gclose, msgs = oe.run(bag, cfg)
        path = os.path.join(a.out_dir, name + '.csv')
        with open(path, 'w') as fh:
            fh.write('t,v,x,y,z\n')
            for _, stamp, o in outs:
                v = o.velocity if o.publish_velocity and math.isfinite(o.velocity) else float('nan')
                if o.publish_position:
                    fh.write('%.4f,%.4f,%.4f,%.4f,%.4f\n' % (stamp / NS, v, o.x, o.y, o.z))
                else:
                    fh.write('%.4f,%.4f,,,\n' % (stamp / NS, v))
        print('%s: %d ответов -> %s' % (name, len(outs), path), flush=True)


if __name__ == '__main__':
    main()
