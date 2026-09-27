"""Выгрузка записей в CSV: data/runs/<запись>/*.csv (копии записей пропускаются).

front.csv, rear.csv       t, t_bag, v            — скорость тележки, м/с (в bag км/ч, здесь уже ÷ 3,6)
notch.csv                 t, t_bag, position     — позиция ручки, от −15 до +15
gnss_rover.csv,           t, t_bag, lat, lon, alt, status, x, y, z  — x, y, z антенны в координатах карты
gnss_master.csv
gnss_rover_vel.csv,       t, t_bag, vx, vy, vz, speed  — vx на восток, vy на север, speed = |v| с vz
gnss_master_vel.csv
t — время измерения (header.stamp), t_bag — время записи в bag; оба в секундах Unix.
Строки — в порядке записи в bag (как их получит нода при ros2 bag play).
Запуск: python tools/export_runs.py   (нужны dataset/parquet и analysis/out/catalog.csv)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geo import lla_to_map_np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from catalog import KMH, OUT_DIR, load  # noqa: E402

RUNS_DIR = ROOT / 'data' / 'runs'
FILES = {'front': 'front', 'rear': 'rear', 'cmd': 'notch', 'rover_fix': 'gnss_rover',
         'master_fix': 'gnss_master', 'rover_vel': 'gnss_rover_vel', 'master_vel': 'gnss_master_vel'}


def times(df):
    return pd.DataFrame({'t': df.t_hdr / 1e9, 't_bag': df.t_bag / 1e9})


def export(bag):
    out = RUNS_DIR / bag
    out.mkdir(parents=True, exist_ok=True)
    for topic, df in load(bag).items():
        if not len(df):
            continue
        res = times(df)
        if topic in ('front', 'rear'):
            res['v'] = df.velocity / KMH
        elif topic == 'cmd':
            res['position'] = df.position
        elif topic.endswith('_fix'):
            res[['lat', 'lon', 'alt', 'status']] = df[['lat', 'lon', 'alt', 'status']]
            res['x'], res['y'], res['z'] = lla_to_map_np(df.lat, df.lon, df.alt)
        else:
            res[['vx', 'vy', 'vz']] = df[['vx', 'vy', 'vz']]
            res['speed'] = np.sqrt(df.vx ** 2 + df.vy ** 2 + df.vz ** 2)
        res.to_csv(out / f'{FILES[topic]}.csv', index=False, float_format='%.9f')


def main():
    cat = pd.read_csv(OUT_DIR / 'catalog.csv', keep_default_na=False, na_values=[''])
    bags = cat[cat.duplicate_of.isna()].bag
    for i, bag in enumerate(bags, 1):
        export(bag)
        print(f'[{i}/{len(bags)}] {bag}', flush=True)


if __name__ == '__main__':
    main()
