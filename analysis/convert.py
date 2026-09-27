"""Конвертация всех rosbag2 в parquet: dataset/parquet/<bag_id>/<топик>.parquet.

Для GNSS fix добавляются локальные метры ENU (e, n, u) относительно первой точки master
с ненулевыми координатами (если master пуст — rover); для GNSS vel — горизонтальная скорость speed = hypot(vx, vy).
Запуск: python analysis/convert.py
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bagio import PARQUET_DIR, bag_dirs, geodetic_to_enu, read_bag


def convert_one(bag_dir):
    out = PARQUET_DIR / bag_dir.name
    out.mkdir(parents=True, exist_ok=True)
    topics = read_bag(bag_dir)
    fix = topics['master_fix'] if len(topics['master_fix']) else topics['rover_fix']
    valid = fix[(fix.lat != 0) & (fix.lon != 0)] if len(fix) else fix
    origin = tuple(valid.iloc[0][['lat', 'lon', 'alt']]) if len(valid) else None
    for short, df in topics.items():
        if short.endswith('_fix') and origin is not None and len(df):
            df['e'], df['n'], df['u'] = geodetic_to_enu(df.lat, df.lon, df.alt, *origin)
        if short.endswith('_vel') and len(df):
            df['speed'] = np.hypot(df.vx, df.vy)
        df.to_parquet(out / f'{short}.parquet', index=False)
    return bag_dir.name, {k: len(v) for k, v in topics.items()}


def main():
    dirs = bag_dirs()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, (name, counts) in enumerate(pool.map(convert_one, dirs), 1):
            print(f'[{i}/{len(dirs)}] {name} {counts}', flush=True)
    print(f'готово за {time.time() - t0:.0f} с', file=sys.stderr)


if __name__ == '__main__':
    main()
