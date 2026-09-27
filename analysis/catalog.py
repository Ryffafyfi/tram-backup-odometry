"""Каталог прогонов: одна строка на bag с качеством данных и основными характеристиками.

Читает dataset/parquet (см. convert.py), пишет analysis/out/catalog.csv.
Время везде — header.stamp (t_hdr), в секундах от начала прогона.
Запуск: python analysis/catalog.py
"""
import hashlib

import numpy as np
import pandas as pd

from bagio import PARQUET_DIR, ROOT

OUT_DIR = ROOT / 'analysis' / 'out'
MOVING = 1.0        # м/с — считаем, что вагон едет
SLIP_DV = 1.0       # м/с — расхождение тележки с эталоном, которое считаем проскальзыванием
SPIKE_DV = 1.0      # м/с за один шаг 0,1 с (= 10 м/с², физически невозможно)
FROZEN_N = 20       # одинаковое ненулевое значение подряд (2 с при 10 Гц)
STOP_V, STOP_T = 0.2, 3.0
KMH = 3.6          # VelocitySensor.velocity в bag записан в км/ч (README ошибочно говорит м/с)


def load(bag):
    d = PARQUET_DIR / bag
    return {p.stem: pd.read_parquet(p) for p in d.glob('*.parquet')}


def gaps(t):
    dt = np.diff(np.sort(t))
    return (dt.max() if len(dt) else np.nan), int((dt > 0.5).sum())


def runs(mask):
    """Интервалы подряд идущих True: список (начало, конец) индексов."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def frozen_runs(v):
    same = np.concatenate([[False], (np.diff(v) == 0) & (v[1:] != 0)])
    return sum(1 for a, b in runs(same) if b - a + 1 >= FROZEN_N)


def integrate(t, v):
    return float(np.sum(np.diff(t) * (v[1:] + v[:-1]) / 2)) if len(t) > 1 else 0.0


def duplicate_key(bag):
    """Отпечаток прогона: одинаковый у записей-копий (по меткам времени и скорости передней тележки)."""
    f = pd.read_parquet(PARQUET_DIR / bag / 'front.parquet')
    return hashlib.md5(f.t_hdr.values.tobytes() + f.velocity.values.tobytes()).hexdigest()


def bag_row(bag):
    d = load(bag)
    t0 = min(df.t_hdr.min() for df in d.values() if len(df))
    for df in d.values():
        df['t'] = (df.t_hdr - t0) / 1e9 if len(df) else pd.Series(dtype=float)
    for k in ('front', 'rear'):
        d[k]['velocity'] = d[k]['velocity'] / KMH
    row = {'bag': bag, 'vehicle': bag.split('_')[0]}
    t_all = pd.concat([df.t for df in d.values() if len(df)])
    row['duration_s'] = t_all.max() - t_all.min()
    for k, df in d.items():
        row[f'n_{k}'] = len(df)

    for k in ('front', 'rear', 'cmd'):
        df = d[k]
        # опоздавшие сообщения: в порядке записи в bag их header.stamp меньше уже пришедших
        late = np.maximum.accumulate(df.t.values) - df.t.values if len(df) else np.array([])
        row[f'{k}_late_msgs'] = int((late > 0).sum())
        row[f'{k}_late_max_s'] = float(late.max()) if len(late) else np.nan
        row[f'{k}_lag_ms'] = ((df.t_bag - df.t_hdr) / 1e6).median()
    for k in d:  # дальше всё считаем в порядке header.stamp
        d[k] = d[k].sort_values('t', kind='stable').reset_index(drop=True)
    for k in ('front', 'rear', 'cmd'):
        df = d[k]
        dt = np.diff(df.t.values)
        row[f'{k}_hz'] = 1 / np.median(dt) if len(dt) else np.nan
        row[f'{k}_max_gap_s'], row[f'{k}_gaps_05s'] = gaps(df.t.values)
    for k in ('front', 'rear'):
        v = d[k].velocity.values
        row[f'{k}_nan'] = int(np.isnan(v).sum())
        row[f'{k}_neg'] = int((v < -0.05).sum())
        ok = np.diff(d[k].t.values) < 0.5  # скачок через пропуск данных — не выброс
        row[f'{k}_spikes'] = int(((np.abs(np.diff(v)) > SPIKE_DV) & ok).sum())
        row[f'{k}_frozen'] = frozen_runs(v)
        row[f'{k}_max'] = np.nanmax(v) if len(v) else np.nan
        row[f'{k}_dist_m'] = integrate(d[k].t.values, np.nan_to_num(v))

    fr = pd.merge_asof(d['front'][['t', 'velocity']].sort_values('t'),
                       d['rear'][['t', 'velocity']].sort_values('t'),
                       on='t', suffixes=('_f', '_r'), tolerance=0.06, direction='nearest')
    mov = fr[[c for c in fr if c.startswith('velocity')]].max(axis=1) > MOVING
    row['front_rear_mismatch_share'] = float(
        ((fr.velocity_f - fr.velocity_r).abs() > 0.5)[mov].mean()) if mov.any() else np.nan

    cmd = d['cmd'].position.values
    row['cmd_min'], row['cmd_max'] = (cmd.min(), cmd.max()) if len(cmd) else (np.nan, np.nan)
    row['cmd_traction_share'] = float((cmd > 0).mean()) if len(cmd) else np.nan
    row['cmd_brake_share'] = float((cmd < 0).mean()) if len(cmd) else np.nan
    row['cmd_changes'] = int((np.diff(cmd) != 0).sum()) if len(cmd) else 0

    ref_src = 'rover' if len(d['rover_vel']) else 'master'  # у rover больше точных решений
    ref = d[f'{ref_src}_vel'].sort_values('t')
    row['ref_src'] = ref_src
    for k in ('master', 'rover'):
        fix = d[f'{k}_fix']
        row[f'{k}_rtk_share'] = float((fix.status == 2).mean()) if len(fix) else np.nan
        row[f'{k}_fix_max_gap_s'] = gaps(fix.t.values)[0] if len(fix) else np.nan
    if len(ref):
        row['ref_max_speed'] = ref.speed.max()
        row['ref_dist_m'] = integrate(ref.t.values, ref.speed.values)
        row['ref_speed_first5s'] = ref.speed[ref.t < ref.t.min() + 5].mean()
        stopped = ref.speed.values < STOP_V
        row['stops'] = sum(1 for a, b in runs(stopped)
                           if ref.t.values[b - 1] - ref.t.values[a] >= STOP_T)
        for k in ('front', 'rear'):
            m = pd.merge_asof(d[k][['t', 'velocity']].sort_values('t'), ref[['t', 'speed']],
                              on='t', tolerance=0.1, direction='nearest').dropna()
            mv = m[m.speed > MOVING]
            row[f'{k}_slip_share'] = float(((mv.velocity - mv.speed).abs() > SLIP_DV).mean()) \
                if len(mv) else np.nan
            clean = m[(m.speed > 3) & ((m.velocity - m.speed).abs() < SLIP_DV)]
            row[f'{k}_scale'] = float((clean.velocity / clean.speed).median()) \
                if len(clean) > 50 else np.nan
    if len(d['master_fix']) and len(d['rover_fix']):
        mr = pd.merge_asof(d['master_fix'][['t', 'e', 'n']].sort_values('t'),
                           d['rover_fix'][['t', 'e', 'n']].sort_values('t'),
                           on='t', tolerance=0.06, direction='nearest').dropna()
        row['antenna_baseline_m'] = float(np.hypot(mr.e_x - mr.e_y, mr.n_x - mr.n_y).median())
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bags = sorted(p.name for p in PARQUET_DIR.iterdir() if p.is_dir())
    cat = pd.DataFrame([bag_row(b) for b in bags])
    cat['short'] = cat.duration_s < 60
    cat['duplicate_of'] = [duplicate_key(b) for b in cat.bag]
    first = cat.groupby('duplicate_of').bag.transform('first')
    cat['duplicate_of'] = np.where(first != cat.bag, first, '')
    cat.to_csv(OUT_DIR / 'catalog.csv', index=False, float_format='%.4g')
    print(f'{len(cat)} прогонов → {OUT_DIR / "catalog.csv"}')


if __name__ == '__main__':
    main()
