"""Эталон («правильный ответ») по GNSS в точке base_link: data/runs/<запись>/ref.csv + data/runs/index.csv.

ref.csv — строка на каждое сообщение rover/fix (если rover нет — master):
  t          время измерения, с
  x, y, z    base_link в координатах карты (z — уровень рельса)
  speed      модуль скорости GNSS, м/с (ближайшее сообщение vel той же антенны)
  heading    курс вагона, рад: от master к rover; без второй антенны — по скорости GNSS на ходу,
             по карте на линии, на стоянке — ближайший известный (см. fill_heading)
  s, cross   метры вдоль линии своего направления (с кольцами) и отклонение вбок, м
  ok         1 — точка годится как эталон: status 2, база антенн ~12,44 м (если есть обе),
             нет скачка относительно соседей
  on_map     1 — точка лежит на линии своего направления (|cross| < 1 м); вне карты (стоянка
             на соседнем пути кольца, съезд) s и cross не имеют смысла
Настоящий эталон жюри — /localization/kinematic_state (GNSS + лидар), его в bag нет;
этот — приближение по сырому GNSS.
Запуск: python tools/reference.py   (после export_runs.py и build_map.py)
"""
import numpy as np
import pandas as pd

from export_runs import OUT_DIR, RUNS_DIR
from to_excel import purpose_and_notes
from geo import ANTENNA_BASELINE, ANTENNAS
from track import DIRECTIONS, Track, detect_direction

TOL = 0.06          # с: допуск сопоставления сообщений разных топиков
BASE_TOL = 0.5      # м: допустимое отклонение базы антенн от 12,436 м
JUMP = 1.0          # м: допустимое расхождение шага точки с шагом по скорости
CROSS_OK = 1.0      # м: дальше от линии — точка вне карты или плохая
HOLD_S = 60.0       # с: на стоянке курс берётся из ближайшего известного момента не дальше этого


def read(run, name):
    p = RUNS_DIR / run / f'{name}.csv'
    return pd.read_csv(p).sort_values('t').reset_index(drop=True) if p.exists() else None


def fill_heading(df, heading, line):
    """Курс там, где нет надёжной второй антенны. По порядку:
    1) направление скорости GNSS, если трамвай едет (> 1 м/с) — он едет только вперёд;
    2) направление карты, если антенна лежит на линии (|cross| < 1 м) — вне карты ближайшая
       точка линии может оказаться на соседней ветке кольца с обратным курсом;
    3) ближайший по времени известный курс не дальше HOLD_S — на стоянке курс не меняется.
    Остальное — NaN (такие точки получают ok = 0)."""
    h = heading.copy()
    moving = (df.speed > 1).to_numpy() & np.isnan(h)
    h[moving] = np.arctan2(df.vy, df.vx).to_numpy()[moving]
    s_ant, c_ant = line.project(df.x, df.y)
    on = (np.abs(c_ant) < CROSS_OK) & np.isnan(h)
    h[on] = line.at(s_ant)[3][on]
    known = pd.Series(h, index=df.t.to_numpy())
    near = pd.merge_asof(df[['t']], known.dropna().rename('h').rename_axis('t').reset_index(),
                         on='t', tolerance=HOLD_S, direction='nearest').h.to_numpy()
    return np.where(np.isnan(h), near, h)


def reference(run, lines):
    ant = 'rover' if read(run, 'gnss_rover') is not None else 'master'
    fix = read(run, f'gnss_{ant}')
    if fix is None:
        return None, None
    fix = fix[fix.status >= 0].reset_index(drop=True)
    vel = read(run, f'gnss_{ant}_vel')
    df = pd.merge_asof(fix, vel[['t', 'vx', 'vy', 'speed']], on='t', tolerance=TOL, direction='nearest') \
        if vel is not None else fix.assign(vx=np.nan, vy=np.nan, speed=np.nan)

    other = read(run, 'gnss_master') if ant == 'rover' else None
    heading = np.full(len(df), np.nan)
    base_ok = np.ones(len(df), bool)
    if other is not None:
        m = pd.merge_asof(df[['t']], other[other.status == 2][['t', 'x', 'y']], on='t',
                          tolerance=0.02, direction='nearest')
        has = m.x.notna().to_numpy()
        base = np.hypot(df.x - m.x, df.y - m.y).to_numpy()
        good = has & (np.abs(base - ANTENNA_BASELINE) < BASE_TOL)
        heading[good] = np.arctan2(df.y - m.y, df.x - m.x)[good]
        base_ok = ~has | good

    found = detect_direction(lines, df.x.to_numpy(), df.y.to_numpy())
    if found is None:
        return None, None
    direction = found[0]
    line = lines[direction]
    heading = fill_heading(df, heading, line)

    dx, _, dz = ANTENNAS[ant]
    bx, by, bz = df.x - dx * np.cos(heading), df.y - dx * np.sin(heading), df.z - dz
    s, cross = line.project(bx, by)

    step = np.hypot(np.diff(bx), np.diff(by))
    expect = np.nan_to_num(df.speed.to_numpy()[1:]) * np.diff(df.t)
    jump = np.r_[False, np.abs(step - expect) > JUMP]
    jump |= np.r_[jump[1:], False]
    ok = (df.status == 2).to_numpy() & base_ok & ~jump & ~np.isnan(heading)

    ref = pd.DataFrame({'t': df.t, 'x': bx, 'y': by, 'z': bz, 'speed': df.speed,
                        'heading': heading, 's': s, 'cross': cross, 'ok': ok.astype(int),
                        'on_map': (np.abs(cross) < CROSS_OK).astype(int)})
    return ref, direction


def gnss_offset(ref):
    """Медиана |cross| на ходу: > 2 м — GNSS записи смещён относительно карты, эталону не верить."""
    if ref is None:
        return None
    moving = ref[(ref.ok == 1) & (ref.speed > 2)]
    return round(float(moving.cross.abs().median()), 2) if len(moving) else None


def main():
    lines = {k: Track.load(k) for k in DIRECTIONS}
    cat = pd.read_csv(OUT_DIR / 'catalog.csv', keep_default_na=False, na_values=[''])
    cat = cat[cat.duplicate_of.isna()].copy()
    rows = []
    for i, r in enumerate(cat.itertuples(), 1):
        ref, direction = reference(r.bag, lines) if pd.notna(r.ref_dist_m) else (None, None)
        if ref is not None:
            ref.to_csv(RUNS_DIR / r.bag / 'ref.csv', index=False, float_format='%.4f')
        _, a, b = DIRECTIONS[direction] if direction else (None, '', '')
        rows.append({'run': r.bag, 'vehicle': r.vehicle,
                     'direction': f'{a} → {b}' if direction else '',
                     'direction_key': direction or '',
                     'duration_min': round(r.duration_s / 60, 1),
                     'gnss_dist_km': round(r.ref_dist_m / 1000, 2) if pd.notna(r.ref_dist_m) else None,
                     'ref_ok_share': round(ref.ok.mean(), 3) if ref is not None else None,
                     'on_map_share': round(ref.on_map.mean(), 3) if ref is not None else None,
                     'gnss_offset_m': gnss_offset(ref)})
        print(f'[{i}/{len(cat)}] {r.bag} {direction or "без эталона"}'
              + (f', ok {ref.ok.mean():.0%}' if ref is not None else ''), flush=True)
    index = pd.DataFrame(rows)
    purpose, notes = purpose_and_notes(cat)
    index['purpose'], index['problems'] = purpose.to_numpy(), notes.to_numpy()
    index.to_csv(RUNS_DIR / 'index.csv', index=False, encoding='utf-8-sig')  # BOM — чтобы Excel понял кириллицу


if __name__ == '__main__':
    main()
