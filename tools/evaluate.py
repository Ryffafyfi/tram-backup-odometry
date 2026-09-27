"""Подсчёт ошибок решения, как у жюри: ответы решения сравниваются с эталоном GNSS (ref.csv).

Ответ решения — CSV на запись: <папка>/<запись>.csv со столбцами t, v, x, y, z
  t — header.stamp ответа, с (время входного сообщения), v — скорость, м/с,
  x, y, z — base_link в координатах карты, м.
Сопоставление — по ближайшему времени с допуском 0,05 с (как у судьи), только точки эталона с ok = 1.

Запуск:
  python tools/evaluate.py out/                    все записи из папки out/
  python tools/evaluate.py out/ 30618_01f73500     одна запись
Результат — таблица в консоли и <папка>/metrics.csv: строка на запись и два итога —
«итого 30618 (как у судьи)» (судья проверяет только трамвай 30618) и «итого все с надёжным эталоном».
Записи с ref_reliable = 0 (GNSS смещён относительно карты, gnss_offset_m > 1 м) в итоги не входят.

Метрики:
  matched       доля точек эталона, для которых нашёлся ответ в пределах 0,05 с. При ответах 20–50 Гц
                должно быть ~100 %; при ~10 Гц (только на сообщения тележек) нормально 85–95 %.
                Сильно меньше — неправильный t в ответах
  v_rmse, v_mae, v_bias            скорость, м/с; bias = среднее (ответ − эталон)
  v_bias_accel / _brake / _stop    смещение на разгоне (a > 0,3 м/с²), торможении (a < −0,3), стоянке
  pos_mean, pos_rmse, pos_max      3D-ошибка положения, м
  x_rmse, y_rmse, z_rmse           ошибки по осям, м
  along_rmse, along_max            ошибка вдоль пути (разница в s), м — где эталон на линии карты
  cross_rmse                       ошибка поперёк пути, м
  drift_pct                        ошибка положения в конце записи / пройденный путь × 100
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from track import DIRECTIONS, Track

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / 'data' / 'runs'
TOL = 0.05
JUDGE_VEHICLE = 30618   # судья проверяет только этот трамвай
MAX_OFFSET = 1.0        # м: у записей с gnss_offset_m больше этого эталон ненадёжен


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan


def path_length(ref):
    """Пройденный путь по скорости эталона: все строки со скоростью, шаги по времени до 1 с."""
    r = ref.dropna(subset=['speed']).sort_values('t')
    dt = np.diff(r.t)
    v = (r.speed.to_numpy()[1:] + r.speed.to_numpy()[:-1]) / 2
    return float(np.sum((dt * v)[dt < 1.0]))


def evaluate_run(out, ref, line):
    path = path_length(ref)
    ref = ref[ref.ok == 1].sort_values('t').reset_index(drop=True)
    out = out.dropna(subset=['t']).sort_values('t').reset_index(drop=True)
    m = pd.merge_asof(ref, out[['t', 'v', 'x', 'y', 'z']].rename(columns=lambda c: c if c == 't' else f'{c}_est'),
                      on='t', tolerance=TOL, direction='nearest')
    res = {'n_ref': len(ref), 'matched': float(m.x_est.notna().mean()) if len(m) else 0.0}
    mv = m.dropna(subset=['v_est', 'speed'])
    dv = mv.v_est - mv.speed
    acc = np.gradient(mv.speed.rolling(5, center=True, min_periods=1).mean(), mv.t) if len(mv) > 1 else np.array([])
    res.update(v_rmse=rmse(dv), v_mae=float(dv.abs().mean()), v_bias=float(dv.mean()),
               v_bias_accel=float(dv[acc > 0.3].mean()), v_bias_brake=float(dv[acc < -0.3].mean()),
               v_bias_stop=float(dv[mv.speed < 0.1].mean()))
    mp = m.dropna(subset=['x_est', 'y_est', 'z_est'])
    ex, ey, ez = mp.x_est - mp.x, mp.y_est - mp.y, mp.z_est - mp.z
    e3 = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
    res.update(pos_mean=float(e3.mean()), pos_rmse=rmse(e3), pos_max=float(e3.max()) if len(e3) else np.nan,
               x_rmse=rmse(ex), y_rmse=rmse(ey), z_rmse=rmse(ez))
    on = mp[mp.on_map == 1]
    if line is not None and len(on):
        s_est, c_est = line.project(on.x_est, on.y_est)
        res.update(along_rmse=rmse(s_est - on.s), along_max=float(np.abs(s_est - on.s).max()),
                   cross_rmse=rmse(c_est - on.cross))
    res['path_m'] = path
    res['drift_pct'] = float(e3.iloc[-1] / path * 100) if len(e3) and path > 0 else np.nan
    return res


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'out'
    runs = sys.argv[2:] or sorted(p.stem for p in out_dir.glob('*.csv') if p.stem != 'metrics')
    index = pd.read_csv(RUNS_DIR / 'index.csv').set_index('run')
    lines = {k: Track.load(k) for k in DIRECTIONS}
    rows = []
    for run in runs:
        out_path = out_dir / f'{run}.csv'
        if not out_path.exists():
            print(f'{run}: нет файла ответа {out_path} — пропускаю')
            continue
        if run not in index.index:
            print(f'{run}: такой записи нет в data/runs — это копия другой записи (см. «Копии» в README) '
                  f'или опечатка в имени — пропускаю')
            continue
        ref_path = RUNS_DIR / run / 'ref.csv'
        if not ref_path.exists():
            print(f'{run}: в записи нет GNSS, эталона нет — пропускаю')
            continue
        out = pd.read_csv(out_path)
        missing = [c for c in ('t', 'v', 'x', 'y', 'z') if c not in out.columns]
        if missing:
            print(f'{run}: в {out_path.name} нет столбцов {", ".join(missing)} '
                  f'(нужны t, v, x, y, z; есть {", ".join(out.columns)}) — пропускаю')
            continue
        key = index.direction_key.get(run)
        line = lines.get(key) if isinstance(key, str) else None
        offset = index.gnss_offset_m.get(run)
        rows.append({'run': run, 'vehicle': index.vehicle.get(run),
                     'ref_reliable': int(not (offset > MAX_OFFSET)),
                     **evaluate_run(out, pd.read_csv(ref_path), line)})
    if not rows:
        sys.exit('нечего считать: в папке нет ответов с эталоном')
    table = pd.DataFrame(rows)
    judge = table[(table.vehicle == JUDGE_VEHICLE) & (table.ref_reliable == 1)]
    totals = [{'run': f'итого {JUDGE_VEHICLE} (как у судьи), записей: {len(judge)}',
               **judge.drop(columns=['run', 'vehicle']).mean(numeric_only=True)},
              {'run': f'итого все с надёжным эталоном, записей: {table.ref_reliable.sum()}',
               **table[table.ref_reliable == 1].drop(columns=['run', 'vehicle']).mean(numeric_only=True)}]
    table = pd.concat([table, pd.DataFrame(totals)], ignore_index=True)
    table.to_csv(out_dir / 'metrics.csv', index=False, float_format='%.4f', encoding='utf-8-sig')  # BOM — для Excel
    show = ['run', 'ref_reliable', 'matched', 'v_rmse', 'v_bias', 'pos_mean', 'pos_rmse', 'pos_max',
            'along_rmse', 'cross_rmse', 'drift_pct']
    with pd.option_context('display.width', 220, 'display.max_columns', 20, 'display.max_colwidth', 60):
        print(table[show].round(3).to_string(index=False))
    if (table.ref_reliable == 0).any():
        print(f'\nref_reliable = 0: GNSS записи смещён относительно карты больше чем на {MAX_OFFSET} м '
              f'(gnss_offset_m в index.csv) — ошибки по ней не показательны и в итог не входят.')
    print(f'\nподробно: {out_dir / "metrics.csv"}')


if __name__ == '__main__':
    main()
