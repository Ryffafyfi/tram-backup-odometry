# Офлайн-инструменты пакета

Работают без ROS, нужны `rosbags`, `numpy`, `pyyaml` (для графиков ещё `matplotlib`). Ноде они
не нужны и в сборку не входят. Запускать из папки `tram_backup_odometry/`.

Проверка решения:

- `offline_eval.py <запись> [--slip N] [--gaps a,b] [--set ключ=значение] [--series err.csv]` —
  прогон записи через ту же логику, что в ноде, с метриками как у судьи. Эталон —
  `/localization/kinematic_state`, если он есть в записи, иначе RTK GNSS самой записи.
  `--slip` добавляет искусственное проскальзывание, `--gaps` — пропуски колёсных данных,
  `--set estimator_class=модуль:Класс` подключает другой оценщик.
- `export_answers.py <папка> <запись> ...` — ответы в CSV `t, v, x, y, z` для
  `tools/evaluate.py` в корне репозитория.
- `check_stamps.py <запись> <записанные ответы>` — проверка, что stamp каждого ответа совпадает
  со stamp входа (ответы записываются через `ros2 bag record /result/velocity /result/position`).
- `plots.py <запись> <папка>` — графики для `docs/img`.

Построение карты и калибровки (по обучающим записям):

- `build_loops.py` — разворотные кольца у конечных по RTK-трекам, `maps/terminal_loops.json`;
- `add_track_from_csv.py` — добавить путь из линии `maps/track_*.csv`, так добавлен путь
  отправления `tallinskaya_depart`;
- `fix_lower_z.py` — поправка высот нижнего пути «Таллинской»;
- `build_branch_track.py` — средний путь «Таллинской», `maps/tallinskaya_middle.json`;
- `stop_landmarks.py` — места остановок, `maps/stop_landmarks.json`;
- `calib_scale.py` — масштаб колёс по RTK и карте;
- `fit_traction_table.py`, `fit_traction_grade.py` — таблица модели тяги (вторая с вычетом
  уклона);
- `fit_traction_envelope.py` — коридор разброса ускорения для детектора проскальзывания.

Пример:

```bash
cd tram_backup_odometry
python3 tools/offline_eval.py /путь/к/записи/30618_88aea4d9
```
