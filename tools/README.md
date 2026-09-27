# Работа с данными

Скрипты для разбора записей датасета, построения эталона и подсчёта ошибок. Они работают без ROS,
нужны `rosbags`, `numpy`, `pandas`, `scipy`, `pyarrow` и `openpyxl`. Ноде они не нужны. Данные
ожидаются в `dataset/data` (записи датасета) и в этот репозиторий не входят.

Порядок запуска из корня репозитория:

```bash
python analysis/convert.py      # rosbag2 -> dataset/parquet
python analysis/catalog.py      # сводка по записям -> analysis/out/catalog.csv
python tools/export_runs.py     # записи в CSV -> data/runs/<запись>/*.csv
python tools/build_map.py       # линии пути с кольцами -> tram_backup_odometry/maps/track_*.csv
python tools/reference.py       # эталон по GNSS в точке base_link -> data/runs/<запись>/ref.csv
python tools/evaluate.py out/   # ошибки ответов решения из out/<запись>.csv, как у жюри
```

- `geo.py` — перевод GNSS в координаты карты: UTM 37N минус (300 000, 6 100 000), точность на
  записях 5–9 см.
- `track.py` и `track_map.py` — линия пути: проекция точки, точка по дуговой координате, уклон,
  кривизна.
- `analysis/map.py` — карта поездок в HTML.
- `analysis/to_excel.py` — выгрузка сводки или одной записи в Excel.
