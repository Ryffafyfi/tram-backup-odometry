# Резервная одометрия трамвая (ROS 2 Humble)

Пакет `tram_backup_odometry` в реальном времени оценивает продольную скорость и положение
трамвая по скоростям передней и задней тележек и позиции контроллера водителя. GNSS нужен
только на старте: первые 5 секунд записи идут на начальную выставку, после этого подписки на
GNSS удаляются, и дальше решение работает без спутников и без IMU.

Сделано на хакатоне Московского транспорта, трек «Резервная одометрия по модели».

Документация:

- [docs/model.md](docs/model.md) — математическая модель: уравнения, входы и выходы, оценка ускорения, модель тяги;
- [docs/parameters.md](docs/parameters.md) — допущения и настраиваемые параметры;
- [docs/evaluation.md](docs/evaluation.md) — методика проверки, точность, быстродействие, графики;
- [docs/roadmap.md](docs/roadmap.md) — ограничения и план развития.

## Входы и выходы

Нода подписывается на:

- `/vehicle/front_bogie_velocity`, `/vehicle/rear_bogie_velocity` (`tram_vehicle_msgs/msg/VelocitySensor`);
- `/vehicle/driver_position_cmd` (`tram_vehicle_msgs/msg/DriverControllerCommand`);
- `/sensing/gnss/{master,rover}/fix` и `/sensing/gnss/{master,rover}/vel` — только на старте.

И публикует:

- `/result/velocity` (`tram_vehicle_msgs/msg/VelocitySensor`) — скорость, м/с;
- `/result/position` (`nav_msgs/msg/Odometry`) — положение `base_link` в системе карты
  (`frame_id: map`, `child_frame_id: base_link`), курс и тангаж в `orientation`, скорость в
  `twist.twist.linear.x`, ковариации положения и скорости;
- `/diagnostics` — флаг проскальзывания, оценка сцепления, режим оценщика, задержки, CPU и
  память; флаг и сцепление дублируются в `/tram_backup_odometry/slip_detected` (Bool) и
  `/tram_backup_odometry/adhesion_estimate` (Float64).

На каждое входное сообщение публикуется ответ со `header.stamp` этого сообщения (время из
записи, а не системные часы). Если входы пропадают, нода продолжает публиковать прогноз по
модели с частотой 20 Гц.

## Сборка

Нужны Ubuntu 22.04 и ROS 2 Humble (`ros-humble-ros-base`). Других зависимостей нет, сборка
работает без интернета.

```bash
mkdir -p ~/tram_ws/src && cd ~/tram_ws/src
git clone https://github.com/Ryffafyfi/tram-backup-odometry.git
cd ~/tram_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

В репозитории есть свой пакет `tram_vehicle_msgs` (копия из датасета с исправленным
`package.xml`, без него colcon пакет не собирает). Если в workspace уже лежит
`tram_vehicle_msgs` из check-code, оставьте один из двух, иначе colcon остановится на
дублирующемся имени пакета. Нода работает и с урезанным пакетом из check-code, где есть
только `VelocitySensor`: тогда позиция контроллера не используется, о чём пишется
предупреждение в лог.

Можно собрать в контейнере: `docker build -t tram-backup-odometry .` (образ
`ros:humble-ros-base`, те же шаги), затем
`docker run --rm -it -v <папка_с_записями>:/bags tram-backup-odometry`.

## Запуск на записи

Сначала нода, потом запись:

```bash
# терминал 1
source ~/tram_ws/install/setup.bash
ros2 launch tram_backup_odometry tram_backup_odometry.launch.py

# терминал 2
source ~/tram_ws/install/setup.bash
ros2 bag play <папка_записи>

# терминал 3, по желанию: судья организаторов и внешний замер задержки
ros2 run hackathon_solution_checker metrics
ros2 run tram_backup_odometry perf_probe --ros-args -p output_file:=/tmp/perf.json
```

То же одной командой (нода, судья, если он собран, замер задержки и воспроизведение; в конце
печатается сводка):

```bash
bash ~/tram_ws/src/tram-backup-odometry/scripts/run_check.sh <папка_записи> ./run_out
```

Если запись запустить повторно без перезапуска ноды, нода увидит скачок времени назад и начнёт
новую сессию: снова откроет окно GNSS для выставки.

## Что проверить

- `ros2 topic hz /result/velocity` — около 39 Гц: ответ на каждый вход (две тележки примерно по
  10 Гц и контроллер 20 Гц).
- `ros2 topic echo /result/position --field header` — stamp совпадает со stamp входов.
- В логе ноды при старте видно, какой оценщик и какая карта загружены. Строка
  `GNSS SUBSCRIPTIONS REMOVED at bag time ... (5.04 s after start)` отмечает момент удаления
  подписок GNSS. Раз в 10 секунд печатается строка `stats:` — число ответов, частота,
  максимальный разрыв между ответами, задержка обработки p50/p99/max, CPU, память, число
  отброшенных входов.
- Те же замеры пишутся в `~/.ros/log/tram_backup_odometry/metrics_<время>.json` (обновляется
  каждые 10 секунд и при завершении).
- `perf_probe` меряет задержку снаружи ноды, как её видит судья: от приёма входа до приёма
  ответа с тем же stamp. Сводка печатается каждые 5 секунд и сохраняется в JSON.
- `ros2 topic echo /diagnostics` — проскальзывание, сцепление, режим оценщика, поправки у мест
  остановок, производительность.

Проверка устойчивости: `ros2 run tram_backup_odometry fault_player <папка_записи> --profile hard`
воспроизводит запись с испорченными сообщениями (NaN, повторы, перестановки, пропуски, выбросы,
залипание датчика) вместо `ros2 bag play`.

Без ROS ту же логику можно прогнать офлайн и получить метрики, совпадающие с судьёй до третьего
знака:

```bash
pip install rosbags numpy pyyaml
python3 tram_backup_odometry/tools/offline_eval.py <папка_записи>
python3 tram_backup_odometry/tools/offline_eval.py <папка_записи> --slip 1      # с проскальзыванием
python3 tram_backup_odometry/tools/offline_eval.py <папка_записи> --gaps 60,5   # с пропусками колёс
```

## Результаты

Проверочная запись `30618_88aea4d9` (21,8 минуты, около 5 км), эталон организаторов:

| | Скорость, RMSE | Положение, RMSE 3D |
|---|---|---|
| Прогон в ROS с судьёй (27.09, сборка до вечерних изменений) | 0,027 м/с | 5,86 м |
| Текущая версия, офлайн той же логикой | 0,025 м/с | 5,08 м |
| Текущая версия, до конечной «Таллинская» (первые 1270 с) | | 0,77 м |

Почти вся ошибка положения набирается в последние 35 секунд. На конечной трамвай свернул на
путь, который по одним колёсам нельзя отличить от основного (подробно в
[docs/model.md](docs/model.md) и [docs/roadmap.md](docs/roadmap.md)).

На обучающих записях, где есть RTK-эталон (59 записей), медиана ошибки положения 1,31 м,
среднее 2,44 м.
Задержка от входа до ответа 2–4 мс (p99) при лимите 100 мс, частота около 39 Гц, нагрузка около
5 % одного ядра, память около 60 МБ. Подробности и методика — в
[docs/evaluation.md](docs/evaluation.md).

## Структура репозитория

```
tram_backup_odometry/         пакет ROS 2 (ament_python)
  tram_backup_odometry/
    node.py                   нода: подписки, публикация, окно GNSS, диагностика, замеры
    core.py                   логика без ROS: проверка входов, время, прогноз при пропусках, резерв
    baseline_estimator.py     оценщик: тележки + модель по позиции контроллера + карта
    estimator.py              точка подключения оценщика (по умолчанию baseline_estimator)
    track_map.py, geo.py      граф путей и перевод GNSS в систему карты
    team_estimator.py         альтернативный оценщик с интерфейсом on_notch / on_bogie / state
    agreed_api.py, track_csv.py   переходник и линии пути для него
    perf_probe.py             внешний замер задержки, частоты и ресурсов
    fault_player.py           воспроизведение записи со сбоями
  config/                     параметры ноды и коэффициенты модели
  maps/                       карта путей, кольца у конечных, места остановок
  launch/, test/              launch-файл и юнит-тесты (pytest, без ROS)
  tools/                      офлайн-проверка, калибровки, построение карты
tram_vehicle_msgs/            сообщения датасета
tools/, analysis/             выгрузка записей в CSV, эталон по GNSS, подсчёт ошибок, обзор данных
scripts/                      прогон записи с судьёй, проверка сборки без сети
docs/                         документация
```

Юнит-тесты: `python3 -m pytest tram_backup_odometry/test -q` (38 тестов).

Альтернативный оценщик (`team_estimator.py` с коэффициентами `config/params.yaml`) подключается
файлом параметров:

```bash
ros2 launch tram_backup_odometry tram_backup_odometry.launch.py \
  params_file:=$(ros2 pkg prefix tram_backup_odometry)/share/tram_backup_odometry/config/tram_backup_odometry_team.yaml
```

Лицензия — Apache 2.0.
