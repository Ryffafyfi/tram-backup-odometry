#!/usr/bin/env bash
# Проверка «сборка с нуля без интернета»: копия репозитория в пустой workspace,
# colcon build в сетевом пространстве имён без сети (unshare -n), затем юнит-тесты.
#
#   source /opt/ros/humble/setup.bash && ./scripts/offline_build_check.sh
set -eu
SRC=$(cd "$(dirname "$0")/.." && pwd)
WS=$(mktemp -d /tmp/tram_ws_offline.XXXXXX)
mkdir -p "$WS/src"
cp -r "$SRC" "$WS/src/tram_solution"
rm -rf "$WS/src/tram_solution/build" "$WS/src/tram_solution/install" "$WS/src/tram_solution/log"
cd "$WS"
BUILD='source /opt/ros/humble/setup.bash && colcon build --event-handlers console_cohesion- summary+'
if unshare -n true 2>/dev/null; then
  NET="unshare -n"
elif unshare -rn true 2>/dev/null; then
  NET="unshare -rn"
else
  NET=""
  echo "ВНИМАНИЕ: unshare недоступен — сборка идёт с сетью (проверьте вручную, отключив сеть)"
fi
echo "workspace: $WS ; сеть при сборке: ${NET:-включена}"
$NET bash -c "getent hosts pypi.org >/dev/null 2>&1 && echo 'сеть ДОСТУПНА' || echo 'сеть недоступна (как и нужно)'"
$NET bash -c "$BUILD"
set +u
source install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q src/tram_solution/tram_backup_odometry/test
ros2 pkg executables tram_backup_odometry
echo "OK: сборка без сети прошла, тесты зелёные ($WS)"
