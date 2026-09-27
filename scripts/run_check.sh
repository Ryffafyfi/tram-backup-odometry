#!/usr/bin/env bash
# Полный прогон одной записи: нода + (опционально) судья + замер быстродействия + ros2 bag play.
#
#   ./scripts/run_check.sh <bag_dir> [out_dir] [rate]
#
# Перед запуском: source /opt/ros/humble/setup.bash && source <ws>/install/setup.bash
# Судья (hackathon_solution_checker) запускается, если он собран в окружении.
# Результаты: <out_dir>/node.log, checker.log, perf.json, play.log, roslog/ (логи ROS и metrics_*.json)
set -u
BAG=${1:?usage: run_check.sh <bag_dir> [out_dir] [rate]}
OUT=${2:-./run_$(basename "$BAG")_$(date +%H%M%S)}
RATE=${3:-1.0}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export ROS_LOG_DIR="$OUT/roslog"

pids=()
cleanup() {
  for p in "${pids[@]}"; do kill -INT "$p" 2>/dev/null; done
  for _ in 1 2 3 4 5 6 7 8; do
    alive=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && break
    sleep 1
  done
  for p in "${pids[@]}"; do kill -KILL "$p" 2>/dev/null; done
}
trap cleanup EXIT

ros2 launch tram_backup_odometry tram_backup_odometry.launch.py > "$OUT/node.log" 2>&1 &
pids+=($!); disown
if ros2 pkg prefix hackathon_solution_checker >/dev/null 2>&1; then
  ros2 run hackathon_solution_checker metrics > "$OUT/checker.log" 2>&1 &
  pids+=($!); disown
else
  echo "hackathon_solution_checker не найден — метрики судьи не считаются" | tee "$OUT/checker.log"
fi
ros2 run tram_backup_odometry perf_probe --ros-args -p output_file:="$OUT/perf.json" > "$OUT/perf.log" 2>&1 &
pids+=($!); disown
sleep 4
echo "Воспроизведение $BAG (rate $RATE) ..."
ros2 bag play "$BAG" --rate "$RATE" > "$OUT/play.log" 2>&1
sleep 2
cleanup
trap - EXIT

echo "================ РЕЗУЛЬТАТЫ ($OUT)"
echo "--- нода (момент удаления подписок GNSS и последняя статистика):"
grep -E "GNSS SUBSCRIPTIONS REMOVED|stats:" "$OUT/node.log" | sed -e 's/\x1b\[[0-9;]*m//g' | grep -E "GNSS|stats" | sed -n '1p;$p'
echo "--- судья:"
grep -E "Velocity metrics|Position metrics" "$OUT/checker.log" | tail -2
echo "--- замер (perf_probe):"
grep -E "velocity:" "$OUT/perf.log" | tail -1
