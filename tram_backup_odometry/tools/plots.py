#!/usr/bin/env python3
"""Графики для документации: траектория, ошибка положения и скорости во времени, задержки.

    python3 tools/plots.py <bag_с_эталоном> <out_dir> [perf.json.series.csv]
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from offline_eval import ATS, default_cfg, run  # noqa: E402
from tram_backup_odometry.track_map import TrackMap  # noqa: E402


def main():
    bag, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cfg = default_cfg()
    core, outs, refs, kind, logs, gclose, msgs = run(bag, cfg)
    ev = [(a, 0, s, r) for a, s, *r in refs] + [(a, 1, s, o) for a, s, o in outs]
    ev.sort(key=lambda e: (e[0], e[1]))
    vel, pos = ATS(), ATS()
    for a, idx, s, p in ev:
        if idx == 0:
            vel.add(0, s, (s, p))
            pos.add(0, s, (s, p))
        else:
            if p.publish_velocity:
                vel.add(1, s, (s, p))
            if p.publish_position:
                pos.add(1, s, (s, p))
    t0 = refs[0][1]
    # --- скорость
    tv = [(r[0] - t0) / 1e9 for (r, o) in vel.pairs]
    vr = [r[1][3] for (r, o) in vel.pairs]
    ve = [o[1].velocity for (r, o) in vel.pairs]
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(tv, vr, lw=0.8, label='эталон (kinematic_state)')
    ax[0].plot(tv, ve, lw=0.6, label='/result/velocity')
    ax[0].set_ylabel('м/с'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title('Скорость: RMSE %.3f м/с' % math.sqrt(sum((a - b) ** 2 for a, b in zip(ve, vr)) / len(vr)))
    ax[1].plot(tv, [a - b for a, b in zip(ve, vr)], lw=0.5, color='C3')
    ax[1].set_ylabel('ошибка, м/с'); ax[1].set_xlabel('время записи, с'); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / 'velocity.png', dpi=90); plt.close(fig)
    # --- положение: ошибка во времени
    tp = [(r[0] - t0) / 1e9 for (r, o) in pos.pairs]
    err = [math.sqrt((o[1].x - r[1][0]) ** 2 + (o[1].y - r[1][1]) ** 2 + (o[1].z - r[1][2]) ** 2)
           for (r, o) in pos.pairs]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tp, err, lw=0.7)
    ax.set_yscale('log'); ax.set_ylim(0.01, 100)
    ax.set_xlabel('время записи, с'); ax.set_ylabel('ошибка 3D, м (лог.)'); ax.grid(alpha=0.3, which='both')
    ax.set_title('Ошибка положения base_link: RMSE %.2f м (до конечной «Таллинская» %.2f м)' % (
        math.sqrt(sum(e * e for e in err) / len(err)),
        math.sqrt(sum(e * e for t, e in zip(tp, err) if t < 1230) / max(1, sum(1 for t in tp if t < 1230)))))
    fig.tight_layout(); fig.savefig(out / 'position_error.png', dpi=90); plt.close(fig)
    # --- траектория на карте
    tm = TrackMap(cfg['map_files'])
    fig, ax = plt.subplots(figsize=(12, 8))
    for p in tm.paths:
        ax.plot(p.xs, p.ys, color='0.8', lw=3, zorder=1)
    ax.plot([r[1][0] for r, o in pos.pairs], [r[1][1] for r, o in pos.pairs], lw=1.2, label='эталон', zorder=2)
    ax.plot([o[1].x for r, o in pos.pairs], [o[1].y for r, o in pos.pairs], lw=0.8, ls='--',
            label='/result/position', zorder=3)
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()
    ax.set_title('Траектория (серым — карта: 2 пути pathgraph + восстановленные петли)')
    fig.tight_layout(); fig.savefig(out / 'trajectory.png', dpi=80); plt.close(fig)
    # --- задержки
    if len(sys.argv) > 3:
        rows = list(csv.reader(open(sys.argv[3])))[1:]
        lat = [float(r[2]) for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(lat, bins=[i * 0.25 for i in range(0, 81)], log=True)
        ax.set_xlabel('задержка вход -> ответ (perf_probe), мс'); ax.set_ylabel('число ответов')
        ax.axvline(100, color='C3'); ax.grid(alpha=0.3)
        s = sorted(lat)
        ax.set_title('p50 %.2f, p99 %.2f, max %.1f мс (порог 100 мс вне шкалы)' % (
            s[len(s) // 2], s[int(0.99 * len(s))], s[-1]))
        fig.tight_layout(); fig.savefig(out / 'latency.png', dpi=90); plt.close(fig)
    print('ok')


if __name__ == '__main__':
    main()
