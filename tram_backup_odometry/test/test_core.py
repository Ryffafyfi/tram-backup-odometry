"""Юнит-тесты логики ноды (без ROS): python3 -m pytest test/ -q"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tram_backup_odometry.core import NS, OdometryCore  # noqa: E402

PKG = os.path.dirname(HERE)
MAPS = [os.path.join(PKG, 'maps', f) for f in
        ('shchukinskaya_tallinskaya.json', 'tallinskaya_shchukinskaya.json', 'terminal_loops.json')]
T0 = 1_788_955_803 * NS


def cfg(**kw):
    c = {'map_files': MAPS, 'gnss_init_duration_sec': 1.0, 'gnss_init_min_fixes': 2,
         'coefficients': {'velocity_scale': 1 / 3.6}}
    c.update(kw)
    return c


def feed(core, topic, t_sec, value, wall=None):
    ns = T0 + int(round(t_sec * NS))
    return core.on_input(topic, ns, value, wall if wall is not None else t_sec)


def test_publishes_velocity_with_input_stamp():
    core = OdometryCore(cfg())
    out = feed(core, 'front', 0.0, 36.0)
    assert out is not None and out.publish_velocity
    assert out.stamp_ns == T0
    out = feed(core, 'rear', 0.05, 36.0)
    assert out.stamp_ns == T0 + 50_000_000
    assert abs(out.velocity - 10.0) < 1e-6
    assert not out.publish_position  # без GNSS положение не публикуется


def test_bad_inputs_are_skipped_not_crash():
    core = OdometryCore(cfg())
    feed(core, 'front', 0.0, 10.0)
    assert feed(core, 'front', 0.1, float('nan')) is None
    assert feed(core, 'front', 0.2, float('inf')) is None
    assert feed(core, 'front', 0.3, None) is None
    assert feed(core, 'front', 0.3, 'abc') is None
    assert feed(core, 'cmd', 0.3, 99) is None
    assert core.on_input('front', 0, 10.0, 0.4) is None           # нулевой stamp
    assert core.on_input('front', -5, 10.0, 0.4) is None
    assert feed(core, 'front', 0.1, 10.0) is not None
    assert feed(core, 'front', 0.1, 10.0) is None                 # повтор
    assert feed(core, 'front', 0.05, 10.0) is None                # из прошлого
    c = core.counters
    assert c['front_bad_value'] == 4 and c['front_duplicate'] == 1 and c['front_out_of_order'] == 1


def test_cross_topic_reordering_is_accepted():
    core = OdometryCore(cfg())
    feed(core, 'cmd', 0.10, 0)
    assert feed(core, 'front', 0.05, 10.0) is not None  # другой топик чуть позже — норма
    assert feed(core, 'rear', 0.02, 10.0) is not None


def test_single_glitched_stamp_is_rejected():
    core = OdometryCore(cfg())
    t = 0.0
    for i in range(50):
        t = i * 0.1
        feed(core, 'front', t, 20.0, wall=t)
        feed(core, 'rear', t + 0.01, 20.0, wall=t + 0.01)
    # сбойный stamp на +1.2 c вперёд, затем нормальные
    assert feed(core, 'front', t + 1.3, 20.0, wall=t + 0.1) is None
    assert feed(core, 'front', t + 0.2, 20.0, wall=t + 0.2) is not None
    assert core.counters.get('front_time_glitch') == 1


def test_real_gap_in_one_topic_is_accepted():
    core = OdometryCore(cfg())
    for i in range(100):
        t = i * 0.05
        feed(core, 'cmd', t, 1, wall=t)
        if i < 10:
            feed(core, 'rear', t, 10.0, wall=t)
    # задняя тележка молчала 4.5 c, остальные шли — возобновление принимается сразу
    assert feed(core, 'rear', 5.0, 10.0, wall=5.0) is not None


def test_restart_of_playback_resets_session():
    core = OdometryCore(cfg())
    for i in range(200):
        t = i * 0.05
        feed(core, 'front', t, 10.0, wall=t)
        feed(core, 'cmd', t + 0.01, 1, wall=t + 0.01)
    core.close_gnss_window()
    assert not core.gnss_open
    w = 20.0
    r1 = feed(core, 'front', 0.0, 10.0, wall=w)
    r2 = feed(core, 'cmd', 0.01, 1, wall=w + 0.01)
    r3 = feed(core, 'front', 0.05, 10.0, wall=w + 0.05)
    assert r1 is None and r2 is None and r3 is not None
    assert core.gnss_open           # новая сессия: окно GNSS снова открыто
    assert core.sessions == 2


def test_delayed_burst_from_one_topic_does_not_reset():
    core = OdometryCore(cfg())
    for i in range(200):
        t = i * 0.05
        feed(core, 'front', t, 10.0, wall=t)
    for k in range(5):  # пачка старых сообщений одного топика
        core.on_input('rear', T0 + int((5.0 + 0.1 * k) * NS), 10.0, 10.0)
    assert core.sessions == 1


def test_gap_fill_predicts_with_estimated_stamp():
    core = OdometryCore(cfg(gap_timeout_sec=0.12, gap_publish_period_sec=0.05))
    for i in range(40):
        t = i * 0.05
        feed(core, 'front', t, 36.0, wall=t)
        feed(core, 'rear', t + 0.001, 36.0, wall=t + 0.001)
    last_wall = 39 * 0.05 + 0.001
    assert core.gap_fill(last_wall + 0.05) is None          # ещё не пропуск
    out = core.gap_fill(last_wall + 0.3)
    assert out is not None and out.source == 'predict'
    assert out.publish_velocity and abs(out.velocity - 10.0) < 0.05
    assert out.stamp_ns > T0 + int(1.95 * NS)
    assert core.gap_fill(last_wall + 100.0) is None         # слишком долго — не прогнозируем


class Exploding:
    def __init__(self, config):
        self.n = 0

    def on_front_velocity(self, t, v):
        self.n += 1
        if self.n > 3:
            raise RuntimeError('boom')

    def on_rear_velocity(self, t, v):
        pass

    def on_driver_cmd(self, t, p):
        pass

    def get_state(self, t):
        return {'velocity': float('nan'), 'velocity_valid': True}


def test_fallback_when_estimator_fails():
    core = OdometryCore(cfg(estimator_class='test_core:Exploding'))
    assert core.fb is not None
    for i in range(10):
        out = feed(core, 'front', i * 0.1, 36.0)
        assert out is not None
        assert out.publish_velocity and math.isfinite(out.velocity)
        assert out.fallback  # NaN/исключения основного -> ответ резервного оценщика
    assert core.main.errors >= 1


def test_missing_estimator_module_uses_fallback():
    core = OdometryCore(cfg(estimator_class='no_such_module:Nope'))
    out = feed(core, 'front', 0.0, 36.0)
    assert out is not None and abs(out.velocity - 10.0) < 1e-6


def test_gnss_init_places_on_map_and_window_closes():
    from tram_backup_odometry.geo import TransverseMercator
    core = OdometryCore(cfg())
    proj = TransverseMercator()
    # точка на пути «щукинская-таллинская» (антенна master), rover впереди на 12.436 м
    import json
    with open(MAPS[0]) as fh:
        pts = json.load(fh)['points']
    p0, p1 = pts[1000], pts[1012]
    inv = _inverse(proj)
    lat_m, lon_m = inv(p0['x'], p0['y'])
    lat_r, lon_r = inv(p1['x'], p1['y'])
    for i in range(30):
        t = i * 0.1
        core.on_gnss_fix('master', T0 + int(t * NS), lat_m, lon_m, p0['z'] + 3.0, 2, [0] * 9, t)
        core.on_gnss_fix('rover', T0 + int(t * NS), lat_r, lon_r, p1['z'] + 3.0, 2, [0] * 9, t)
        out = feed(core, 'front', t + 0.01, 0.0, wall=t + 0.01)
        if core.gnss_should_close():
            core.close_gnss_window()
    assert not core.gnss_open
    assert out.publish_position
    # base_link на 9.873 м впереди master вдоль пути
    exp = pts[1010]
    assert math.hypot(out.x - exp['x'], out.y - exp['y']) < 0.5
    assert abs(out.z - exp['z']) < 0.2


def _inverse(proj):
    """Численное обращение проекции (для теста)."""
    def inv(x, y):
        lat, lon = 55.8, 37.45
        for _ in range(30):
            e, n = proj.forward(lat, lon)
            ex, ny = e + proj.ox - x, n + proj.oy - y
            if abs(ex) < 1e-4 and abs(ny) < 1e-4:
                break
            lat -= ny / 111320.0
            lon -= ex / (111320.0 * math.cos(math.radians(lat)))
        return lat, lon
    return inv


def _drive(core, t0, t1, v_front, v_rear, cmd=0, dt=0.05):
    """Подать входы с шагом dt: тележки поочерёдно, контроллер каждый шаг."""
    out = None
    t = t0
    k = 0
    while t < t1 - 1e-9:
        vf = v_front(t) if callable(v_front) else v_front
        vr = v_rear(t) if callable(v_rear) else v_rear
        topic, val = ('front', vf) if k % 2 == 0 else ('rear', vr)
        if val is not None:
            out = feed(core, topic, t, val * 3.6, wall=t) or out
        out = feed(core, 'cmd', t + 0.01, cmd, wall=t + 0.01) or out
        t += dt
        k += 1
    return out


def test_stuck_rear_sensor_is_ignored():
    core = OdometryCore(cfg())
    _drive(core, 0.0, 5.0, 10.0, 10.0, cmd=1)
    out = _drive(core, 5.0, 10.0, 10.0, 0.0, cmd=1)   # задняя «залипла» на 0, тяга
    assert abs(out.velocity - 10.0) < 0.3
    assert out.state.slip  # флаг проскальзывания/отказа выставлен


def test_single_spike_is_rejected():
    core = OdometryCore(cfg())
    _drive(core, 0.0, 5.0, 8.0, 8.0)
    out = feed(core, 'front', 5.0, (8.0 * 3 + 10) * 3.6, wall=5.0)
    assert abs(out.velocity - 8.0) < 0.3
    out = _drive(core, 5.05, 6.0, 8.0, 8.0)
    assert abs(out.velocity - 8.0) < 0.1 and not out.state.slip


def test_braking_model_used_when_wheels_lost():
    core = OdometryCore(cfg())
    _drive(core, 0.0, 5.0, 10.0, 10.0, cmd=-7)
    # колёсные данные пропали, контроллер (тормоз -7) продолжает приходить
    out = _drive(core, 5.0, 7.0, None, None, cmd=-7)
    assert out is not None and out.publish_velocity
    assert 7.5 < out.velocity < 9.7   # замедляется по модели (~0.8 м/с^2), а не держит 10


# ---------------------------------------------------------------- ориентиры-остановки (оценщик)
def _lm_world(tmp_path, landmarks, length=3000.0):
    import json
    n = int(length / 10) + 1
    pts = [{'x': 1000.0 + 10.0 * i, 'y': 2000.0, 'z': 150.0} for i in range(n)]
    mp = tmp_path / 'line.json'
    mp.write_text(json.dumps({'points': pts, 'paths': [{'ext_id': 'line', 'point_indices': list(range(n))}]}))
    lp = tmp_path / 'lm.json'
    lp.write_text(json.dumps([{'track': 'line', 's': s, 'n': k, 'mad': 0.1} for s, k in landmarks]))
    from tram_backup_odometry.baseline_estimator import BaselineEstimator
    est = BaselineEstimator({'map_files': [str(mp)], 'stop_landmarks_file': str(lp),
                             'coefficients': {'velocity_scale': 1 / 3.6}})
    return est


def _lm_place(est, s, var):
    est.on_front_velocity(0.0, 0.0)
    est.on_rear_velocity(0.0, 0.0)
    est.mode, est.pid, est.dir = 'MAP', 0, 1
    est.s_ref, est.D_ref = s, est.D
    est.gnss_closed = True
    est.var_along, est.D_var = var, est.D


def _lm_drive(est, t, dist, v_true=5.0, factor=1.0, stop_sec=8.0, acc=0.8):
    """Проехать dist м (разгон/торможение acc м/с², колёса показывают factor*v) и постоять."""
    dt = 0.05
    moved, v = 0.0, 0.0
    while moved < dist - 1e-3:
        v_brake = math.sqrt(max(2.0 * acc * (dist - moved), 0.0))
        v = max(min(v_true, v + acc * dt, v_brake), 0.05)
        step = min(v * dt, dist - moved)
        moved += step
        t += dt
        est.on_front_velocity(t, v * factor * 3.6)
        est.on_rear_velocity(t + 0.01, v * factor * 3.6)
    for _ in range(int(stop_sec / dt)):
        t += dt
        est.on_front_velocity(t, 0.0)
        est.on_rear_velocity(t + 0.01, 0.0)
    return t


def _lm_s(est):
    return est.s_ref + est.dir * (est.D - est.D_ref)


def test_first_stop_prefers_primary_landmark_then_creep_to_secondary(tmp_path):
    # у платформы два места остановки: основное (100, часто) и вторичное (105.3, реже)
    est = _lm_world(tmp_path, [(100.0, 23), (105.3, 7)])
    _lm_place(est, 0.0, 3.5 ** 2)
    t = _lm_drive(est, 0.0, 102.3)          # оценка 102.3, на самом деле трамвай у 100 (ошибка 2.3 м)
    assert abs(_lm_s(est) - 100.0) < 0.3    # первая остановка — к основному месту
    t = _lm_drive(est, t, 5.3)              # подтянулся ко вторичному месту
    assert abs(_lm_s(est) - 105.3) < 0.2
    assert est.n_lm == 2


def test_random_stop_near_landmark_is_not_matched(tmp_path):
    est = _lm_world(tmp_path, [(300.0, 20)])
    _lm_place(est, 0.0, 0.3 ** 2)
    _lm_drive(est, 0.0, 304.4)              # остановка в 4.4 м от ориентира при малой неопределённости
    assert est.n_lm == 0 and abs(_lm_s(est) - 304.4) < 0.1


def test_gross_wheel_scale_error_is_detected(tmp_path):
    # колёса занижают путь на 0.6 %: остановки точно у ориентиров каждые 300 м
    marks = [(300.0 * k, 20) for k in range(1, 9)]
    est = _lm_world(tmp_path, marks)
    _lm_place(est, 0.0, 0.2 ** 2)
    t = 0.0
    for k in range(8):
        t = _lm_drive(est, t, 300.0, factor=1 / 1.006)
    assert est.n_scale_fix >= 1
    assert abs(est.scale / est.scale0 - 1.006) < 0.0015
    assert abs(_lm_s(est) - 2400.0) < 0.5
    est.reset()                          # новая сессия (запись заново) — масштаб исходный
    assert est.scale == est.scale0 and est.n_scale_fix == 0


def test_large_wheel_scale_error_found_by_stop_sequence(tmp_path):
    # колёса занижают путь на 1.5 %: ориентиры уже не попадают в строб, масштаб находится по
    # серии остановок (RANSAC), после чего положение исправляется
    marks = [(250.0 * k + (37.0 if k % 2 else 0.0), 20) for k in range(1, 16)]
    est = _lm_world(tmp_path, marks, length=4500.0)
    _lm_place(est, 0.0, 0.2 ** 2)
    est._rs_init = (0, 0.0, est.D)          # стартовая привязка (как после окна GNSS)
    t, pos = 0.0, 0.0
    for s_lm, _ in marks:
        t = _lm_drive(est, t, s_lm - pos, factor=1 / 1.015)
        pos = s_lm
    assert est.n_scale_fix >= 1
    assert abs(est.scale / est.scale0 - 1.015) < 0.002
    assert abs(_lm_s(est) - pos) < 1.0
