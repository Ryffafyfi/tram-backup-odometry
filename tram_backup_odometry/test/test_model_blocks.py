"""Юнит-тесты модельных блоков базового оценщика (без ROS): ускорение, проскальзывание обеих
тележек по модели, адаптация модели (РМНК), защита детектора масштаба от проскальзывания."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tram_backup_odometry.baseline_estimator import BaselineEstimator  # noqa: E402


def _est(**kw):
    c = {'coefficients': {'velocity_scale': 1 / 3.6}}
    c.update(kw)
    return BaselineEstimator(c)


def _run(est, t0, t1, u, v0, wheel=None, true_accel=None, dt=0.1):
    """Трамвай разгоняется/тормозит по модели (или true_accel); тележки поочерёдно, вторая — на
    20 мс «старше» первой (как в записях: сообщения тележек приходят с перестановкой).
    wheel(t, v) -> (v_front, v_rear) — искажение показаний (проскальзывание)."""
    t, v = t0, v0
    while t < t1 - 1e-9:
        est.on_driver_cmd(t, u)
        a = true_accel(t, v) if true_accel else est.model.accel(u, v)
        v = max(v + a * dt, 0.0)
        vf, vr = wheel(t, v) if wheel else (v, v)
        est.on_front_velocity(t + dt, vf * 3.6)
        est.on_rear_velocity(t + dt - 0.02, vr * 3.6)
        t += dt
    return t, v


def test_measured_acceleration_is_unbiased_with_reordered_bogies():
    est = _est()
    _run(est, 0.0, 3.0, 0, 5.0, true_accel=lambda t, v: 0.0)
    _run(est, 3.0, 8.0, 5, 5.0, true_accel=lambda t, v: 1.0)
    # раньше половина измерений (dt = 0) не попадала в ускорение: выходило ~0.5 м/с²
    assert abs(est.a - 1.0) < 0.1


def test_both_bogie_slip_under_traction_is_caught_by_model():
    est = _est()
    t, v = _run(est, 0.0, 20.0, 6, 3.0)                  # обычный разгон, колёса честные
    assert est.n_sa_events == 0 and not est.slip
    t0 = t
    # обе тележки боксуют: +30 % за 1 с (ускорение колёс ~ +3 м/с² сверх модели)
    t, v_true = _run(est, t, t + 2.0, 6, v, wheel=lambda tt, vv: (vv * (1 + 0.3 * min(1.0, tt - t0)),) * 2)
    assert est.n_sa_events >= 1 and est.slip
    assert abs(est.v - v_true) < 0.6                      # оценка по модели, а не за колёсами
    assert v_true * 1.2 < est.front[1]                    # колёса ушли далеко вверх


def test_no_slip_events_on_normal_driving():
    est = _est()
    t, v = _run(est, 0.0, 15.0, 8, 0.0)
    t, v = _run(est, t, t + 10.0, 0, v)
    t, v = _run(est, t, t + 8.0, -6, v)
    assert est.n_sa_events == 0


def test_model_adapt_learns_heavier_tram():
    est = _est()
    est.mode = 'MAP'                                       # адаптация — после привязки к карте
    k_true = 0.8                                           # тяжелее: тяга даёт 80 % от таблицы
    t, v = _run(est, 0.0, 1.0, 0, 5.0)
    for _ in range(6):
        t, v = _run(est, t, t + 4.0, 7, v, true_accel=lambda tt, vv: k_true * est.model.accel(7, vv))
        t, v = _run(est, t, t + 4.0, -5, v, true_accel=lambda tt, vv: est.model.accel(-5, vv))
        v = max(v, 4.0)
    assert est.n_adapt > 50
    assert abs(est.k_mod[1] - k_true) < 0.1
    assert abs(est.k_mod[-1] - 1.0) < 0.1


def test_scale_anomaly_ignores_interval_with_slip():
    est = _est()
    thr = est.scale_detect_thr
    # два интервала с одинаковым уходом > порога, но на втором было проскальзывание
    est._lm_hist, est.D, est._rates = [(0.0, True, 0)], 400.0, [2 * thr]
    est._slip_sec = 3.0
    est._scale_anomaly(0, 2 * thr * 400.0)
    assert est.n_scale_fix == 0 and est._rates == []
    # без проскальзывания те же два интервала — поправка масштаба
    est._rates = [2 * thr]
    est.D = 800.0
    est._scale_anomaly(0, 2 * thr * 400.0)
    assert est.n_scale_fix == 1


def test_false_alarm_on_hard_braking_is_undone():
    # торможение сильнее модели (например, рельсовый тормоз): колёса честные, модель «не верит».
    # Детектор может сработать, но разрыв «модель − колёса» не закрывается -> ложная тревога,
    # возврат к колёсам и ретроспективная поправка пути
    est = _est()
    t, v = _run(est, 0.0, 12.0, 0, 11.0, true_accel=lambda tt, vv: 0.0)
    D0, dist = est.D, 0.0
    t_end = t + 6.0
    while t < t_end - 1e-9:
        est.on_driver_cmd(t, -3)
        v2 = max(v - 1.6 * 0.1, 0.0)
        dist += 0.5 * (v + v2) * 0.1
        v = v2
        est.on_front_velocity(t + 0.1, v * 3.6)
        est.on_rear_velocity(t + 0.08, v * 3.6)
        t += 0.1
    assert est.n_sa_events >= 1 and est.n_sa_undo >= 1
    assert abs(est.v - v) < 0.3
    assert abs((est.D - D0) - dist) < 1.0


def test_branch_missing_from_graph_gives_weighted_position(tmp_path):
    # ветка, которой нет в графе (как средний путь «Таллинской»): после стрелки положение —
    # среднее «основной путь / ветка» с весом вероятности ветки, разброс гипотез — в дисперсию
    import json
    n = 301
    pts = [{'x': 1000.0 + i, 'y': 2000.0, 'z': 150.0} for i in range(n)]
    mp = tmp_path / 'line.json'
    mp.write_text(json.dumps({'points': pts, 'paths': [{'ext_id': 'main', 'point_indices': list(range(n))}]}))
    bp = [{'x': 1100.0, 'y': 2000.0 + i, 'z': 150.0} for i in range(61)]     # от s = 100 на север, 60 м
    br = tmp_path / 'branch.json'
    br.write_text(json.dumps({'points': bp, 'paths': [{'ext_id': 'side', 'point_indices': list(range(61))}],
                              'meta': {'switch_path': 'main', 'switch_s': 100.0, 'probability': 0.2}}))
    est = BaselineEstimator({'map_files': [str(mp)], 'branch_map_files': [str(br)],
                             'coefficients': {'velocity_scale': 1 / 3.6}})
    est.on_front_velocity(0.0, 0.0)
    est.on_rear_velocity(0.0, 0.0)
    est.mode, est.pid, est.dir, est.D_ref, est.gnss_closed = 'MAP', 0, 1, est.D, True

    def pos(s):
        est.s_ref = s
        st = est.get_state(0.0)
        return st.x, st.y, st.var_position

    x, y, _ = pos(90.0)                                  # до стрелки — основной путь
    assert abs(x - 1090.0) < 1e-6 and abs(y - 2000.0) < 1e-6
    x, y, var = pos(130.0)                               # 30 м за стрелкой
    assert abs(x - (0.8 * 1130.0 + 0.2 * 1100.0)) < 1e-6 and abs(y - (0.8 * 2000.0 + 0.2 * 2030.0)) < 1e-6
    assert var >= 0.2 * 0.8 * (30.0 ** 2 + 30.0 ** 2)
    x, y, _ = pos(175.0)                                 # дальше конца ветки — только основной путь
    assert abs(x - 1175.0) < 1e-6 and abs(y - 2000.0) < 1e-6
    est.still_since = -30.0                              # стоит 30 с: вес ветки убывает (τ = 15 с)
    x, _, _ = pos(130.0)
    assert abs(x - (1130.0 - 0.2 * 2.718281828 ** -2 * 30.0)) < 1e-3
    est.branch_blend = False
    x, y, _ = pos(130.0)
    assert abs(x - 1130.0) < 1e-6 and abs(y - 2000.0) < 1e-6


def test_bogie_calibration_learns_front_rear_ratio():
    # у передней тележки колёса «больше» на 2 % (например, после обточки задней): оценщик сам
    # находит отношение и при пропаже задней тележки не завышает скорость на 1 %
    est = _est()
    t, v = 0.0, 8.0
    dt = 0.1
    for i in range(8000):                                 # 800 с равномерного хода, пары синхронные
        est.on_driver_cmd(t, 0)
        est.on_front_velocity(t + dt, v * 1.02 * 3.6)
        est.on_rear_velocity(t + dt, v * 3.6)
        t += dt
    assert abs(est.bogie_rho - 0.0198) < 0.002             # ln(1.02)
    assert abs(est.v - v * 1.01) < 0.02                    # среднее тележек — как раньше (масштаб общий)
    for i in range(30):                                   # задняя пропала — только передняя
        est.on_driver_cmd(t, 0)
        est.on_front_velocity(t + dt, v * 1.02 * 3.6)
        t += dt
    assert abs(est.v - v * 1.01) < 0.02                    # без калибровки было бы v·1.02
