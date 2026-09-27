"""Переходник для оценщика по договору команды + загрузка конфигурации (без ROS).

    python3 -m pytest test/ -q
"""

import math
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tram_backup_odometry.agreed_api import AgreedApiAdapter, detect_api  # noqa: E402
from tram_backup_odometry.baseline_estimator import BaselineEstimator  # noqa: E402
from tram_backup_odometry.config_util import finalize_config  # noqa: E402
from tram_backup_odometry.core import NS, OdometryCore  # noqa: E402

PKG = os.path.dirname(HERE)
MAPS = [os.path.join(PKG, 'maps', f) for f in
        ('shchukinskaya_tallinskaya.json', 'tallinskaya_shchukinskaya.json', 'terminal_loops.json')]
CSVS = [os.path.join(PKG, 'maps', f) for f in
        ('track_tallinskaya_shchukinskaya.csv', 'track_shchukinskaya_tallinskaya.csv')]
T0 = 1_788_955_803 * NS

# точка на линии Таллинская -> Щукинская (s = 1000 м) — правдоподобное положение для фиктивного оценщика
from tram_backup_odometry.track_csv import TrackLine  # noqa: E402
_P = [float(a[0]) for a in TrackLine.from_csv(CSVS[0]).at(1000.0)]
PX, PY, PZ = _P[0], _P[1], _P[2]


# ---------------------------------------------------------------- оценщики «команды» для тестов
class TeamEstimator:
    """Как в распределении задач: __init__(params, track_map), on_notch, on_bogie, state."""
    instances = []

    def __init__(self, params, track_map):
        self.params = params
        self.track_map = track_map
        self.bogie, self.notch, self.fix, self.vel = [], [], [], []
        self.v = None
        TeamEstimator.instances.append(self)

    def on_notch(self, t, position):
        self.notch.append((t, position))

    def on_bogie(self, t, which, velocity):
        self.bogie.append((t, which, velocity))
        self.v = velocity

    def on_gnss_fix(self, t, antenna, lat, lon, alt):
        self.fix.append((t, antenna, lat, lon, alt))

    def on_gnss_vel(self, t, antenna, vx, vy):
        self.vel.append((t, antenna, vx, vy))

    def state(self, t):
        if self.v is None:
            return {}
        return {'v': self.v, 'x': PX, 'y': PY, 'z': PZ, 'heading': 0.5,
                'sigma_s': 2.0, 'flags': {'slip': False}}


class TeamWithStatus(TeamEstimator):
    def on_gnss_fix(self, t, antenna, lat, lon, alt, status):
        self.fix.append((t, antenna, lat, lon, alt, status))


class TeamVelocityOnly(TeamEstimator):
    def state(self, t):
        return {'speed': 7.0}


class TeamBroken(TeamEstimator):
    def on_bogie(self, t, which, velocity):
        raise RuntimeError('boom')

    def state(self, t):
        raise RuntimeError('boom')


class UnknownApi:
    def __init__(self, *a):
        pass

    def update(self, *a):
        pass


_mod = types.ModuleType('fake_team_estimators')
for _c in (TeamEstimator, TeamWithStatus, TeamVelocityOnly, TeamBroken, UnknownApi):
    setattr(_mod, _c.__name__, _c)
sys.modules['fake_team_estimators'] = _mod


def cfg(cls_name, **kw):
    c = {'map_files': MAPS, 'track_csv_files': CSVS, 'gnss_init_duration_sec': 1.0,
         'gnss_init_min_fixes': 2, 'estimator_class': 'fake_team_estimators:%s' % cls_name,
         'coefficients': {'velocity_scale': 1 / 3.6, 'my_coef': 1.5},
         'bogie_distance': 7.55, 'antenna_master_xyz': [-9.873, 0.0, 3.0]}
    c.update(kw)
    return c


def feed(core, topic, t_sec, value, wall=None):
    ns = T0 + int(round(t_sec * NS))
    return core.on_input(topic, ns, value, wall if wall is not None else t_sec)


# ---------------------------------------------------------------- тесты
def test_detect_api():
    assert detect_api(BaselineEstimator) == 'native'
    assert detect_api(TeamEstimator) == 'agreed'
    assert detect_api(UnknownApi) is None


def test_team_estimator_gets_mps_header_time_and_notch():
    TeamEstimator.instances.clear()
    core = OdometryCore(cfg('TeamEstimator'))
    assert 'team API' in core.main_name
    team = TeamEstimator.instances[-1]
    out = feed(core, 'front', 0.0, 36.0)          # 36 км/ч
    t, which, v = team.bogie[-1]
    assert which == 'front' and abs(v - 10.0) < 1e-9
    assert abs(t - T0 / NS) < 1e-6                 # абсолютные секунды header.stamp
    assert out.stamp_ns == T0 and abs(out.velocity - 10.0) < 1e-9
    assert out.publish_position and (out.x, out.y, out.z) == (PX, PY, PZ)
    assert abs(out.var_position - 4.0) < 1e-9 and not out.fallback
    feed(core, 'rear', 0.05, 18.0)
    assert team.bogie[-1][1] == 'rear' and abs(team.bogie[-1][2] - 5.0) < 1e-9
    feed(core, 'cmd', 0.07, 5)
    tn, pos = team.notch[-1]
    assert pos == 5 and isinstance(pos, int) and abs(tn - (T0 / NS + 0.07)) < 1e-6
    # params: параметры ноды + коэффициенты матана; track_map: линии аналитика
    assert team.params['my_coef'] == 1.5 and team.params['bogie_distance'] == 7.55
    assert set(team.track_map) == {'tallinskaya_shchukinskaya', 'shchukinskaya_tallinskaya'}


def test_track_map_project_and_at_are_consistent():
    TeamEstimator.instances.clear()
    OdometryCore(cfg('TeamEstimator'))
    tm = TeamEstimator.instances[-1].track_map
    line = tm['tallinskaya_shchukinskaya']
    x, y, z, h = line.at([1000.0, 2500.5])
    s, c = line.project(x, y)
    assert abs(s[0] - 1000.0) < 0.05 and abs(s[1] - 2500.5) < 0.05
    assert max(abs(c[0]), abs(c[1])) < 0.05
    assert line.s_min < 0 < 4708 < line.s_max       # петли достроены
    assert 'grade' in line.df and len(line.df) == len(line.s)
    # направление по точкам поездки
    xs, ys, _, _ = line.at([100.0 + i for i in range(50)])
    k, _, _ = tm.detect_direction(xs, ys)
    assert k == 'tallinskaya_shchukinskaya'


def test_gnss_signature_is_respected():
    TeamEstimator.instances.clear()
    core = OdometryCore(cfg('TeamEstimator'))
    core.on_gnss_fix('master', T0, 55.8, 37.45, 170.0, 2, [0] * 9, 0.0)
    core.on_gnss_vel('master', T0 + 1000, 1.0, 2.0, 0.1, 0.0)
    team = TeamEstimator.instances[-1]
    assert team.fix[-1] == (T0 / NS, 'master', 55.8, 37.45, 170.0)
    assert team.vel[-1][1:] == ('master', 1.0, 2.0)
    TeamEstimator.instances.clear()
    core = OdometryCore(cfg('TeamWithStatus'))
    core.on_gnss_fix('rover', T0, 55.8, 37.45, 170.0, 2, [0] * 9, 0.0)
    assert TeamEstimator.instances[-1].fix[-1][-1] == 2


def test_velocity_only_team_estimator_position_from_baseline():
    core = OdometryCore(cfg('TeamVelocityOnly'))
    out = feed(core, 'front', 0.0, 36.0)
    assert abs(out.velocity - 7.0) < 1e-9          # скорость — от команды
    assert not out.publish_position                # до GNSS-старта положения нет ни у кого


def test_unknown_api_falls_back_loudly():
    logs = []
    core = OdometryCore(cfg('UnknownApi'), log=lambda lvl, msg: logs.append((lvl, msg)))
    assert 'failed' in core.main_name and core.fb is None
    assert any(lvl == 'error' and 'unknown interface' in msg for lvl, msg in logs)
    assert any('NOT RUNNING' in msg for _, msg in logs)
    out = feed(core, 'front', 0.0, 36.0)
    assert out is not None and abs(out.velocity - 10.0) < 1e-6   # работает базовый


def test_broken_team_estimator_answers_from_fallback():
    core = OdometryCore(cfg('TeamBroken'))
    out = None
    for i in range(20):
        out = feed(core, 'front', i * 0.1, 36.0, wall=i * 0.1)
    assert out.fallback and abs(out.velocity - 10.0) < 1e-6
    assert core.stats()['out_fallback'] == 20 and core.stats()['main_errors'] > 0


def test_restart_recreates_team_estimator_and_time_origin():
    TeamEstimator.instances.clear()
    core = OdometryCore(cfg('TeamEstimator'))
    for i in range(200):
        t = i * 0.05
        feed(core, 'front', t, 10.0, wall=t)
        feed(core, 'cmd', t + 0.01, 1, wall=t + 0.01)
    n_before = len(TeamEstimator.instances)
    w = 20.0
    feed(core, 'front', 0.0, 10.0, wall=w)
    feed(core, 'cmd', 0.01, 1, wall=w + 0.01)
    feed(core, 'front', 0.05, 10.0, wall=w + 0.05)
    assert core.sessions == 2
    assert len(TeamEstimator.instances) == n_before + 1     # нет reset() — создан заново
    team = TeamEstimator.instances[-1]
    assert team.bogie and abs(team.bogie[-1][0] - (T0 / NS + 0.05)) < 1e-6


def test_state_key_variants_and_missing_heading():
    TeamEstimator.instances.clear()
    ad = AgreedApiAdapter(TeamEstimator, cfg('TeamEstimator'))
    line = ad.track_map['shchukinskaya_tallinskaya']
    x0, y0, z0, _ = line.at(1500.0)
    x1, y1, _, h1 = line.at(1510.0)
    st = ad._convert({'speed': 5.0, 'position': (float(x0[0]), float(y0[0])),
                      'std_position': 3.0, 'slip_detected': True})
    assert st.velocity_valid and st.position_valid and st.slip
    assert abs(st.z - z0[0]) < 0.05                 # z из карты
    assert abs(st.var_position - 9.0) < 1e-9
    st = ad._convert({'speed': 5.0, 'x': float(x1[0]), 'y': float(y1[0]), 'z': 150.0})
    assert abs(math.remainder(st.yaw - h1[0], 2 * math.pi)) < 0.05   # курс по движению
    st = ad._convert({'x': float('nan'), 'y': 1.0, 'z': 2.0})
    assert not st.position_valid and not st.velocity_valid
    assert ad._convert({'v': 3.0, 'valid': False, 'x': 1.0, 'y': 2.0, 'z': 3.0}).position_valid is False


def test_finalize_config_loads_team_and_baseline_coefficients():
    import yaml
    with open(os.path.join(PKG, 'config', 'tram_backup_odometry.yaml')) as fh:
        c = yaml.safe_load(fh)['tram_backup_odometry']['ros__parameters']
    logs = []
    c = finalize_config(c, PKG, lambda lvl, msg: logs.append((lvl, msg)))
    assert len(c['map_files']) == 3 and all(os.path.isabs(p) for p in c['map_files'])
    assert len(c['track_csv_files']) == 2
    assert 'velocity_scale' in c['baseline_coefficients']     # страховка — из своего файла
    assert 'velocity_scale' not in c['coefficients']          # params.yaml — файл матана
    assert not [m for lvl, m in logs if lvl == 'error']


def test_baseline_ignores_team_coefficients_when_it_has_its_own():
    est = BaselineEstimator({'coefficients': {'velocity_scale': 1.0},
                             'baseline_coefficients': {'velocity_scale': 0.5}})
    assert abs(est.scale0 - 0.5) < 1e-12
    est = BaselineEstimator({'coefficients': {'velocity_scale': 0.25}})   # старый формат
    assert abs(est.scale0 - 0.25) < 1e-12


def test_team_estimator_file_runs_through_node_logic():
    """Оценщик команды (team_estimator.py, файл mathpython.py): старт по GNSS, движение, ответы."""
    import json
    import yaml
    from tram_backup_odometry.geo import TransverseMercator
    from test_core import _inverse
    with open(os.path.join(PKG, 'config', 'params.yaml'), encoding='utf-8') as fh:
        coef = yaml.safe_load(fh)
    logs = []
    core = OdometryCore(cfg('x', estimator_class='tram_backup_odometry.team_estimator:Estimator',
                            agreed_api_velocity_scale=1.0, coefficients=coef,
                            gnss_init_duration_sec=1.0, gnss_init_min_fixes=2),
                        log=lambda lvl, msg: logs.append((lvl, msg)))
    assert 'team API' in core.main_name
    with open(MAPS[0]) as fh:
        pts = json.load(fh)['points']
    inv = _inverse(TransverseMercator())
    (lat_m, lon_m), (lat_r, lon_r) = inv(pts[1000]['x'], pts[1000]['y']), inv(pts[1012]['x'], pts[1012]['y'])
    out = None
    for i in range(200):                       # 10 с: 36 км/ч, ручка +3, GNSS первые 2 с
        t = i * 0.05
        if t < 2.0:
            core.on_gnss_fix('master', T0 + int(t * NS), lat_m, lon_m, pts[1000]['z'] + 3.0, 2, [0] * 9, t)
            core.on_gnss_fix('rover', T0 + int(t * NS), lat_r, lon_r, pts[1012]['z'] + 3.0, 2, [0] * 9, t)
        feed(core, 'cmd', t, 3, wall=t)
        out = feed(core, 'front' if i % 2 else 'rear', t + 0.01, 36.0, wall=t + 0.01)
        if core.gnss_should_close():
            core.close_gnss_window()
    assert not core.gnss_open
    assert core.stats()['main_errors'] == 0
    assert out.publish_position and not out.fallback
    assert abs(out.velocity - 10.0) < 1e-6
    exp = pts[1010]                              # base_link на 9.873 м впереди master
    d = math.hypot(out.x - exp['x'], out.y - exp['y'])
    assert 60.0 < d < 110.0                      # ~8 с движения по 10 м/с после старта
    assert 140.0 < out.z < 180.0                 # высота из карты, а не 0
