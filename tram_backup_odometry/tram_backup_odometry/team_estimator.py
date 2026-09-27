"""Альтернативный оценщик с интерфейсом on_notch / on_bogie / state (исходный файл mathpython.py).

Подключается файлом config/tram_backup_odometry_team.yaml через переходник agreed_api.py:
  * params    — параметры ноды + config/params.yaml (коэффициенты физической модели);
  * track_map — линии пути maps/track_*.csv с методами select_direction / project_to_path /
                get_pose_at / get_special_points;
  * gnss_transformer.to_utm(lat, lon, alt) -> x, y, z сразу в координатах карты;
  * скорость тележек приходит сырой (км/ч): оценщик сам делит на 3.6, поэтому для него
    agreed_api_velocity_scale = 1.0.

Исправления при подключении (помечены # FIX):
  1. on_notch сдвигал self.state.t — время, от которого интегрируется путь, и сообщения тележек
     (их stamp на ~50 мс позже ручки) отбрасывались как «из прошлого». Теперь у ручки своё время.
  2. z всегда был 0.0 -> высота берётся из карты.
  3. атрибут self.state закрывал метод state(t) — атрибут переименован в self.st.
Модель по ручке (_get_accel) не реализована и возвращает 0.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Protocol, Dict, Tuple


class TrackMap(Protocol):
    def select_direction(self, x: float, y: float, heading: float) -> str: ...
    def project_to_path(self, x: float, y: float) -> float: ...
    def get_pose_at(self, s: float) -> Tuple[float, float, float, float]: ...
    def get_special_points(self) -> list: ...


class GnssTransformer(Protocol):
    def to_utm(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]: ...


@dataclass
class VehicleState:
    t: float = 0.0
    v: float = 0.0
    s: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    heading: float = 0.0
    cov_v: float = 1.0
    cov_s: float = 1.0
    front_ok: bool = True
    rear_ok: bool = True
    at_stop: bool = False
    slip_active: bool = False
    direction: Optional[str] = None


@dataclass
class BogieData:
    last_t: float = -1.0
    last_v: float = 0.0


class Estimator:
    def __init__(self, params: dict, track_map: TrackMap, gnss_transformer: GnssTransformer):
        self.p = params
        self.map = track_map
        self.gnss = gnss_transformer

        self.rover_x = params.get("rover_x", 2.563)
        self.master_x = params.get("master_x", -9.873)
        self.bogie_span = params.get("bogie_span", 7.55)

        self.st = VehicleState()  # FIX: было self.state — закрывало метод state(t)
        self.front = BogieData()
        self.rear = BogieData()
        
        self.init_done = False
        self.gnss_buf: Dict[str, Dict[float, Tuple[float, float, float]]] = {"master": {}, "rover": {}}
        self.stop_timer = 0.0
        
        self.v_stop_thr = params.get("stop_v_thr", 0.05)
        self.stop_dur = params.get("stop_agree_s", 2.0)
        self.pred_v = 0.0
        self._t_notch = None  # FIX: время ручки отдельно от времени интегрирования пути

    def on_notch(self, t: float, position):
        if not self.init_done:
            self._bootstrap(t)
            return
        
        dt = t - self._t_notch if self._t_notch is not None else 0.0
        self._t_notch = t
        if dt <= 0:
            return

        a = self._get_accel(position)
        self.pred_v = max(0.0, self.st.v + a * dt)

    def on_bogie(self, t: float, which: str, velocity_kmh: float):
        if not self.init_done:
            return
            
        v_ms = velocity_kmh / 3.6
        if which == "front":
            self.front.last_t, self.front.last_v = t, v_ms
        else:
            self.rear.last_t, self.rear.last_v = t, v_ms

        v_meas = 0.5 * (self.front.last_v + self.rear.last_v)
        
        a_model = self.pred_v - self.st.v
        if a_model > 0:
            v_meas = min(v_meas, min(self.front.last_v, self.rear.last_v))
        elif a_model < 0:
            v_meas = max(v_meas, max(self.front.last_v, self.rear.last_v))

        dt = t - self.st.t
        if abs(self.front.last_v) < self.v_stop_thr and abs(self.rear.last_v) < self.v_stop_thr:
            self.stop_timer += dt
            if self.stop_timer >= self.stop_dur:
                v_meas = 0.0
                self.st.at_stop = True
        else:
            self.stop_timer = 0.0
            self.st.at_stop = False

        self._update(t, v_meas)

    def on_gnss_fix(self, t: float, antenna: str, lat: float, lon: float, alt: float):
        if self.init_done:
            return
        self.gnss_buf[antenna][t] = (lat, lon, alt)

    def on_gnss_vel(self, t: float, antenna: str, vx: float, vy: float):
        pass

    def state(self, t: float) -> dict:
        return {
            "t": t,
            "velocity": self.st.v,
            "s": self.st.s,
            "x": self.st.x,
            "y": self.st.y,
            "z": self.st.z,
            "heading": self.st.heading,
            "cov_v": self.st.cov_v,
            "cov_s": self.st.cov_s,
            "front_ok": self.st.front_ok,
            "rear_ok": self.st.rear_ok,
            "at_stop": self.st.at_stop,
            "slip_active": self.st.slip_active,
            "direction": self.st.direction,
        }

    def _bootstrap(self, t: float) -> bool:
        mf = self.gnss_buf["master"]
        rf = self.gnss_buf["rover"]
        if not (mf and rf):
            return False
            
        t0 = min(max(mf), max(rf))
        mlat, mlon, malt = mf[min(mf, key=lambda k: abs(k - t0))]
        rlat, rlon, ralt = rf[min(rf, key=lambda k: abs(k - t0))]
        
        mx, my, mz = self.gnss.to_utm(mlat, mlon, malt)
        rx, ry, rz = self.gnss.to_utm(rlat, rlon, ralt)
        
        heading = math.atan2(ry - my, rx - mx)
        
        bl_x = mx + abs(self.master_x) * math.cos(heading)
        bl_y = my + abs(self.master_x) * math.sin(heading)
        
        self.st.direction = self.map.select_direction(bl_x, bl_y, heading)
        self.st.s = self.map.project_to_path(bl_x, bl_y)
        self.st.x, self.st.y, self.st.z, _ = self.map.get_pose_at(self.st.s)  # FIX: точка на пути, z из карты
        self.st.heading = heading
        self.st.t = t
        self.init_done = True
        return True

    def _get_accel(self, position) -> float:
        return 0.0

    def _update(self, t: float, v_ms: float):
        dt = t - self.st.t
        if dt <= 0:
            return
            
        self.st.v = v_ms
        self.st.s += v_ms * dt
        
        x, y, z, _ = self.map.get_pose_at(self.st.s)
        self.st.x, self.st.y, self.st.z = x, y, z  # FIX: высота из карты, а не 0
        
        rear_s = self.st.s - self.bogie_span
        rx, ry, _, _ = self.map.get_pose_at(rear_s)
        self.st.heading = math.atan2(y - ry, x - rx)
        
        self.st.t = t