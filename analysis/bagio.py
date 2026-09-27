"""Чтение rosbag2 кейса без установленного ROS (через библиотеку rosbags)."""
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'dataset' / 'data'
PARQUET_DIR = ROOT / 'dataset' / 'parquet'
MSG_DIR = ROOT / 'dataset' / 'tram_vehicle_msgs' / 'msg'

TOPICS = {
    '/vehicle/front_bogie_velocity': 'front',
    '/vehicle/rear_bogie_velocity': 'rear',
    '/vehicle/driver_position_cmd': 'cmd',
    '/sensing/gnss/master/fix': 'master_fix',
    '/sensing/gnss/master/vel': 'master_vel',
    '/sensing/gnss/rover/fix': 'rover_fix',
    '/sensing/gnss/rover/vel': 'rover_vel',
}


def make_typestore():
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    for name in ('VelocitySensor', 'DriverControllerCommand'):
        text = (MSG_DIR / f'{name}.msg').read_text(encoding='utf-8')
        types.update(get_types_from_msg(text, f'tram_vehicle_msgs/msg/{name}'))
    store.register(types)
    return store


def _stamp(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def _row(short, msg):
    if short in ('front', 'rear'):
        return {'velocity': msg.velocity}
    if short == 'cmd':
        return {'position': msg.position}
    if short.endswith('_fix'):
        return {'lat': msg.latitude, 'lon': msg.longitude, 'alt': msg.altitude,
                'status': msg.status.status, 'service': msg.status.service,
                'cov_e': msg.position_covariance[0], 'cov_n': msg.position_covariance[4],
                'cov_u': msg.position_covariance[8],
                'cov_type': msg.position_covariance_type}
    if short.endswith('_vel'):
        lin, ang = msg.twist.linear, msg.twist.angular
        return {'vx': lin.x, 'vy': lin.y, 'vz': lin.z,
                'wx': ang.x, 'wy': ang.y, 'wz': ang.z}
    raise ValueError(short)


def read_bag(bag_dir, typestore=None):
    """Возвращает {короткое имя топика: DataFrame}.

    t_bag — время записи в bag (нс), t_hdr — header.stamp (нс),
    frame_id — header.frame_id; остальные столбцы — поля сообщения.
    """
    typestore = typestore or make_typestore()
    rows = {short: [] for short in TOPICS.values()}
    with AnyReader([Path(bag_dir)], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in TOPICS]
        for conn, t_bag, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            short = TOPICS[conn.topic]
            row = {'t_bag': t_bag, 't_hdr': _stamp(msg.header),
                   'frame_id': msg.header.frame_id}
            row.update(_row(short, msg))
            rows[short].append(row)
    return {short: pd.DataFrame(r) for short, r in rows.items()}


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """WGS84: широта/долгота/высота → локальные метры ENU относительно точки (lat0, lon0, alt0)."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)

    def ecef(la, lo, h):
        la, lo = np.radians(la), np.radians(lo)
        n = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
        return ((n + h) * np.cos(la) * np.cos(lo),
                (n + h) * np.cos(la) * np.sin(lo),
                (n * (1 - e2) + h) * np.sin(la))

    x, y, z = ecef(np.asarray(lat), np.asarray(lon), np.asarray(alt))
    x0, y0, z0 = ecef(lat0, lon0, alt0)
    dx, dy, dz = x - x0, y - y0, z - z0
    la0, lo0 = np.radians(lat0), np.radians(lon0)
    e = -np.sin(lo0) * dx + np.cos(lo0) * dy
    n = (-np.sin(la0) * np.cos(lo0) * dx - np.sin(la0) * np.sin(lo0) * dy
         + np.cos(la0) * dz)
    u = (np.cos(la0) * np.cos(lo0) * dx + np.cos(la0) * np.sin(lo0) * dy
         + np.sin(la0) * dz)
    return e, n, u


def bag_dirs():
    return sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
