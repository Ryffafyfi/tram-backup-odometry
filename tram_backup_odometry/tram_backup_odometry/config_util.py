"""Сборка конфигурации ноды из YAML: пути к файлам и коэффициенты (без ROS).

Одна и та же функция используется нодой (node.py) и офлайн-инструментами (tools/offline_eval.py),
поэтому нода и офлайн-прогон видят одинаковую конфигурацию.

Файлы коэффициентов:
  * ``coefficients_file`` (``config/params.yaml``) — файл физической модели: коэффициенты модели и
    размеры трамвая. Передаётся основному оценщику как ``config['coefficients']``
    (оценщику с интерфейсом on_notch/on_bogie/state — как ``params``).
  * ``baseline_coefficients_file`` (``config/baseline_params.yaml``) — коэффициенты базового
    (резервного) оценщика, ``config['baseline_coefficients']``. Отдельный файл, чтобы замена
    params.yaml на коэффициенты физической модели не меняла поведение страховки.
"""

import os


def resolve_path(p, share):
    """Путь из YAML: абсолютный — как есть, относительный — от share пакета."""
    p = os.path.expanduser(str(p))
    return p if os.path.isabs(p) else os.path.join(str(share), p)


def load_yaml_mapping(path, log=None, what='coefficients', missing_level='warn'):
    """YAML-файл -> dict. Отсутствующий или битый файл -> {} и сообщение в лог.

    Допускается формат параметров ROS: ``{node: {ros__parameters: {...}}}``.
    """
    log = log or (lambda level, msg: None)
    if not path or not os.path.isfile(path):
        log(missing_level, '%s file not found: %s (defaults used)' % (what, path))
        return {}
    try:
        import yaml
        with open(path, 'r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError('top level of %s must be a mapping' % path)
        if len(data) == 1:
            inner = next(iter(data.values()))
            if isinstance(inner, dict) and 'ros__parameters' in inner:
                data = inner['ros__parameters'] or {}
        log('info', '%s: %s (%d keys)' % (what, path, len(data)))
        return data
    except Exception as e:  # noqa: BLE001
        log('error', 'cannot read %s %s: %r (defaults used)' % (what, path, e))
        return {}


def finalize_config(cfg, share, log=None):
    """Разрешить пути (относительно share пакета) и загрузить файлы коэффициентов.

    Меняет и возвращает ``cfg``.
    """
    log = log or (lambda level, msg: None)
    maps = [resolve_path(p, share) for p in (cfg.get('map_files') or []) if str(p).strip()]
    for p in maps:
        if not os.path.isfile(p):
            log('error', 'map file not found: %s' % p)
    cfg['map_files'] = [p for p in maps if os.path.isfile(p)]

    branches = [resolve_path(p, share) for p in (cfg.get('branch_map_files') or []) if str(p).strip()]
    for p in branches:
        if not os.path.isfile(p):
            log('warn', 'branch map file not found: %s' % p)
    cfg['branch_map_files'] = [p for p in branches if os.path.isfile(p)]

    if str(cfg.get('stop_landmarks_file') or '').strip():
        cfg['stop_landmarks_file'] = resolve_path(cfg['stop_landmarks_file'], share)

    csvs = [resolve_path(p, share) for p in (cfg.get('track_csv_files') or []) if str(p).strip()]
    for p in csvs:
        if not os.path.isfile(p):
            log('warn', 'track csv file not found: %s' % p)
    cfg['track_csv_files'] = [p for p in csvs if os.path.isfile(p)]

    cfg['coefficients_file'] = resolve_path(cfg.get('coefficients_file') or 'config/params.yaml',
                                            share)
    cfg['coefficients'] = load_yaml_mapping(cfg['coefficients_file'], log, 'coefficients')

    bfile = str(cfg.get('baseline_coefficients_file') or '').strip()
    if bfile:
        cfg['baseline_coefficients_file'] = resolve_path(bfile, share)
        cfg['baseline_coefficients'] = load_yaml_mapping(
            cfg['baseline_coefficients_file'], log, 'baseline coefficients', missing_level='error')
    return cfg
