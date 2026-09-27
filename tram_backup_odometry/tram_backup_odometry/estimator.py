"""Оценщик, который использует нода (параметр estimator_class).

Основной оценщик решения — BaselineEstimator из baseline_estimator.py: слияние двух тележек,
модель «позиция контроллера -> ускорение», движение по карте путей, ориентиры-остановки.
Модуль оставлен точкой подключения другой модели: сюда кладётся класс Estimator с одним из двух
интерфейсов, нода определяет интерфейс сама:

* Estimator(config): on_front_velocity / on_rear_velocity (сырые км/ч), on_driver_cmd, get_state —
  интерфейс ноды (estimator_api.py);
* Estimator(params, track_map): on_notch, on_bogie(t, which, v) (v в м/с), on_gnss_fix, on_gnss_vel,
  state(t) -> dict — подключается через переходник agreed_api.py.

В окружении проверки есть только numpy и стандартная библиотека. Какой оценщик работает — строка
``estimator: ...`` в логе ноды при старте.
"""

from .baseline_estimator import BaselineEstimator

Estimator = BaselineEstimator

__all__ = ['Estimator']
