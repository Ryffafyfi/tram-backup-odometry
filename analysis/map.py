"""Карта: где ездил трамвай по GNSS во всех уникальных прогонах → analysis/out/map.html.

Открывается в любом браузере двойным щелчком (подложка — Esri World Street Map, нужен интернет).
Запуск: python analysis/map.py
"""
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from bagio import PARQUET_DIR
from catalog import OUT_DIR, STOP_T, STOP_V, runs

TERMINALS = {'A': (55.8104, 37.4623), 'B': (55.7995, 37.3890)}


def track(bag):
    """Точки rover (если нет — master) раз в секунду + остановки."""
    src = 'rover' if (PARQUET_DIR / bag / 'rover_fix.parquet').exists() else 'master'
    fix = pd.read_parquet(PARQUET_DIR / bag / f'{src}_fix.parquet')
    if not len(fix):
        src = 'master'
        fix = pd.read_parquet(PARQUET_DIR / bag / 'master_fix.parquet')
    fix = fix[(fix.lat != 0) & (fix.status >= 0)].sort_values('t_hdr')
    vel = pd.read_parquet(PARQUET_DIR / bag / f'{src}_vel.parquet').sort_values('t_hdr')
    vel['speed'] = np.hypot(vel.vx, vel.vy)
    t = (vel.t_hdr.values - vel.t_hdr.values[0]) / 1e9
    stops = []
    for a, b in runs(vel.speed.values < STOP_V):
        if t[b - 1] - t[a] >= STOP_T:
            i = np.searchsorted(fix.t_hdr.values, vel.t_hdr.values[a])
            i = min(i, len(fix) - 1)
            stops.append([round(fix.lat.values[i], 6), round(fix.lon.values[i], 6),
                          round(float(t[b - 1] - t[a]))])
    pts = fix.iloc[::10]
    return fix, [[round(a, 6), round(b, 6)] for a, b in zip(pts.lat, pts.lon)], stops


def nearest_terminal(lat, lon):
    return min(TERMINALS, key=lambda k: np.hypot(lat - TERMINALS[k][0], lon - TERMINALS[k][1]))


def main():
    cat = pd.read_csv(OUT_DIR / 'catalog.csv', keep_default_na=False, na_values=[''])
    use = cat[cat.duplicate_of.isna() & ~cat.short & cat.ref_dist_m.notna()]
    items, clouds = [], []
    for r in use.itertuples():
        fix, line, stops = track(r.bag)
        start = nearest_terminal(fix.lat.iloc[0], fix.lon.iloc[0])
        end = nearest_terminal(fix.lat.iloc[-1], fix.lon.iloc[-1])
        items.append({'bag': r.bag, 'vehicle': str(r.vehicle), 'dir': f'{start}→{end}',
                      'dist': round(r.ref_dist_m), 'line': line, 'stops': stops})
        lat0 = np.radians(55.805)
        clouds.append(np.c_[fix.lon.values * 111320 * np.cos(lat0), fix.lat.values * 110540])

    # один ли маршрут: насколько далеко точки каждого прогона от точек всех остальных
    for i, it in enumerate(items):
        others = np.vstack([c[::5] for j, c in enumerate(clouds) if j != i])
        dist, _ = cKDTree(others).query(clouds[i][::5])
        it['off95'] = round(float(np.quantile(dist, 0.95)), 1)
        it['offmax'] = round(float(dist.max()), 1)
    off = pd.DataFrame(items)[['bag', 'dir', 'off95', 'offmax']]
    print('Направления:', off.dir.value_counts().to_dict())
    print('Отклонение от остальных прогонов, м (95-й процентиль по прогонам):',
          off.off95.describe().round(1).to_dict())
    print('Прогоны с точками дальше 30 м от всех остальных:\n',
          off[off.offmax > 30].to_string(index=False))

    html = TEMPLATE.replace('__DATA__', json.dumps(items, ensure_ascii=False)) \
                   .replace('__TERMINALS__', json.dumps(TERMINALS))
    path = OUT_DIR / 'map.html'
    path.write_text(html, encoding='utf-8')
    print(path)


TEMPLATE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Маршрут трамвая по GNSS</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  html, body { margin: 0; height: 100%; font: 14px Arial, sans-serif; }
  #map { height: 100%; }
  .info { background: #fff; padding: 8px 10px; border-radius: 6px; box-shadow: 0 1px 4px #0004; max-width: 300px; }
  .info b { font-size: 15px; }
</style></head>
<body><div id="map"></div>
<script>
const data = __DATA__, terminals = __TERMINALS__;
const map = L.map('map');
// тайлы OpenStreetMap и CARTO не отдаются страницам, открытым с диска (file://), поэтому подложка — Esri
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 19, referrerPolicy: 'no-referrer',
  attribution: 'Подложка &copy; Esri и поставщики данных'}).addTo(map);
const colors = {'A→B': '#1f6fd1', 'B→A': '#d1461f'};
const groups = {};
const stopsLayer = L.layerGroup();
const all = [];
for (const r of data) {
  const g = groups[r.dir] ??= L.layerGroup().addTo(map);
  const line = L.polyline(r.line, {color: colors[r.dir] || '#555', weight: 3, opacity: 0.35});
  line.bindTooltip(`${r.bag}<br>трамвай ${r.vehicle}, ${r.dir}, ${r.dist} м`, {sticky: true});
  line.on('mouseover', () => line.setStyle({opacity: 1, weight: 5}));
  line.on('mouseout', () => line.setStyle({opacity: 0.35, weight: 3}));
  line.addTo(g);
  all.push(...r.line);
  for (const [la, lo, s] of r.stops)
    L.circleMarker([la, lo], {radius: 3, color: '#333', weight: 1, fillOpacity: 0.6})
      .bindTooltip(`${r.bag}: стоял ${s} с`).addTo(stopsLayer);
}
for (const [k, [la, lo]] of Object.entries(terminals))
  L.marker([la, lo]).bindTooltip(`Конечная ${k}`, {permanent: true, direction: 'top'}).addTo(map);
const overlays = {};
for (const d of Object.keys(groups).sort())
  overlays[`<span style="color:${colors[d]}">■</span> ${d} (${data.filter(r => r.dir === d).length})`] = groups[d];
overlays['Остановки (скорость ≈ 0 дольше 3 с)'] = stopsLayer;
L.control.layers(null, overlays, {collapsed: false}).addTo(map);
map.fitBounds(all);
const info = L.control({position: 'bottomleft'});
info.onAdd = () => Object.assign(L.DomUtil.create('div', 'info'), {innerHTML:
  `<b>${data.length} уникальных прогонов с GNSS</b><br>Синие — от конечной A к B, красные — обратно.<br>` +
  `Наведи на линию, чтобы увидеть прогон. Слой «Остановки» включается справа.`});
info.addTo(map);
</script></body></html>
"""

if __name__ == '__main__':
    main()
