from qgis.core import QgsProject, QgsPointXY, QgsGeometry
from PyQt5.QtCore import QVariant  
import numpy as np
import math
import time
import threading

project = QgsProject.instance()

CONFIG = {
    "SMRT_POLYGON": "SRTM_Irkutsk_Poligon_Interval 5",
    "R_LAYER": "УДС_link",
    "STOPS_LAYER": "ООТ_stoppoint_stoppoint"
}


from qgis.core import (
    QgsProject,
    QgsGeometry,
    QgsPointXY,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeatureRequest
)
from qgis.PyQt.QtCore import QVariant

def point_near_roads(point_x, point_y, road_layer,
                     distance_threshold=50.0, point_crs_epsg=3857) -> bool:
    # принимает на вход координаты точки, слой проверки и дистанцию
    # на выход дается True, если указанная точка лежит рядом с слоем в пределах дистанции
    # (или если лежит прямо внутри слоя), иначе False
    if road_layer is None or not road_layer.isValid():
        raise ValueError("road_layer is not valid")

    point_crs = QgsCoordinateReferenceSystem(point_crs_epsg)
    layer_crs = road_layer.crs()

    # если нужно, то делаем преобразование из point_crs в layer_crs
    if not point_crs == layer_crs:
        xform = QgsCoordinateTransform(point_crs, layer_crs, QgsProject.instance())
        try:
            pt_transformed = xform.transform(QgsPointXY(point_x, point_y))
        except Exception as e:
            raise RuntimeError(f"CRS transform failed: {e}")
    else:
        pt_transformed = QgsPointXY(point_x, point_y)

    point_geom = QgsGeometry.fromPointXY(pt_transformed)

    # делаем слой-ограничитель, который отфильтрует
    # лишние данные за пределами distance_threshold
    buffer_geom = point_geom.buffer(distance_threshold, 8)
    bbox = buffer_geom.boundingBox()

    request = QgsFeatureRequest().setFilterRect(bbox)
    for feat in road_layer.getFeatures(request):
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue
        # пробуем посчитать дистанцию для слоя crs
        try:
            d = geom.distance(point_geom)
        except Exception:
            # как fallback пробуем простую дистанцию между вершинами bbox
            d = geom.boundingBox().distance(point_geom.boundingBox())
        if d <= distance_threshold:
            return True
    return False

layers = QgsProject.instance().mapLayersByName(CONFIG["STOPS_LAYER"])
if not layers:
    raise ValueError(f'Layer {CONFIG["STOPS_LAYER"]} not found')
layer = layers[0]

# crs в EPSG 3857
dst_crs = QgsCoordinateReferenceSystem('EPSG:3857')
layer_crs = layer.crs() if layer.crs() is not None else QgsCoordinateReferenceSystem('EPSG:3857')
if layer_crs != dst_crs:
    xform = QgsCoordinateTransform(layer_crs, dst_crs, QgsProject.instance())
else:
    xform = None

points = []
for feat in layer.getFeatures():
    geom = feat.geometry()
    if geom is None:
        continue
    if geom.isMultipart():
        pts = geom.asMultiPoint()
    else:
        pts = [geom.asPoint()]
    for p in pts:
        if xform:
            p = xform.transform(p)
        x = float(p.x())
        y = float(p.y())
        points.append([x, y])


# инициализируем выходной слой
layer = QgsVectorLayer("Polygon?crs=EPSG:3857", "custom_poly", "memory")
prov = layer.dataProvider()

def main_worker(point_x, point_y):
    global layer, prov

    center_point = QgsPointXY(point_x, point_y)

    R = 500 # максимальный радиус

    # слой высотных полигонов
    pol_layer = QgsProject.instance().mapLayersByName(CONFIG["SMRT_POLYGON"])[0]

    def get_point_height(x, y) -> float:
        pt = QgsGeometry.fromPointXY(QgsPointXY(x, y))
        max_h = None
        for feat in pol_layer.getFeatures():
            geom = feat.geometry()
            if geom.contains(pt):
                max_h = feat["MIN"]
                break
        return max_h

    h_0 = get_point_height(point_x, point_y)

    polygon_points = []

    for theta in range(-180, 180, 5):
        rad = np.deg2rad(theta)
        
        m_cos = math.cos(rad)
        m_sin = math.sin(rad)

        roads = QgsProject.instance().mapLayersByName(CONFIG["R_LAYER"])[0]
        
        result_R = 50 # минимальный радиус где можно пройти даже без дорог

        # увеличение радиуса там, есть дороги
        # идем от максимальной дальности к центру
        for ray_length in range(5, 1, -1):
            temp_x = float(round(point_x + (R * m_cos / 5 * ray_length), 2))
            temp_y = float(round(point_y + (R * m_sin / 5 * ray_length), 2))
            near_road_flag = point_near_roads(temp_x, temp_y, roads, distance_threshold=30.0)
            if near_road_flag:
                # если нашли, проверим, что дорога действильно позволяет пройти к нашей точке
                temp_x = float(round(point_x + (R * m_cos / 5 * (ray_length / 2)), 2))
                temp_y = float(round(point_y + (R * m_sin / 5 * (ray_length / 2)), 2))
                near_road_flag_2 = point_near_roads(temp_x, temp_y, roads, distance_threshold=50.0)
                if near_road_flag_2:
                    result_R = R / 5 * ray_length
                    break

        if result_R < 50:
            result_R = 50

        test_x = float(round(point_x + (result_R * m_cos), 2))
        test_y = float(round(point_y + (result_R * m_sin), 2))

        max_h = get_point_height(test_x, test_y)
        try:
            dh = max_h - h_0
        except:
            dh = 0

        # корректировка радиуса в заивисимости от высоты
        if dh > 0:
            r_xy_sq = result_R*result_R - dh*dh*1000 # тяжесть подъема
        else:
            r_xy_sq = result_R*result_R + dh*dh*1000 # легкость спуска
        if r_xy_sq < 0:
            r_xy_sq = 0

        r_xy = np.sqrt(r_xy_sq)

        # итоговые координаты
        x_p = float(point_x + r_xy * math.cos(rad))
        y_p = float(point_y + r_xy * math.sin(rad))
        


        
        polygon_points.append(QgsPointXY(x_p, y_p))



    # замыкаем
    polygon_points.append(polygon_points[0])
    
    # фильтруем данные на аномальные пики (точёные)
    def filter_spikes(points, max_dist, check_next=False):
        if not points:
            return []
        kept = [points[0]]
        for i in range(1, len(points)):
            p = points[i]
            prev = kept[-1]
            dx = p.x() - prev.x()
            dy = p.y() - prev.y()
            dist = (dx*dx + dy*dy) ** 0.5
            if dist <= max_dist:
                kept.append(p)
            else:
                if check_next and i + 1 < len(points):
                    nxt = points[i+1]
                    dx2 = p.x() - nxt.x()
                    dy2 = p.y() - nxt.y()
                    dist2 = (dx2*dx2 + dy2*dy2) ** 0.5
                    if dist2 <= max_dist:
                        kept.append(p)
        return kept

    max_dist = 50.0
    filtered = filter_spikes(polygon_points, max_dist, check_next=True)
    if filtered and (filtered[0].x() != filtered[-1].x() or filtered[0].y() != filtered[-1].y()):
        filtered.append(filtered[0])

    # добавляем окружность на слой
    polygon_geom = QgsGeometry.fromPolygonXY([filtered])

    feat = QgsFeature()
    feat.setGeometry(polygon_geom)
    prov.addFeature(feat)
    layer.updateExtents()

for point_x, point_y in points[:100]:
    threading.Thread(target=main_worker, args=(point_x, point_y)).start()
    time.sleep(0.1)

while threading.active_count() > 1:
    time.sleep(1)

QgsProject.instance().addMapLayer(layer)

symbol = QgsFillSymbol.createSimple({'color': '255,0,0,0'}) # RGBA цвет тут игнорируется 
symbol.setColor(QColor(0, 255, 0)) # fill color
symbol.setOpacity(0.6) # opacity

# apply to layer
renderer = QgsSingleSymbolRenderer(symbol)
layer.setRenderer(renderer)
layer.triggerRepaint()


print("Готово!")
