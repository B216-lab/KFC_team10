from qgis.core import QgsProject, QgsPointXY, QgsGeometry
from PyQt5.QtCore import QVariant
from qgis.PyQt import QtWidgets
from qgis.utils import iface
from qgis.core import QgsProject
from qgis.gui import QgsMapToolEmitPoint
from qgis.core import (
    QgsCoordinateTransform,
    QgsProject,
    QgsSpatialIndex,
    QgsFeatureRequest,
    QgsDistanceArea,
)
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

class RadiusSelectionTool(QgsMapToolEmitPoint):
    def __init__(self, canvas, layer, radius, limit, callback):
        self.canvas = canvas
        super().__init__(self.canvas)
        self.layer = layer
        self.radius = radius
        self.limit = limit
        self.callback = callback
        self.index = QgsSpatialIndex(self.layer.getFeatures())
        self.d = QgsDistanceArea()
        self.d.setSourceCrs(self.layer.crs(), QgsProject.instance().transformContext())

    def canvasReleaseEvent(self, event):
        point_canvas = self.toMapCoordinates(event.pos())
        transform = QgsCoordinateTransform(
            QgsProject.instance().crs(),
            self.layer.crs(),
            QgsProject.instance().transformContext(),
        )
        point_layer = transform.transform(point_canvas)

        candidate_ids = self.index.nearestNeighbor(point_layer, self.limit * 5)
        features_with_dist = []
        request = QgsFeatureRequest().setFilterFids(candidate_ids)
        for feat in self.layer.getFeatures(request):
            geom = feat.geometry()
            dist = self.d.measureLine(point_layer, geom.asPoint())

            if dist <= self.radius:
                features_with_dist.append((dist, feat))

        features_with_dist.sort(key=lambda x: x[0])
        result_features = [f[1] for f in features_with_dist[: self.limit]]

        if self.callback:
            self.callback(result_features)

        self.canvas.unsetMapTool(self)
        self.deactivate()


def get_features_in_radius(layer, radius, limit=10, callback=None):
    canvas = iface.mapCanvas()
    if not layer or not layer.isValid():
        iface.messageBar().pushMessage(
            "Ошибка", "Слой не найден или невалиден", level=3
        )
        return None

    def default_callback(features):
        if not features:
            iface.messageBar().pushMessage("Поиск", "Ничего не найдено", level=1)
            return
        ids = [f.id() for f in features]
        layer.selectByIds(ids)
        iface.messageBar().pushMessage("Успех", f"Найдено: {len(features)}", level=1)

    final_callback = callback if callback else default_callback

    tool = RadiusSelectionTool(canvas, layer, radius, limit, final_callback)
    canvas.setMapTool(tool)
    iface.messageBar().pushMessage(
        "Инфо", f"Кликните на карте (Радиус: {radius}м, Лимит: {limit})", level=0
    )

    return tool


TARGET_LAYER_NAME = "ООТ_stoppoint_stoppoint"

try:
    layers = QgsProject.instance().mapLayersByName(TARGET_LAYER_NAME)
    if not layers:
        raise IndexError
    node_layer = layers[0]
    print(f"Слой '{TARGET_LAYER_NAME}' успешно найден.")
except IndexError:
    print(f"ОШИБКА: Слой '{TARGET_LAYER_NAME}' не найден!")
    print("\nДоступные слои в проекте (скопируйте нужное имя):")
    for layer_id, layer in QgsProject.instance().mapLayers().items():
        print(f"- {layer.name()}")
    node_layer = None

ATTR_NAME = "NO"
STOP_IDS_RESULT = []

tr = QgsCoordinateTransform(node_layer.crs(), QgsCoordinateReferenceSystem("EPSG:3857"), QgsProject.instance())
def process_stops(features_list):
    global STOP_IDS_RESULT

    if not features_list:
        print("В радиусе остановок не найдено.")
        STOP_IDS_RESULT = []
        return

    print(f"\nРезультаты поиска (Всего: {len(features_list)})")
    found_coords = []

    for feat in features_list:
        geom = feat.geometry()
        pt = geom.asPoint()
        pt_trans=tr.transform(pt)
        coords = (pt_trans.x(), pt_trans.y())
        found_coords.append(coords)
        try:
            val = feat[ATTR_NAME]
            print(
                f"Остановка (ID: {feat.id()}): {ATTR_NAME}={val} -> Координаты: {coords}"
            )
        except KeyError:
            print(f"Остановка (ID: {feat.id()}): -> Координаты: {coords}")

    STOP_IDS_RESULT = found_coords
    print("-" * 30)
    print("Итоговый список координат (X, Y):", STOP_IDS_RESULT)
    if node_layer:
        node_layer.selectByIds([f.id() for f in features_list])

print('Выберите точку на карте')
if node_layer:
    t = get_features_in_radius(node_layer, radius=50000, limit=30, callback=process_stops)
    _active_tool = t


from qgis.PyQt import QtWidgets, QtCore

class ControllerWidget(QtWidgets.QWidget):
    def __init__(self, canvas, tool, layer=None, parent=None):
        super().__init__(parent, QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint)
        self.canvas = canvas
        self.tool = tool
        self.layer = layer
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6,6,6,6)
        layout.addWidget(QtWidgets.QLabel(f"Радиус: {tool.radius} м — Лимит: {tool.limit}"))
        self.finish_btn = QtWidgets.QPushButton("Finish")
        self.keep_btn = QtWidgets.QPushButton("Keep")
        self.clear_btn = QtWidgets.QPushButton("Clear sel")
        layout.addWidget(self.clear_btn)
        layout.addWidget(self.keep_btn)
        layout.addWidget(self.finish_btn)
        self.finish_btn.clicked.connect(self.on_finish)
        self.keep_btn.clicked.connect(self.on_keep)
        self.clear_btn.clicked.connect(self.on_clear)
        self.loop = QtCore.QEventLoop()

        # position top-left of map canvas
        canvas_pos = canvas.mapToGlobal(canvas.rect().topLeft())
        self.move(canvas_pos + QtCore.QPoint(8, 8))
        self.show()

    def on_finish(self):
        try:
            self.canvas.unsetMapTool(self.tool)
            self.tool.deactivate()
        except Exception:
            pass
        if self.layer:
            self.layer.removeSelection()
        self.loop.quit()
        self.close()

    def on_keep(self):
        # leave tool active but quit loop so script continues
        self.loop.quit()
        self.close()

    def on_clear(self):
        if self.layer:
            self.layer.removeSelection()

# after setting the tool:
if node_layer and _active_tool:
    ctrl = ControllerWidget(iface.mapCanvas(), _active_tool, node_layer)
    ctrl.show()                         # ensure widget is shown
    iface._radius_tool_ctrl = ctrl      # keep reference to avoid GC
    ctrl.loop.exec_()                   # pause further code until user quits the controller

print("Controller closed — continuing script.")
# code here will run only after user pressed Finish or Keep

def run_layer_and_float_dialog(title="Выберите слой и введите дистанцию", label_layer="Слой остановок:", label_value="Максимальная дистанция:", default_value=1.0):
    # Инициализируем окно ввода
    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle(title)
    layout = QtWidgets.QVBoxLayout(dlg)

    # Выбор слоя
    h_layer = QtWidgets.QHBoxLayout()
    lbl_layer = QtWidgets.QLabel(label_layer)
    cb_layers = QtWidgets.QComboBox()

    layers = [lyr for lyr in QgsProject.instance().mapLayers().values()]
    for lyr in layers:
        cb_layers.addItem(lyr.name(), lyr.id())
    h_layer.addWidget(lbl_layer)
    h_layer.addWidget(cb_layers)
    layout.addLayout(h_layer)

    # Ввод дистанции
    h_value = QtWidgets.QHBoxLayout()
    lbl_value = QtWidgets.QLabel(label_value)
    le_value = QtWidgets.QLineEdit(str(default_value))
    h_value.addWidget(lbl_value)
    h_value.addWidget(le_value)
    layout.addLayout(h_value)

    btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    layout.addWidget(btns)

    def accept():
        if not le_value.text():
            QtWidgets.QMessageBox.warning(dlg, "Требуется ввод", "Введите числовое значение")
            return
        dlg.done(QtWidgets.QDialog.Accepted)

    btns.accepted.connect(accept)
    btns.rejected.connect(dlg.reject)

    result = dlg.exec_()
    if result == QtWidgets.QDialog.Accepted:
        layer_name = cb_layers.currentText()
        value = float(le_value.text())
        return layer_name, value
    return None, None

# ввод слоя дорог и значения максимальной дистанции
CONFIG['STOPS_LAYER'], R = run_layer_and_float_dialog(default_value=500)

# инициализируем выходной слой
result_layer = QgsVectorLayer("Polygon?crs=EPSG:3857", "reachable_stops_areas", "memory")
prov = result_layer.dataProvider()


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




def main_worker(point_x, point_y):
    point_x = round(point_x, 2)
    point_y = round(point_y, 2)
    global result_layer, prov
    R = 500
    center_point = QgsPointXY(point_x, point_y)

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
            r_xy_sq = result_R*result_R + dh*dh*1000 # тяжесть подъема
        else:
            r_xy_sq = result_R*result_R - dh*dh*1000 # легкость спуска
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
    result_layer.updateExtents()

for point_x, point_y in STOP_IDS_RESULT:
    threading.Thread(target=main_worker, args=(float(point_x), float(point_y))).start()
    time.sleep(0.1)

while threading.active_count() > 1:
    time.sleep(1)

QgsProject.instance().addMapLayer(result_layer)

symbol = QgsFillSymbol.createSimple({'color': '255,0,0,0'}) # RGBA цвет тут игнорируется 
symbol.setColor(QColor(0, 255, 0)) # цвет слоя
symbol.setOpacity(0.6) # прозрачность слоя

# вывод слоя
renderer = QgsSingleSymbolRenderer(symbol)
result_layer .setRenderer(renderer)
result_layer.triggerRepaint()


print("Готово!")
