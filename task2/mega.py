from qgis.core import (  # Импорт основных классов QGIS для работы с проектами, слоями, геометриями
    QgsProject,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsSpatialIndex,
    QgsSymbol,
    QgsSingleSymbolRenderer,
    QgsWkbTypes,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform
)
from qgis.analysis import (  # Импорт модулей анализа для построения сетевых графов
    QgsVectorLayerDirector,
    QgsNetworkDistanceStrategy,
    QgsGraphAnalyzer,
    QgsGraphBuilder,
)
from qgis.gui import QgsMapToolEmitPoint, QgsVertexMarker  # Импорт GUI-компонентов для интерактивной работы
import math  # Математические функции для геометрических операций
from PyQt5.QtGui import QColor  # Работа с цветами для визуализации
from PyQt5.QtCore import Qt  # Константы Qt для настроек отображения
from PyQt5.QtWidgets import QInputDialog  # Диалоговые окна для ввода данных

CONFIG = {  # Конфигурация имен слоев проекта
    "BUILDINGS_POPULATION": "Здания_насел_attract",
    "HARDWARE": "Highway_OSM_Irkutsk",
    "EDGE_UDS": "УДС_link",
}

VEL_LIMITS_CFG = [  # Конфигурация порогов доступности для велосипеда
    {
        "dist": 2490.0,
        "buf": 50.0,
        "name": "Access_Green_10min",
        "color": QColor(0, 255, 0, 150),
    },
    {
        "dist": 4980.0,
        "buf": 75.0,
        "name": "Access_Yellow_20min",
        "color": QColor(255, 255, 0, 150),
    },
    {
        "dist": 7470,
        "buf": 100.0,
        "name": "Access_Red_30min",
        "color": QColor(255, 0, 0, 150),
    },
]

LIMITS_CFG = [  # Конфигурация порогов доступности для пешехода
    {
        "dist": 840.0,
        "buf": 50.0,
        "name": "Access_Green_10min",
        "color": QColor(0, 255, 0, 150),
    },
    {
        "dist": 1680.0,
        "buf": 75.0,
        "name": "Access_Yellow_20min",
        "color": QColor(255, 255, 0, 150),
    },
    {
        "dist": 2520.0,
        "buf": 100.0,
        "name": "Access_Red_30min",
        "color": QColor(255, 0, 0, 150),
    },
]
CAR_LIMITS_CFG = [  # Конфигурация порогов доступности для автомобиля/такси
    {
        "dist": 600,
        "buf": 50.0,
        "name": "Access_Green_10min",
        "color": QColor(0, 255, 0, 150),
    },
    {
        "dist": 1200,
        "buf": 75.0,
        "name": "Access_Yellow_20min",
        "color": QColor(255, 255, 0, 150),
    },
    {
        "dist": 1800,
        "buf": 100.0,
        "name": "Access_Red_30min",
        "color": QColor(255, 0, 0, 150),
    },
]
MAX_GAP_JUMP = 25.0  # Максимальный разрыв для соединения узлов пешеходной сети


class TimeBasedStrategy(QgsNetworkStrategy):
    def __init__(self, speed_attribute, default_speed=50.0):
        super().__init__()
        self.speed_attribute = speed_attribute  # Атрибут скорости из данных
        self.default_speed = default_speed  # Скорость по умолчанию
    def cost(self, distance, feature):
        length = distance  # Длина ребра графа
        if feature.isValid():
            if feature.fieldNameIndex("LENGTH") >= 0:
                length_value = feature["LENGTH"]  # Попытка взять длину из атрибутов
                if length_value is not None:
                    try:
                        length_val = float(length_value)
                        if length_val > 0:
                            length = length_val
                    except:
                        pass

        speed_km_h = self.default_speed  # Инициализация скорости

        if self.speed_attribute and feature.isValid():
            if feature.fieldNameIndex(self.speed_attribute) >= 0:
                value = feature[self.speed_attribute]  # Получение значения скорости
                if value is not None:
                    parsed_speed = parse_speed(value)  # Парсинг строки скорости
                    if parsed_speed > 0:
                        speed_km_h = parsed_speed
        if speed_km_h <= 0:
            return float('inf')  # Нулевая скорость делает ребро недоступным

        speed_m_s = speed_km_h * 1000 / 3600  # Конвертация км/ч в м/с

        return length / speed_m_s  # Время прохождения ребра


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])  # Векторное произведение для QuickHull

def parse_speed(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).lower().replace("km/h", "").replace(",", ".").strip()  # Очистка строки скорости
    try:
        return float(s)
    except:
        return 0.0

def create_polygon_from_points(points):
    if len(points) < 3:
        return QgsGeometry()  # Недостаточно точек для полигона

    polygon_points = points.copy()
    polygon_points.append(points[0])  # Замыкание полигона
    polygon_geom = QgsGeometry.fromPolygonXY([polygon_points])

    if not polygon_geom.isGeosValid():
        polygon_geom = polygon_geom.makeValid()  # Исправление геометрии

    return polygon_geom

def quickhull_2d(points):
    if len(points) <= 3:
        return points

    formatted_points = []
    for p in points:
        if isinstance(p, (list, tuple)):
            formatted_points.append([float(p[0]), float(p[1])])
        elif isinstance(p, QgsPointXY):
            formatted_points.append([float(p.x()), float(p.y())])
        elif hasattr(p, 'x') and hasattr(p, 'y'):
            formatted_points.append([float(p.x()), float(p.y())])
        else:
            continue
    if len(formatted_points) <= 3:
        return formatted_points
    left = min(formatted_points, key=lambda p: p[0])  # Самая левая точка
    right = max(formatted_points, key=lambda p: p[0])  # Самая правая точка

    hull = set()
    hull.add(tuple(left))
    hull.add(tuple(right))
    left_points = []
    right_points = []

    for p in formatted_points:
        if p == left or p == right:
            continue
        cross_val = cross(left, right, p)
        if cross_val > 0:
            left_points.append(p)  # Точки слева от линии
        else:
            right_points.append(p)  # Точки справа от линии

    def find_hull(points_list, p1, p2, side_points):
        if not side_points:
            return
        max_dist = -1
        farthest = None

        for p in side_points:
            dist = abs(cross(p1, p2, p))  # Расстояние до линии
            if dist > max_dist:
                max_dist = dist
                farthest = p  # Наиболее удаленная точка

        hull.add(tuple(farthest))

        new_points1 = []
        new_points2 = []

        for p in side_points:
            if p == farthest:
                continue
            if cross(p1, farthest, p) > 0:
                new_points1.append(p)
            elif cross(farthest, p2, p) > 0:
                new_points2.append(p)

        find_hull(points_list, p1, farthest, new_points1)  # Рекурсивный вызов для левой части
        find_hull(points_list, farthest, p2, new_points2)  # Рекурсивный вызов для правой части

    find_hull(formatted_points, left, right, left_points)
    find_hull(formatted_points, right, left, right_points)
    hull_list = [list(p) for p in hull]

    if len(hull_list) > 2:
        center_x = sum(p[0] for p in hull_list) / len(hull_list)
        center_y = sum(p[1] for p in hull_list) / len(hull_list)

        def angle_from_center(point):
            return math.atan2(point[1] - center_y, point[0] - center_x)  # Угол для сортировки

        hull_list.sort(key=angle_from_center, reverse=True)  # Сортировка по углу

    return hull_list


def run_accessibility_analysis_max_opt(start_point, selected_mode, start_speed):
    project = QgsProject.instance()

    buildings_layer_list = project.mapLayersByName(CONFIG["BUILDINGS_POPULATION"])
    flag = False  # Флаг типа сети (пешеход/велосипед vs авто)

    if selected_mode == "foot" or selected_mode == "bicycle":
        walk_layers = project.mapLayersByName(CONFIG["HARDWARE"])  # Слой пешеходных дорог
        flag = True
    elif selected_mode == "car" or selected_mode == "taxi":
        walk_layers = project.mapLayersByName(CONFIG["EDGE_UDS"])  # Слой автодорог
    else:
        walk_layers = project.mapLayersByName(CONFIG["HARDWARE"])
        flag = True

    if not buildings_layer_list or not walk_layers:
        print(f"ошибка: Не найдены необходимые слои!")
        return

    walk_layer = walk_layers[0]
    buildings_layer_original = buildings_layer_list[0]
    crs = walk_layer.crs()
    project_crs = QgsProject.instance().crs()

    transform_buildings = QgsCoordinateTransform(buildings_layer_original.crs(), crs, project)
    transformed_features = []

    for feature in buildings_layer_original.getFeatures():
        if feature.hasGeometry():
            new_feature = QgsFeature(feature)
            geom = feature.geometry()
            try:
                geom.transform(transform_buildings)  # Трансформация в CRS сети
                new_feature.setGeometry(geom)
                transformed_features.append(new_feature)
            except Exception as e:
                print(f"Failed to transform feature {feature.id()}: {e}")

    if len(transformed_features) == 0:
        buildings_layer = buildings_layer_original
    else:
        first_geom = transformed_features[0].geometry()
        layer_type = "Point"
        if first_geom:
            if first_geom.type() == QgsWkbTypes.PolygonGeometry:
                layer_type = "Polygon"
            elif first_geom.type() == QgsWkbTypes.LineGeometry:
                layer_type = "LineString"

        buildings_layer = QgsVectorLayer(f"{layer_type}?crs={crs.authid()}",
                                         "Buildings_Transformed", "memory")  # Временный слой зданий
        buildings_layer.dataProvider().addFeatures(transformed_features)

    if buildings_layer.featureCount() == 0:
        skip_building_analysis = True
    else:
        skip_building_analysis = False
    if crs.authid() != project_crs.authid():
        transform_point = QgsCoordinateTransform(project_crs, crs, project)
        try:
            start_point_transformed = transform_point.transform(start_point)  # Трансформация стартовой точки
            start_point = start_point_transformed
        except Exception as e:
            print(f"Transformation error: {e}")
            start_point = walk_layer.extent().center()
    if not walk_layer.extent().contains(start_point):
        distance_to_extent = walk_layer.extent().distance(start_point)
    director = QgsVectorLayerDirector(
        walk_layer, -1, "", "", "", QgsVectorLayerDirector.DirectionBoth  # Настройка директора сети
    )

    if flag:
        strategy = QgsNetworkDistanceStrategy()  # Стратегия по расстоянию
    else:
        default_speed_val = start_speed if start_speed > 0 else 50.0
        strategy = TimeBasedStrategy("V0PRT", default_speed=default_speed_val)  # Стратегия по времени

    director.addStrategy(strategy)
    builder = QgsGraphBuilder(crs)

    try:
        director.makeGraph(builder, [])  # Построение графа
        graph = builder.graph()
        vertex_count = graph.vertexCount()
    except Exception as e:
        print(f"ошибка при построении графа: {e}")
        import traceback
        traceback.print_exc()
        return

    all_graph_points = [graph.vertex(i).point() for i in range(vertex_count)]
    feature_list = []
    for i, pt in enumerate(all_graph_points):
        f = QgsFeature(i)
        f.setGeometry(QgsGeometry.fromPointXY(pt))
        feature_list.append(f)

    graph_index = QgsSpatialIndex()  # Пространственный индекс вершин
    for f in feature_list:
        graph_index.addFeature(f)

    edges_added = 0
    for i in range(vertex_count):
        v = graph.vertex(i)
        if len(v.outgoingEdges()) > 1:
            continue
        v_pt = all_graph_points[i]
        nearest_ids = graph_index.nearestNeighbor(v_pt, 5)  # Поиск ближайших вершин
        for n_id in nearest_ids:
            if n_id == i:
                continue
            dist = v_pt.distance(all_graph_points[n_id])
            if flag:
                if dist <= MAX_GAP_JUMP:  # Соединение разрывов в пешеходной сети
                    graph.addEdge(i, n_id, [dist])
                    graph.addEdge(n_id, i, [dist])
                    edges_added += 1
                    break
            else:
                if dist <= 2:  # Соединение разрывов в автомобильной сети
                    graph.addEdge(i, n_id, [dist])
                    graph.addEdge(n_id, i, [dist])
                    edges_added += 1
                    break

    entry_node_indices = set()
    if not skip_building_analysis and buildings_layer.featureCount() > 0:
        for feature in buildings_layer.getFeatures():
            if not feature.hasGeometry():
                continue
            geometry = feature.geometry()
            if geometry.type() == QgsWkbTypes.PointGeometry:
                building_point = geometry.asPoint()
            else:
                building_point = geometry.centroid().asPoint()  # Центроид здания
            nearest_ids = graph_index.nearestNeighbor(building_point, 1)
            if nearest_ids:
                entry_node_indices.add(nearest_ids[0])  # Точки входа от зданий
    nearest_ids = graph_index.nearestNeighbor(start_point, 1)
    if not nearest_ids:
        nearest_ids = graph_index.nearestNeighbor(start_point, 5, 1000)  # Расширенный поиск стартовой вершины
        if not nearest_ids:
            start_point = walk_layer.extent().center()
            nearest_ids = graph_index.nearestNeighbor(start_point, 1)
    if nearest_ids:
        start_vertex_id = nearest_ids[0]
        start_vertex_point = all_graph_points[start_vertex_id]
    else:
        return

    try:
        (tree, costs) = QgsGraphAnalyzer.dijkstra(graph, start_vertex_id, 0)  # Алгоритм Дейкстры
        reachable_count = sum(1 for cost in costs if cost != float('inf'))
        if reachable_count > 0:
            valid_costs = [c for c in costs if c != float('inf')]
            min_cost = min(valid_costs) if valid_costs else 0
            max_cost = max(valid_costs) if valid_costs else 0
    except Exception as e:
        print(f"ошибка при выполнении алгоритма Дейкстры: {e}")
        import traceback
        traceback.print_exc()
        return
    building_distances = []
    if len(entry_node_indices) > 0:
        for building_vertex_id in entry_node_indices:
            distance_to_building = costs[building_vertex_id]
            if distance_to_building != float('inf'):
                building_distances.append((building_vertex_id, distance_to_building))  # Расстояния до зданий
    sorted_limits = None
    if flag:
        if selected_mode == "foot":
            sorted_limits = sorted(LIMITS_CFG, key=lambda x: x["dist"])  # Пороги для пешехода
        else:
            sorted_limits = sorted(VEL_LIMITS_CFG, key=lambda x: x["dist"])  # Пороги для велосипеда
    else:
        sorted_limits = sorted(CAR_LIMITS_CFG, key=lambda x: x["dist"])  # Пороги для автомобиля
    buckets = [[] for _ in range(len(sorted_limits))]
    max_dist = sorted_limits[-1]["dist"]
    segments_collected = 0
    for i in range(len(costs)):
        cost = costs[i]
        if cost > max_dist or cost == float("inf"):
            continue
        edge_id = tree[i]
        if edge_id == -1:
            continue
        edge = graph.edge(edge_id)
        segment = [all_graph_points[edge.fromVertex()], all_graph_points[edge.toVertex()]]  # Сегмент ребра
        for b_idx, lim in enumerate(sorted_limits):
            if cost <= lim["dist"]:
                buckets[b_idx].append(segment)  # Распределение сегментов по корзинам
                segments_collected += 1
                break
    layers_created = 0
    for i in range(len(sorted_limits) - 1, -1, -1):  # Обработка от дальних к ближним зонам
        cfg = sorted_limits[i]
        segments = buckets[i]
        if not segments:
            continue

        try:
            multi_line = QgsGeometry.fromMultiPolylineXY(segments)
            buffered = multi_line.buffer(cfg["buf"], 4)  # Буферизация линий

            if buffered and not buffered.isNull():
                if buffered.isGeosValid():
                    polygon_geom = buffered.convexHull()  # Выпуклая оболочка

                    if polygon_geom and not polygon_geom.isNull() and polygon_geom.isGeosValid():
                        layer_name = f"{cfg['name']}_{selected_mode}"
                        if not flag and start_speed > 0:
                            layer_name += f"_{int(start_speed)}kmh"
                        layer = QgsVectorLayer(f"Polygon?crs={crs.authid()}", layer_name, "memory")  # Слой доступности
                        feat = QgsFeature()
                        feat.setGeometry(polygon_geom)
                        layer.dataProvider().addFeatures([feat])
                        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
                        symbol.setColor(cfg["color"])
                        symbol.setOpacity(0.6)
                        if symbol.symbolLayer(0):
                            symbol.symbolLayer(0).setStrokeStyle(Qt.NoPen)  # Без контура
                        layer.setRenderer(QgsSingleSymbolRenderer(symbol))
                        project.addMapLayer(layer)
                        layers_created += 1
        except Exception as e:
            print(f"ошибка при создании полигона для лимита {cfg['dist']}: {e}")
            import traceback
            traceback.print_exc()
    try:
        analysis_layer = QgsVectorLayer(f"Point?crs={crs.authid()}", f"Start_Point_{selected_mode}", "memory")  # Слой стартовой точки
        analysis_feat = QgsFeature()
        analysis_feat.setGeometry(QgsGeometry.fromPointXY(start_vertex_point))
        analysis_layer.dataProvider().addFeatures([analysis_feat])
        symbol = QgsSymbol.defaultSymbol(analysis_layer.geometryType())
        symbol.setColor(QColor(0, 255, 0))
        symbol.setSize(8)
        analysis_layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        project.addMapLayer(analysis_layer)
    except Exception as e:
        print(f"ошибка при создании стартовой точки: {e}")
class AccessibilityMapTool(QgsMapToolEmitPoint):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.start_point = None
        self.selected_mode = None  # Выбранный режим передвижения
        self.markers = []
        self.speed = None

    def deactivate(self):
        self.reset()
        super().deactivate()

    def reset(self):
        if self.markers:
            for marker in self.markers:
                try:
                    self.canvas.scene().removeItem(marker)  # Удаление маркеров с канваса
                except:
                    pass
            self.markers = []
        self.start_point = None
        self.selected_mode = None
        self.speed = None

    def canvasReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        point_canvas = self.toMapCoordinates(event.pos())
        if self.start_point is None:
            self.reset()
            self.start_point = point_canvas  # Установка стартовой точки
            self.add_marker(point_canvas, QColor(0, 200, 0))
            print(f"Стартовая точка: {self.start_point.x():.2f}, {self.start_point.y():.2f}")

        elif self.selected_mode is None:
            items = ["Пешеход", "Велосипед", "Автомобиль", "Такси"]
            item, ok = QInputDialog.getItem(
                None, "Выбор транспорта", "Как будем двигаться?", items, 0, False
            )
            if ok and item:
                mode_map = {
                    "Пешеход": "foot",
                    "Велосипед": "bicycle",
                    "Автомобиль": "car",
                    "Такси": "taxi"
                }
                self.selected_mode = mode_map.get(item, "foot")
                if self.selected_mode == "foot" or self.selected_mode == "bicycle":
                    run_accessibility_analysis_max_opt(self.start_point, self.selected_mode, 0)  # Запуск для немоторизованных
                    self.reset()
                else:
                    print("можете ввести скорость")
        else:
            if self.selected_mode == "car" or self.selected_mode == "taxi":
                number, ok = QInputDialog.getDouble(
                    None,  # parent
                    f"скорость ({self.selected_mode})",
                    f"введите скорость движения (км/ч):",
                    50,
                    0,
                    200,
                    1
                )
                if ok:
                    try:
                        run_accessibility_analysis_max_opt(self.start_point, self.selected_mode, number)  # Запуск с заданной скоростью
                    except Exception as e:
                        print(f"ошибка при анализе доступности: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("Ввод скорости отменен")
                self.reset()

    def add_marker(self, point, color):
        marker = QgsVertexMarker(self.canvas)
        marker.setCenter(point)
        marker.setColor(color)
        marker.setIconType(QgsVertexMarker.ICON_CROSS)  # Крестообразный маркер
        marker.setIconSize(10)
        marker.setPenWidth(3)
        self.markers.append(marker)
try:
    if iface:
        tool = AccessibilityMapTool(iface.mapCanvas())
        iface.mapCanvas().setMapTool(tool)  # Активация инструмента
except Exception as e:
    print(f"ошибка при инициализации инструмента: {e}")
    import traceback
    traceback.print_exc()