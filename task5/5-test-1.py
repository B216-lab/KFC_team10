# -*- coding: utf-8 -*-

import math
import heapq
from collections import defaultdict
from qgis.core import (
    QgsProject,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsSpatialIndex,
    QgsField,
    QgsVectorLayer,
    QgsWkbTypes,
    QgsVectorFileWriter,
    QgsFields
)
from PyQt5.QtCore import QVariant
import processing

NAME_UDS = "УДС_link"
NAME_ZONES = "Расчетные_транспортные_zone"
NAME_BUILDINGS = "Здания_насел_attract"
ATTR_POPULATION = "visits_count"  

# Интервалы и пределы
INTERVAL_STEP = 5  
MAX_TIME_MIN = 30   

COORD_PRECISION = 3  # округление координат узлов для агрегации

# Скорости (км/ч) для каждого режима
SPEEDS = {
    "car": 40.0,
    "transit": 20.0,
    "bike": 15.0,
    "walk": 5.0
}

# Веса для итогового средневзвешенного значения (сумма не обязательно 1 — нормируется)
MODE_WEIGHTS = {
    "car": 1.0,
    "transit": 1.0,
    "bike": 0.8,
    "walk": 0.5
}


def get_layer(name):
    layers = QgsProject.instance().mapLayersByName(name)
    return layers[0] if layers else None


class SimpleGraph:
    """Простой неориентированный взвешенный граф узлов.
       Узлы ключуются по округлённым координатам (tuple).
    """
    def __init__(self, mode="car"):
        self.edges = {}  # dict: coord_tuple -> {neighbor_coord_tuple: weight_min}
        self.mode = mode
        # скорость в метрах в минуту
        self.speed_m_per_min = (SPEEDS[mode] * 1000.0) / 60.0
        self.nodes_spatial_idx = QgsSpatialIndex()
        self.nodes_list = []

    def add_edge(self, p1, p2, length_m):
        u = (round(p1.x(), COORD_PRECISION), round(p1.y(), COORD_PRECISION))
        v = (round(p2.x(), COORD_PRECISION), round(p2.y(), COORD_PRECISION))
        if u == v:
            return
        # вес — время в минутах
        weight_min = length_m / self.speed_m_per_min
        self.edges.setdefault(u, {})
        self.edges.setdefault(v, {})
        # сохраняем минимальный вес на ребре (на случай повторов)
        if v not in self.edges[u] or weight_min < self.edges[u][v]:
            self.edges[u][v] = weight_min
        if u not in self.edges[v] or weight_min < self.edges[v][u]:
            self.edges[v][u] = weight_min

    def build_spatial_index(self):
        """Построить spatial index для узлов (используется для привязки координат)."""
        self.nodes_spatial_idx = QgsSpatialIndex()
        self.nodes_list = list(self.edges.keys())
        for idx, coord in enumerate(self.nodes_list):
            f = QgsFeature()
            f.setId(idx)
            pt = QgsPointXY(coord[0], coord[1])
            f.setGeometry(QgsGeometry.fromPointXY(pt))
            self.nodes_spatial_idx.addFeature(f)

    def get_nearest_node(self, point):
        """Возвращает координату ближайшего узла (tuple) либо None."""
        nn_ids = self.nodes_spatial_idx.nearestNeighbor(point, 1)
        if not nn_ids:
            return None
        idx = nn_ids[0]
        return self.nodes_list[idx]


def dijkstra_with_limit(graph, start_node, max_time):
    """Дейкстра от start_node, возвращает dict: node_coord -> time_min (<= max_time)."""
    pq = [(0.0, start_node)]
    distances = {start_node: 0.0}
    while pq:
        current_dist, u = heapq.heappop(pq)
        if current_dist > distances.get(u, float('inf')):
            continue
        if current_dist > max_time:
            continue
        for v, wt in graph.edges.get(u, {}).items():
            nd = current_dist + wt
            if nd <= max_time and (v not in distances or nd < distances[v]):
                distances[v] = nd
                heapq.heappush(pq, (nd, v))
    return distances


def compute_iso_intervals(distances, node_population_map, interval_step):
    """Группирует узлы по интервалам изохрон (0-interval_step, interval_step-2*interval_step, ...).
       Возвращает dict: interval_upper_bound -> total_visits_in_that_interval
    """
    buckets = defaultdict(float)
    for node_coord, t in distances.items():
        if node_coord not in node_population_map:
            continue
        visits = node_population_map[node_coord]
        # интервал верхней границы (например, t=3 при step=5 -> 5)
        interval_upper = int(math.ceil(t / interval_step)) * interval_step
        if interval_upper == 0:
            interval_upper = interval_step
        buckets[interval_upper] += visits
    return dict(buckets)


def calculate_ta_from_buckets(buckets, interval_step):
    """Вычисляет индекс доступности по интервальному распределению.
       Мы используем простую схему: средневзвешенное значение интервалов, вес = количество посещений.
       В формуле можем использовать верхнюю границу интервала (interval_upper) в минутах.
       Возвращает TA (в минутах) — чем меньше, тем лучше (или можно инвертировать по необходимости).
    """
    if not buckets:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for interval_upper, visits in buckets.items():
        numerator += interval_upper * visits
        denominator += visits
    return numerator / denominator if denominator > 0 else 0.0


def main():
    print("=== Начало расчета транспортной доступности (расширенная версия) ===")
    layer_uds = get_layer(NAME_UDS)
    layer_zones = get_layer(NAME_ZONES)
    layer_blds = get_layer(NAME_BUILDINGS)

    if not layer_uds:
        print(f"ОШИБКА: Слой '{NAME_UDS}' не найден!")
        return None
    if not layer_zones:
        print(f"ОШИБКА: Слой '{NAME_ZONES}' не найден!")
        return None
    if not layer_blds:
        print(f"ОШИБКА: Слой '{NAME_BUILDINGS}' не найден!")
        return None

    print("1) Построение графов для каждого режима...")
    graphs = {}
    # Для каждого режима строим отдельный граф (здесь — использованы одни и те же геометрии, только скорость и вес другие)
    for mode in SPEEDS.keys():
        graphs[mode] = SimpleGraph(mode=mode)

    # Собираем ребра из слоя УДС
    for feat in layer_uds.getFeatures():
        geom = feat.geometry()
        if not geom:
            continue
        # поддержка как многострок, так и одной линии
        if geom.isMultipart():
            multi = geom.asMultiPolyline()
            lines = multi if multi else []
        else:
            single = geom.asPolyline()
            lines = [single] if single else []
        for line in lines:
            for i in range(len(line) - 1):
                p1 = line[i]
                p2 = line[i + 1]
                # расстояние в единицах CRS (предполагаем метры)
                dx = p1.x() - p2.x()
                dy = p1.y() - p2.y()
                dist = math.sqrt(dx * dx + dy * dy)
                # добавляем ребро во все графы
                for g in graphs.values():
                    g.add_edge(p1, p2, dist)

    # Строим индексы узлов для всех графов
    for mode, g in graphs.items():
        print(f"  Строим spatial index для режима: {mode} (узлов: {len(g.edges)})")
        g.build_spatial_index()

    print("2) Привязка посещений/населения к узлам графа...")
    # Проверяем атрибут population
    field_names = [f.name() for f in layer_blds.fields()]
    has_population = ATTR_POPULATION in field_names
    if has_population:
        print(f"  Найден атрибут посещений: '{ATTR_POPULATION}'")
    else:
        print(f"  Атрибут '{ATTR_POPULATION}' не найден — используем значение 1 для каждого здания")

    # Для привязки используем любой граф (координаты узлов одинаковы для всех режимов)
    # Выберем граф 'car' для поиска ближайших узлов
    any_graph = graphs[next(iter(graphs))]
    node_population_map = {}  # node_coord -> sum(population)
    for feat in layer_blds.getFeatures():
        geom = feat.geometry()
        if not geom:
            continue
        # получаем центроид (поддержка multipart)
        try:
            center_pt = geom.centroid().asPoint()
        except Exception:
            # fallback: если полигон мульти - берем centroid первого полигона
            if geom.isMultipart():
                multi = geom.asMultiPolygon()
                if multi and multi[0] and multi[0][0]:
                    poly = QgsGeometry.fromPolygonXY(multi[0])
                    center_pt = poly.centroid().asPoint()
                else:
                    continue
            else:
                continue

        nearest = any_graph.get_nearest_node(center_pt)
        if not nearest:
            continue

        pop = 1.0
        if has_population:
            try:
                val = feat[ATTR_POPULATION]
                if val is not None:
                    pop = float(val)
            except Exception:
                pop = 1.0
        node_population_map[nearest] = node_population_map.get(nearest, 0.0) + pop

    print(f"  Привязано население/посещений к {len(node_population_map)} узлам")

    print("3) Создание результирующего слоя (в памяти)...")
    crs_auth = layer_zones.crs().authid()
    out_layer = QgsVectorLayer(f"Polygon?crs={crs_auth}", "Результат_Доступности_full", "memory")
    provider = out_layer.dataProvider()

    # Копируем все поля из исходного слоя районов (передаём список QgsField)
    provider.addAttributes(list(layer_zones.fields()))
    # Добавляем поля для каждого режима и итоговый
    new_fields = [
        QgsField("TA_car", QVariant.Double),
        QgsField("TA_transit", QVariant.Double),
        QgsField("TA_bike", QVariant.Double),
        QgsField("TA_walk", QVariant.Double),
        QgsField("TA_total", QVariant.Double),
        QgsField("Pop_in_TA", QVariant.Double),
        QgsField("Nodes_in_TA", QVariant.Int)
    ]
    provider.addAttributes(new_fields)
    out_layer.updateFields()

    print("4) Расчёт доступности по районам...")
    features_to_add = []
    total_zones = layer_zones.featureCount()
    for idx, zone in enumerate(layer_zones.getFeatures(), start=1):
        geom = zone.geometry()
        if not geom:
            continue
        new_feat = QgsFeature(out_layer.fields())
        new_feat.setGeometry(geom)
        # копируем исходные атрибуты
        attrs = list(zone.attributes())

        # центр района
        try:
            center_pt = geom.centroid().asPoint()
        except Exception:
            # fallback при проблемах
            center_pt = geom.boundingBox().center()

        # находим стартовый узел (используем any_graph)
        start_node = any_graph.get_nearest_node(center_pt)

        # хранение результатов по режимам
        ta_per_mode = {}
        total_population_in_any_mode = 0.0
        nodes_count_any_mode = 0

        if start_node:
            for mode, g in graphs.items():
                # запускаем Дейкстру для данного режима
                distances = dijkstra_with_limit(g, start_node, MAX_TIME_MIN)
                # группируем по интервалам
                buckets = compute_iso_intervals(distances, node_population_map, INTERVAL_STEP)
                # рассчитываем TA для режима
                ta_val = calculate_ta_from_buckets(buckets, INTERVAL_STEP)
                ta_per_mode[mode] = ta_val

        else:
            # если не найден стартовый узел — все TA остаются 0
            for mode in graphs.keys():
                ta_per_mode[mode] = 0.0

        union_nodes = set()
        for mode, g in graphs.items():
            if start_node:
                distances = dijkstra_with_limit(g, start_node, MAX_TIME_MIN)
                for node_coord in distances.keys():
                    if node_coord in node_population_map:
                        union_nodes.add(node_coord)
        total_population_in_any_mode = sum(node_population_map.get(n, 0.0) for n in union_nodes)
        nodes_count_any_mode = len(union_nodes)

        # вычисляем итоговое средневзвешенное TA (нормируем веса)
        weights = MODE_WEIGHTS.copy()
        # нормировка
        w_sum = sum(weights.values()) if sum(weights.values()) > 0 else 1.0
        ta_total = 0.0
        for mode, w in weights.items():
            ta_total += (w / w_sum) * ta_per_mode.get(mode, 0.0)

        # добавляем результаты в атрибуты (в том же порядке, как добавлены поля)
        attrs.extend([
            ta_per_mode.get("car", 0.0),
            ta_per_mode.get("transit", 0.0),
            ta_per_mode.get("bike", 0.0),
            ta_per_mode.get("walk", 0.0),
            ta_total,
            total_population_in_any_mode,
            nodes_count_any_mode
        ])

        new_feat.setAttributes(attrs)
        features_to_add.append(new_feat)

        if idx % 10 == 0 or idx == total_zones:
            print(f"  Обработано: {idx}/{total_zones} районов")

    # добавляем фичи и обновляем экстенты
    provider.addFeatures(features_to_add)
    out_layer.updateExtents()

    # добавляем слой в проект
    QgsProject.instance().addMapLayer(out_layer)

    output_path = r"C:\temp\результат_доступности_full.shp"
    try:
        params = {
            'INPUT': out_layer,
            'OUTPUT': output_path,
            'FILE_ENCODING': 'UTF-8'
        }
        processing.run("native:savefeatures", params)
        print(f"  Результаты сохранены в: {output_path}")
    except Exception as e:
        print(f"  Не удалось сохранить в файл: {e}\n  Слой добавлен в память проекта.")

    print("=== Расчёт завершён ===")
    return out_layer


# запуск

result_layer = main()
if result_layer:
    print("\nСлой создан успешно:")
    print(f"  Имя: {result_layer.name()}")
    print(f"  Количество объектов: {result_layer.featureCount()}")
    print("  Поля:")
    for f in result_layer.fields():
        print(f"    - {f.name()}: {f.typeName()}")
else:
    print("Не удалось создать результирующий слой.")
