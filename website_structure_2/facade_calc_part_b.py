import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString
import pandas as pd
import pyproj
import requests
import tensorflow as tf
import joblib
import os

# Correctly specify the model and scaler paths
model_path = r"C:\Users\Mohammad\OneDrive\IaaC\M3\Studio\Forked Repo\AIAStudioG03\models\studio_trained_controlled.h5"
scaler_path = r"C:\Users\Mohammad\OneDrive\IaaC\M3\Studio\Forked Repo\AIAStudioG03\models\Sscaler_nooutlier.pkl"

# Load the model and scaler
model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

def get_building_info(lat, lon):
    # Create a bounding box around the given coordinates
    distance = 1000  # distance in meters for the bounding box
    bbox = ox.utils_geo.bbox_from_point(point=(lat, lon), dist=distance)

    # Load the OSMnx graph for the bounding box
    G = ox.graph.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], network_type='all')
    buildings = ox.geometries_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], tags={'building': True})

    # Calculate the distance from the given point to all building centroids and find the closest building
    buildings['centroid'] = buildings.centroid
    given_point = Point(lon, lat)
    buildings['distance'] = buildings['centroid'].apply(lambda x: x.distance(given_point))
    closest_building = buildings.loc[buildings['distance'].idxmin()]

    # Retrieve the corresponding building polygon
    building_polygon = closest_building['geometry']

    # Define the CRS (Coordinate Reference System) based on the given latitude and longitude
    british_national_grid_crs = 'EPSG:27700'

    # Convert the closest building and street network to the British National Grid CRS
    closest_building_gdf = gpd.GeoDataFrame([closest_building], crs=buildings.crs).set_geometry('geometry')
    closest_building_gdf = closest_building_gdf.to_crs(british_national_grid_crs)
    street_edges = ox.graph_to_gdfs(G, nodes=False).to_crs(british_national_grid_crs)

    # Extract edges for the closest building
    closest_building_edges = extract_edges(closest_building_gdf.geometry.iloc[0])

    # Buffer the street edges
    buffered_street_edges = street_edges.buffer(10)

    # Calculate street-facing edges
    street_facing_edges = calculate_street_facing_edges(closest_building_edges, buffered_street_edges)

    # Mapping dictionary for building categories
    building_type_mapping = {
        'house': 'residential',
        'apartments': 'residential',
        'residential': 'residential',
        'semidetached_house': 'residential',
        'terrace': 'residential',
        'dormitory': 'residential',
        'retail': 'retail',
        'commercial': 'commercial',
        'detached': 'residential',
        'garages': 'other',
        'office': 'office',
        'university': 'education',
        'school': 'education',
        'garage': 'other',
        'roof': 'other',
        'church': 'religious',
        'shed': 'other',
        'service': 'other',
        'industrial': 'industrial',
        'train_station': 'transport',
        'hotel': 'hospitality',
        'pub': 'hospitality',
        'air_shaft': 'other',
        'warehouse': 'industrial',
        'hospital': 'healthcare',
        'construction': 'other',
        'public': 'institution',
        'bridge': 'transport',
        'college': 'education',
        'kiosk': 'commercial',
        'civic': 'institution',
        'block': 'other',
        'no': 'other',
        'healthcare': 'healthcare',
        'bunker': 'other',
        'toilets': 'public',
        'hall_of_residence': 'education',
        'restaurant': 'hospitality',
        'kindergarten': 'education',
        'greenhouse': 'other',
        'conservatory': 'other',
        'tower': 'other',
        'hut': 'other',
        'museum': 'institution',
        'presbytery': 'religious',
        'outbuilding': 'other',
        'chapel': 'religious',
        'silo': 'industrial',
        'cafe': 'hospitality',
        'sports_centre': 'sports',
        'multiple': 'other',
        'air_vent': 'other',
        'commerical': 'commercial',
        'container': 'other',
        'student_residence': 'education',
        'shelter': 'public',
        'ruins': 'other',
        'substation': 'other',
        'transportation': 'transport',
        'balcony': 'other',
        'council_flats': 'residential',
        'disused_station': 'transport',
        'portacabins': 'other',
        'cinema': 'hospitality',
        'boathouse': 'other',
        'artists_studio': 'institution',
        'chimney': 'other',
        'vent_shaft': 'other',
        'library': 'institution',
        'gatehouse': 'institution',
        'sports_hall': 'sports',
        'convent': 'religious',
    }

    # Apply the mapping to categorize the closest building type
    closest_building_category = building_type_mapping.get(closest_building['building'], 'other')
    closest_building_gdf['building_category'] = closest_building_category

    # Function to prepare features and predict building height
    def predict_building_height(building):
        features = pd.DataFrame({
            "Building_FootprintArea": [building['geometry'].area],
            "Category_commercial": [1 if building.get('building_category') == 'commercial' else 0],
            "Category_education": [1 if building.get('building_category') == 'education' else 0],
            "Category_hospitality": [1 if building.get('building_category') == 'hospitality' else 0],
            "Category_industrial": [1 if building.get('building_category') == 'industrial' else 0],
            "Category_office": [1 if building.get('building_category') == 'office' else 0],
            "Category_institution": [1 if building.get('building_category') == 'institution' else 0],
            "Category_residential": [1 if building.get('building_category') == 'residential' else 0],
            "Category_retail": [1 if building.get('building_category') == 'retail' else 0],
            "Category_nan": [0],  # Placeholder for missing category
            "latitude": [building['centroid'].y],
            "longitude": [building['centroid'].x],
            "brick": [0.45779499411582947],
            "ceramic": [0.058589570224285126],
            "glass": [0.03563307225704193],
            "metal": [0.000011294204341538716],
            "paint": [0.015030953101813793],
            "tile": [0.07241785526275635],
            "wood": [0.20852695405483246]
        })

        # Ensure correct order of columns
        features = features[[
            "Building_FootprintArea", "Category_commercial", "Category_education", "Category_hospitality",
            "Category_industrial", "Category_office", "Category_institution", "Category_residential",
            "Category_retail", "Category_nan", "latitude", "longitude", "brick", "ceramic", "glass", "metal",
            "paint", "tile", "wood"
        ]]

        # Debug print to check the features DataFrame
        print("Features DataFrame before scaling:")
        print(features)

        # Scale the features
        scaled_features = scaler.transform(features)

        # Debug print to check the scaled features
        print("Scaled Features DataFrame:")
        print(scaled_features)

        # Predict the height using the loaded model
        height_prediction = model.predict(scaled_features)

        # Debug print to check the predicted height
        print(f"Predicted building height: {height_prediction[0][0]}")

        return height_prediction[0][0]

    # Function to calculate street-facing edge lengths
    def calculate_street_facing_length(edges, buffer_gdf):
        total_length = 0
        for edge in edges:
            if buffer_gdf.intersects(edge).any():
                total_length += edge.length
        return total_length

    # Function to calculate facade area
    def calculate_facade_area(building, edges, buffer_gdf):
        if pd.notnull(building['building:levels']):
            building_levels = float(building['building:levels'])
            building_height = building_levels * 3.0  # Assuming floor-to-floor height is 3 meters
            height_source = 'OSM'
        else:
            # Predict the building height if levels data is not available
            building_height = float(predict_building_height(building))
            if building_height < 3:
                building_height = 3  # Ensure minimum building height is 3 meters
            height_source = 'Prediction Model'

        facade_edge_length = float(calculate_street_facing_length(edges, buffer_gdf))
        facade_area = building_height * facade_edge_length

        print(f"Facade area: {facade_area}, Building height: {building_height}, Height source: {height_source}")
        return facade_area, building_height, height_source

    # Calculate facade area for the closest building
    closest_building_facade_area, closest_building_height, closest_building_height_source = calculate_facade_area(closest_building_gdf.iloc[0], street_facing_edges, buffered_street_edges)

    # Extract edges for the single building
    building_edges = extract_edges(closest_building['geometry'])

    # Store the calculated results from Cell 07 in the closest_building GeoDataFrame
    closest_building_gdf['facade_area'] = closest_building_facade_area
    closest_building_gdf['building_height'] = closest_building_height
    closest_building_gdf['height_source'] = closest_building_height_source

    # Calculate material areas for the closest building
    brick_area = closest_building_facade_area * 0.45779499411582947
    ceramic_area = closest_building_facade_area * 0.058589570224285126
    glass_area = closest_building_facade_area * 0.03563307225704193
    metal_area = closest_building_facade_area * 0.000011294204341538716
    paint_area = closest_building_facade_area * 0.015030953101813793
    tile_area = closest_building_facade_area * 0.07241785526275635
    wood_area = closest_building_facade_area * 0.20852695405483246

    # Round the material areas to two decimal places
    material_areas = {
        'brick': round(brick_area, 2),
        'ceramic': round(ceramic_area, 2),
        'glass': round(glass_area, 2),
        'metal': round(metal_area, 2),
        'paint': round(paint_area, 2),
        'tile': round(tile_area, 2),
        'wood': round(wood_area, 2)
    }

    # Round the footprint area to two decimal places
    footprint_area = round(closest_building_gdf.geometry.iloc[0].area, 2)

    return {
        'osm_id': str(closest_building.name),
        'building_name': closest_building.get('name', 'N/A'),
        'building_category': closest_building_category,
        'building_height': str(round(closest_building_height, 2)),
        'height_source': closest_building_height_source,
        'footprint_area': str(footprint_area),
        'crs': british_national_grid_crs,
        'facade_area': str(round(closest_building_facade_area, 2)),
        'material_areas': material_areas
    }

# Helper functions
def extract_edges(polygon):
    if (polygon.geom_type == 'Polygon'):
        return [LineString([polygon.exterior.coords[i], polygon.exterior.coords[i+1]]) for i in range(len(polygon.exterior.coords)-1)]
    return []

def calculate_street_facing_edges(edges, buffer_gdf):
    street_facing_edges = [edge for edge in edges if buffer_gdf.intersects(edge).any()]
    return street_facing_edges
