"""
Data handling utilities for the Vehicle Routing Problem
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance

def generate_distance_matrix(coordinates):
    """
    Generates a distance matrix from a list of coordinates.
    
    Args:
        coordinates: List of (lat, lon) tuples for each location
    
    Returns:
        distance_matrix: 2D numpy array of distances between locations
    """
    num_locations = len(coordinates)
    distance_matrix = np.zeros((num_locations, num_locations))
    
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                # Use haversine distance for geographical coordinates
                lat1, lon1 = coordinates[i]
                lat2, lon2 = coordinates[j]
                distance_matrix[i][j] = haversine_distance(lat1, lon1, lat2, lon2)
    
    return distance_matrix

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    
    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point
    
    Returns:
        distance: Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r

def get_random_coordinates(num_locations, seed=None):
    """
    Generates random geographic coordinates within a reasonable range.
    
    Args:
        num_locations: Number of locations to generate
        seed: Random seed for reproducibility
    
    Returns:
        coordinates: List of (lat, lon) tuples
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate coordinates centered around a point
    # (Using arbitrary center point)
    center_lat, center_lon = 40.7128, -74.0060  # New York City
    
    # Generate points within approximately a 50km radius
    radius = 0.5  # Roughly in degrees
    
    coordinates = []
    for _ in range(num_locations):
        # Random offset from center
        lat = center_lat + np.random.uniform(-radius, radius)
        lon = center_lon + np.random.uniform(-radius, radius)
        coordinates.append((lat, lon))
    
    return coordinates

def create_data_model(distance_matrix, num_vehicles, depot=0):
    """
    Creates a data model dictionary for the VRP solvers.
    
    Args:
        distance_matrix: 2D array of distances between locations
        num_vehicles: Number of vehicles available
        depot: Index of the depot location (default: 0)
    
    Returns:
        data_model: Dictionary containing the problem data
    """
    data_model = {}
    data_model['distance_matrix'] = distance_matrix
    data_model['num_vehicles'] = num_vehicles
    data_model['depot'] = depot
    
    return data_model

def load_csv_data(file_path):
    """
    Loads location data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        coordinates: List of (lat, lon) tuples
        location_names: List of location names
    """
    df = pd.read_csv(file_path)
    
    # Extract required columns
    required_cols = ['name', 'latitude', 'longitude']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    location_names = df['name'].tolist()
    coordinates = list(zip(df['latitude'], df['longitude']))
    
    return coordinates, location_names

def export_results(routes, objective_value, location_names, method_name, file_path):
    """
    Exports the VRP solution results to a CSV file.
    
    Args:
        routes: List of routes, where each route is a list of location indices
        objective_value: Total distance of the solution
        location_names: List of location names
        method_name: Name of the solution method
        file_path: Path to save the CSV file
    """
    results = []
    
    for vehicle_id, route in enumerate(routes):
        route_distance = 0
        
        for i in range(len(route) - 1):
            results.append({
                'method': method_name,
                'vehicle_id': vehicle_id + 1,
                'step': i + 1,
                'from_location': location_names[route[i]],
                'to_location': location_names[route[i+1]],
                'from_index': route[i],
                'to_index': route[i+1]
            })
    
    results_df = pd.DataFrame(results)
    results_df['total_objective_value'] = objective_value
    
    results_df.to_csv(file_path, index=False)
    
    return results_df
