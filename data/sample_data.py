"""
Sample data for Vehicle Routing Problem demonstrations
"""

# Sample locations for the Vehicle Routing Problem
# Depot is the first location, others are delivery points
sample_locations = [
    {"name": "Distribution Center (Depot)", "lat": 40.7128, "lon": -74.0060},  # NYC (Depot)
    {"name": "Customer A", "lat": 40.7282, "lon": -73.7949},  # Location near NYC
    {"name": "Customer B", "lat": 40.6782, "lon": -73.9442},  # Brooklyn
    {"name": "Customer C", "lat": 40.7831, "lon": -73.9712},  # Upper Manhattan
    {"name": "Customer D", "lat": 40.7609, "lon": -73.9839},  # Midtown
    {"name": "Customer E", "lat": 40.7214, "lon": -74.0052},  # Downtown
    {"name": "Customer F", "lat": 40.6937, "lon": -73.9851},  # Downtown Brooklyn
    {"name": "Customer G", "lat": 40.7466, "lon": -73.9254},  # Long Island City
    {"name": "Customer H", "lat": 40.8448, "lon": -73.8648},  # Bronx
    {"name": "Customer I", "lat": 40.7789, "lon": -73.9692},  # Upper East Side
    {"name": "Customer J", "lat": 40.7484, "lon": -73.9857},  # Empire State
    {"name": "Customer K", "lat": 40.7127, "lon": -74.0134},  # Financial District
    {"name": "Customer L", "lat": 40.7295, "lon": -73.9065},  # Williamsburg
    {"name": "Customer M", "lat": 40.6872, "lon": -73.9418},  # Prospect Park
    {"name": "Customer N", "lat": 40.8120, "lon": -73.9259},  # Harlem
    {"name": "Customer O", "lat": 40.7769, "lon": -73.9530},  # Central Park East
]

# Distance matrix example for a small problem
# This is a 5x5 matrix (depot + 4 locations)
sample_small_distance_matrix = [
    [0, 10, 15, 20, 25],   # From depot
    [10, 0, 35, 25, 30],   # From location 1
    [15, 35, 0, 30, 20],   # From location 2
    [20, 25, 30, 0, 15],   # From location 3
    [25, 30, 20, 15, 0]    # From location 4
]

# A more complex example for 10 locations (depot + 9 locations)
sample_distance_matrix = [
    # Depot, L1, L2, L3, L4, L5, L6, L7, L8, L9
    [0, 12, 19, 31, 22, 17, 23, 29, 14, 11],  # From Depot
    [12, 0, 15, 37, 21, 28, 36, 33, 25, 18],  # From Location 1
    [19, 15, 0, 25, 19, 9, 15, 22, 18, 23],   # From Location 2
    [31, 37, 25, 0, 14, 28, 31, 42, 22, 29],  # From Location 3
    [22, 21, 19, 14, 0, 11, 12, 15, 27, 14],  # From Location 4
    [17, 28, 9, 28, 11, 0, 16, 27, 21, 19],   # From Location 5
    [23, 36, 15, 31, 12, 16, 0, 15, 16, 29],  # From Location 6
    [29, 33, 22, 42, 15, 27, 15, 0, 21, 24],  # From Location 7
    [14, 25, 18, 22, 27, 21, 16, 21, 0, 13],  # From Location 8
    [11, 18, 23, 29, 14, 19, 29, 24, 13, 0]   # From Location 9
]

# Vehicle capacities example
sample_vehicle_capacities = [100, 100, 100, 100]

# Demand example (customer requirements)
sample_demands = [0, 15, 20, 18, 17, 22, 15, 19, 23, 21]  # Depot has 0 demand

# Time windows example - (start_time, end_time) for each location
sample_time_windows = [
    (0, 120),    # Depot
    (10, 50),    # Location 1
    (30, 70),    # Location 2
    (15, 60),    # Location 3
    (40, 80),    # Location 4
    (25, 75),    # Location 5
    (20, 65),    # Location 6
    (35, 85),    # Location 7
    (45, 90),    # Location 8
    (30, 95)     # Location 9
]

# Service times at each location
sample_service_times = [0, 10, 15, 12, 8, 10, 5, 7, 9, 11]  # Depot has 0 service time
