"""
Classical implementations of Vehicle Routing Problem solvers
"""

import numpy as np
from scipy.spatial import distance
import random

# For OR-Tools
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

def solve_vrp_ortools(data):
    """
    Solves the Vehicle Routing Problem using Google OR-Tools.
    
    Args:
        data: A dictionary containing the problem data:
            - distance_matrix: 2D array with distances between locations
            - num_vehicles: Number of vehicles in the fleet
            - depot: Index of the depot location (usually 0)
    
    Returns:
        routes: A list of routes, where each route is a list of location indices
        objective_value: The total distance of all routes
    """
    if not ORTOOLS_AVAILABLE:
        print("OR-Tools not available, using fallback implementation")
        return solve_vrp_nearest_neighbor(data)
    
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']),
        data['num_vehicles'],
        data['depot']
    )
    
    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)
    
    # Create and register a transit callback
    def distance_callback(from_index, to_index):
        # Returns the distance between the two nodes
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.time_limit.seconds = 5  # Limit runtime for larger problems
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    # Extract solution
    routes = []
    objective_value = 0
    
    if solution:
        for vehicle_id in range(data['num_vehicles']):
            route = []
            index = routing.Start(vehicle_id)
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                index = solution.Value(routing.NextVar(index))
            
            # Add the ending depot
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            
            routes.append(route)
            
        objective_value = solution.ObjectiveValue()
    else:
        # No solution found, return empty routes
        routes = [[] for _ in range(data['num_vehicles'])]
        objective_value = float('inf')
    
    return routes, objective_value

def solve_vrp_genetic_algorithm(data, pop_size=50, generations=100):
    """
    Solves the Vehicle Routing Problem using a Genetic Algorithm.
    
    Args:
        data: A dictionary containing the problem data:
            - distance_matrix: 2D array with distances between locations
            - num_vehicles: Number of vehicles in the fleet
            - depot: Index of the depot location (usually 0)
        pop_size: Size of the population (number of candidate solutions)
        generations: Number of generations to evolve
    
    Returns:
        routes: A list of routes, where each route is a list of location indices
        objective_value: The total distance of all routes
    """
    num_locations = len(data['distance_matrix'])
    num_vehicles = data['num_vehicles']
    depot = data['depot']
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        # Create a random solution
        individuals = list(range(1, num_locations))  # Skip depot (0)
        random.shuffle(individuals)
        
        # Split into routes
        solution = []
        vehicle_capacity = (num_locations - 1) // num_vehicles
        remainder = (num_locations - 1) % num_vehicles
        
        start_idx = 0
        for v in range(num_vehicles):
            route_size = vehicle_capacity + (1 if v < remainder else 0)
            route = [depot] + individuals[start_idx:start_idx + route_size] + [depot]
            solution.append(route)
            start_idx += route_size
        
        population.append(solution)
    
    # Evaluate fitness (lower is better)
    def calculate_fitness(solution):
        total_distance = 0
        for route in solution:
            for i in range(len(route) - 1):
                total_distance += data['distance_matrix'][route[i]][route[i+1]]
        return total_distance
    
    # Tournament selection
    def selection(population, fitnesses):
        tournament_size = 3
        selected = []
        
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
            selected.append(population[winner_idx])
        
        return selected
    
    # Crossover - exchange routes between parents
    def crossover(parent1, parent2):
        if random.random() > 0.7:  # Crossover probability
            return parent1, parent2
        
        child1, child2 = [], []
        crossover_point = random.randint(1, num_vehicles-1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        # Ensure all locations are covered exactly once (excluding depot)
        # This is a simplification - a real GA would need more complex repair mechanisms
        
        return child1, child2
    
    # Mutation - swap two non-depot locations
    def mutation(solution):
        if random.random() > 0.2:  # Mutation probability
            return solution
        
        # Choose a random route
        route_idx = random.randint(0, len(solution)-1)
        route = solution[route_idx]
        
        # Can only mutate if there are at least 2 non-depot locations
        if len(route) > 3:  # Depot appears twice in each route
            loc1_idx = random.randint(1, len(route)-2)  # Skip depots
            loc2_idx = random.randint(1, len(route)-2)
            while loc1_idx == loc2_idx:
                loc2_idx = random.randint(1, len(route)-2)
            
            # Swap locations
            route[loc1_idx], route[loc2_idx] = route[loc2_idx], route[loc1_idx]
        
        return solution
    
    # Main GA loop
    best_solution = None
    best_fitness = float('inf')
    
    for generation in range(generations):
        # Calculate fitness
        fitnesses = [calculate_fitness(solution) for solution in population]
        
        # Find best solution
        min_fitness_idx = fitnesses.index(min(fitnesses))
        current_best = population[min_fitness_idx]
        current_best_fitness = fitnesses[min_fitness_idx]
        
        if current_best_fitness < best_fitness:
            best_solution = current_best
            best_fitness = current_best_fitness
        
        # Selection
        selected = selection(population, fitnesses)
        
        # Create new population
        new_population = []
        
        # Elitism - keep best solution
        new_population.append(current_best)
        
        # Crossover and mutation
        while len(new_population) < pop_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            child1, child2 = crossover(parent1, parent2)
            
            child1 = mutation(child1)
            child2 = mutation(child2)
            
            new_population.append(child1)
            new_population.append(child2)
        
        # Trim if needed
        if len(new_population) > pop_size:
            new_population = new_population[:pop_size]
        
        population = new_population
    
    # Format routes for consistency with other solvers
    formatted_routes = []
    for vehicle_route in best_solution:
        # Remove the duplicate depot at the end
        route = vehicle_route[:-1]
        formatted_routes.append(route)
    
    return formatted_routes, best_fitness

def solve_vrp_nearest_neighbor(data):
    """
    Solves the Vehicle Routing Problem using a Nearest Neighbor heuristic.
    Used as a fallback when OR-Tools is not available.
    
    Args:
        data: A dictionary containing the problem data:
            - distance_matrix: 2D array with distances between locations
            - num_vehicles: Number of vehicles in the fleet
            - depot: Index of the depot location (usually 0)
    
    Returns:
        routes: A list of routes, where each route is a list of location indices
        objective_value: The total distance of all routes
    """
    num_locations = len(data['distance_matrix'])
    num_vehicles = data['num_vehicles']
    depot = data['depot']
    distance_matrix = data['distance_matrix']
    
    # Initialize empty routes
    routes = [[] for _ in range(num_vehicles)]
    
    # Start all routes at the depot
    for route in routes:
        route.append(depot)
    
    # Keep track of unvisited locations
    unvisited = set(range(num_locations))
    unvisited.remove(depot)
    
    # Assign locations to vehicles using nearest neighbor heuristic
    current_vehicle = 0
    
    while unvisited:
        current_route = routes[current_vehicle]
        current_location = current_route[-1]
        
        # Find closest unvisited location
        best_location = None
        best_distance = float('inf')
        
        for location in unvisited:
            dist = distance_matrix[current_location][location]
            if dist < best_distance:
                best_distance = dist
                best_location = location
        
        # Add closest location to route
        current_route.append(best_location)
        unvisited.remove(best_location)
        
        # Move to next vehicle (round-robin assignment)
        current_vehicle = (current_vehicle + 1) % num_vehicles
    
    # Add depot at the end of each route
    for route in routes:
        route.append(depot)
    
    # Calculate total distance
    total_distance = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i+1]]
    
    return routes, total_distance
