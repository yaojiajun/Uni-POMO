import torch
def heuristics_v2(all_nodes_xy: torch.Tensor, current_node: torch.Tensor, delivery_node_demands: torch.Tensor, time_windows: torch.Tensor, current_load: torch.Tensor, current_time: torch.Tensor, pickup_node_demands: torch.Tensor) -> torch.Tensor:
    customer_nodes_xy = all_nodes_xy[1:, :]
    
    # Calculate pairwise distances from current nodes to potential customer nodes
    distances_to_customers = torch.cdist(customer_nodes_xy, all_nodes_xy, p=2)
    
    # Randomly shuffle the distances for enhanced randomness
    shuffled_indices = torch.randperm(distances_to_customers.size(0))
    distances_to_customers = distances_to_customers[shuffled_indices]
    
    # Introduce randomness with problem-specific knowledge
    random_scores = torch.randn(distances_to_customers.size()) + torch.rand(1) * delivery_node_demands
    
    # Incorporate randomness in the heuristic score matrix
    heuristic_scores = distances_to_customers + random_scores
    
    return heuristic_scores
