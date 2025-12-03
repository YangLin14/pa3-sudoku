def is_consistent(region, color, assignment, graph):
    """
    Check if assigning a color to a region is consistent with the current assignment.
    
    Args:
        region: The region to check (string)
        color: The color to assign (string)
        assignment: Current partial assignment (dict: region -> color)
        graph: The constraint graph (dict: region -> list of neighbors)
    
    Returns:
        bool: True if assignment is consistent, False otherwise
    
    A region-color assignment is consistent if no neighboring region
    has the same color.
    """
    # TODO: Implement consistency check
    # Check all neighbors of the region
    # If any neighbor is already assigned the same color, return False
    # Otherwise return True
    for neighbor in graph[region]:
        if neighbor in assignment and assignment[neighbor] == color:
            return False
    return True


def backtracking_search(graph, colors):
    """
    Solve the map coloring problem using backtracking search with heuristics.
    
    Args:
        graph: The constraint graph (dict: region -> list of neighbors)
        colors: List of available colors (list of strings)
    
    Returns:
        dict: Complete assignment (region -> color) if solution found
        None: If no solution exists
    
    Use the following heuristics:
    - MRV (Minimum Remaining Values) for variable selection
    - Degree Heuristic as a tiebreaker
    - LCV (Least Constraining Value) for value ordering
    """
    # TODO
    def get_legal_value(region, colors, assignment, graph):
        legal = []
        for color in colors:
            if is_consistent(region, color, assignment, graph):
                legal.append(color)
        return legal
        
    def select_unassigned_value(unassigned, assignment, colors, graph):
        candidates = []
        min_mrv = float('inf')
        
        for region in unassigned:
            num_legal = len(get_legal_value(region, colors, assignment, graph))
            
            if num_legal < min_mrv:
                min_mrv = num_legal
                candidates = [region]
            elif num_legal == min_mrv:
                candidates.append(region)
                
        if len(candidates) == 1:
            return candidates[0]
        
        max_degree = -1
        best_region = None
        for region in candidates:
            degree = sum(1 for n in graph[region] if n not in assignment)
            
            if degree > max_degree:
                max_degree = degree
                best_region = region

        return best_region
    
    def order_domain_values(region, assignment, colors, graph):
        legal_value = get_legal_value(region, colors, assignment, graph)
        scores = []
        for color in legal_value:
            conflict = 0
            for neighbor in graph[region]:
                if neighbor not in assignment:
                    if color in get_legal_value(neighbor, colors, assignment, graph):
                        conflict += 1
            scores.append((conflict, color))
        scores.sort()
        return [color for conflict, color in scores]
        
    def backtrack_recursive(assignment):
        if len(assignment) == len(graph):
            return assignment
            
        unassigned = [r for r in graph if r not in assignment]
        region = select_unassigned_value(unassigned, assignment, colors, graph)
        
        ordered_colors = order_domain_values(region, assignment, colors, graph)
        
        for color in ordered_colors:
            assignment[region] = color
            
            result = backtrack_recursive(assignment.copy())
            if result is not None:
                return result
            
            del assignment[region]
        return None
            
    return backtrack_recursive({})
        
from collections import deque
def ac3(graph, colors):
    """
    Apply AC-3 algorithm to enforce arc consistency.
    
    Args:
        graph: The constraint graph (dict: region -> list of neighbors)
        colors: List of available colors (list of strings)
    
    Returns:
        dict: Reduced domains (region -> list of colors) if consistent
        None: If any domain becomes empty (no solution possible)
    
    Algorithm:
    1. Initialize all domains to full set of colors
    2. Create queue of all arcs (Xi, Xj) where Xj is neighbor of Xi
    3. While queue is not empty:
       - Remove arc (Xi, Xj) from queue
       - If Revise(Xi, Xj) removes values from Xi's domain:
         - If Xi's domain is empty, return None
         - Add all arcs (Xk, Xi) to queue (where Xk is neighbor of Xi, except Xj)
    4. Return reduced domains
    
    Revise(Xi, Xj):
    - Remove value x from Xi's domain if:
      - For all values y in Xj's domain, x == y (constraint violated)
    - Return True if domain was revised, False otherwise
    """
    # TODO
    domain = {region: list(colors) for region in graph}
    
    queue = deque()
    for region in graph:
        for neighbor in graph[region]:
            queue.append((region, neighbor))
    
    def revise(Xi, Xj):
        revised = False
        for x in domain[Xi][:]:
            is_satisfy = False
            for y in domain[Xj]:
                if x != y:
                    is_satisfy = True
                    break
            if not is_satisfy:
                domain[Xi].remove(x)
                revised = True
        return revised
        
    while queue:
        Xi, Xj = queue.popleft()
        
        if revise(Xi, Xj):
            if not domain[Xi]:
                return None
            for Xk in graph[Xi]:
                if Xk != Xj:
                    queue.append((Xk, Xi))
    
    return domain