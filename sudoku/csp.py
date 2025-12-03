from typing import List, Set, Dict, Tuple, Optional
import copy


def solve(grid: List[List[str]], *, use_mrv: bool = False, use_lcv: bool = False, 
            use_fc: bool = False, use_ac3: bool = False) -> Optional[List[List[str]]]:
    """
    Solve a Sudoku puzzle using CSP backtracking with optional optimizations.
    
    Args:
        grid: 9x9 grid where each cell is a string '0'-'9' ('0' means empty)
        use_mrv: If True, use Minimum Remaining Values heuristic for variable selection
        use_lcv: If True, use Least Constraining Value heuristic for value ordering
        use_fc: If True, use Forward Checking during search
        use_ac3: If True, use AC-3 algorithm for initial constraint propagation
    
    Returns:
        Solved 9x9 grid as list of lists of strings, or None if no solution exists
    
    Example:
        >>> grid = [['5','3','0', ...], ...]  # 0 represents empty cells
        >>> solution = solve(grid, use_mrv=True, use_fc=True)
    """
    # Initialize the CSP
    csp = SudokuCSP(grid)
    
    # Apply AC-3 if requested for initial constraint propagation
    if use_ac3:
        if not csp.ac3():
            return None  # No solution exists
        
    # Solve using backtracking with specified heuristics
    if csp.backtrack(use_mrv=use_mrv, use_lcv=use_lcv, use_fc=use_fc):
        return csp.get_solution()
    return None  # No solution found


class SudokuCSP:
    """
    Represents a Sudoku puzzle as a Constraint Satisfaction Problem.
    
    Variables: Each empty cell (r, c) in the 9x9 grid
    Domain: Numbers 1-9 for each variable
    Constraints: Sudoku rules (row, column, and 3x3 box uniqueness)
    """
    
    def __init__(self, grid: List[List[str]]):
        """
        Initialize the CSP from a Sudoku grid.
        
        Args:
            grid: 9x9 grid where '0' represents an empty cell
        """
        self.size = 9
        self.grid = [row[:] for row in grid]  # Deep copy
        
        # Initialize domains for each cell
        # domains[r][c] is a set of possible values for cell (r, c)
        self.domains: List[List[Set[int]]] = [[set() for _ in range(9)] for _ in range(9)]
        
        for r in range(9):
            for c in range(9):
                if grid[r][c] == '0':
                    # Empty cell: calculate possible values
                    self.domains[r][c] = self._get_legal_values(r, c)
                else:
                    # Given cell: domain is the single given value
                    self.domains[r][c] = {int(grid[r][c])}
                    
    def _get_legal_values(self, row: int, col: int) -> Set[int]:
        """
        Get all legal values for a cell based on current assignments.
        
        Args:
            row: Row index (0-8)
            col: Column index (0-8)
            
        Returns:
            Set of integers 1-9 that don't violate Sudoku constraints
        """
        # TODO
        legal = set()
        for value in range(1, 10):
            if self.is_consistent(row, col, value):
                legal.add(value)
        return legal
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Get all cells that are constrained with the given cell.
        
        Neighbors are cells in the same row, column, or 3x3 box.
        
        Args:
            row: Row index (0-8)
            col: Column index (0-8)
            
        Returns:
            List of (row, col) tuples representing neighbor cells
        """
        # TODO
        neighbors = set()
        # Add neighbors from the same row
        for c in range(self.size):
            if c != col:
                neighbors.add((row, c))
        # Add neighbors from the same column
        for r in range(self.size):
            if r != row:
                neighbors.add((r, col))
        # Add neighbors from the same 3x3 box
        box_row_start = (row // 3) * 3
        box_col_start = (col // 3) * 3
        for r in range(box_row_start, box_row_start + 3):
            for c in range(box_col_start, box_col_start + 3):
                if (r, c) != (row, col):
                    neighbors.add((r, c))

        return list(neighbors)
   
    def select_unassigned_variable(self, use_mrv: bool) -> Optional[Tuple[int, int]]:
        """
        Select the next variable (empty cell) to assign.
        
        Args:
            use_mrv: If True, use Minimum Remaining Values heuristic
                    (select variable with fewest legal values)
                    If False, select first unassigned variable
        
        Returns:
            (row, col) tuple of selected cell, or None if all assigned
        """
        # TODO
        best = None

        if not use_mrv:
            for r in range(self.size):
                for c in range(self.size):
                    if self.grid[r][c] == '0':
                        return (r, c)
            return None
        else:
            min_mrv = float('inf')
            for r in range(self.size):
                for c in range(self.size):
                    if self.grid[r][c] == '0':
                        domain = len(self.domains[r][c])
                        if domain < min_mrv:
                            min_mrv = domain
                            best = (r, c)
            return best
    
    def order_domain_values(self, row: int, col: int, use_lcv: bool) -> List[int]:
        """
        Order the values in the domain of a variable.
        
        Args:
            row: Row index of the variable
            col: Column index of the variable
            use_lcv: If True, use Least Constraining Value heuristic
                    (order values by how much they constrain neighbors)
                    If False, use arbitrary ordering
        
        Returns:
            List of values from the domain, ordered appropriately
        """
        # TODO
        domain = self.domains[row][col]

        if not use_lcv:
            return list(domain)
        
        counts = []
        neighbors = self.get_neighbors(row, col)

        for value in domain:
            count = 0
            for r, c in neighbors:
                if value in self.domains[r][c]:
                    count += 1
            counts.append((count, value))

        counts.sort()
        return [value for count, value in counts]
    
    def is_consistent(self, row: int, col: int, value: int) -> bool:
        """
        Check if assigning a value to a cell is consistent with constraints.
        
        Args:
            row: Row index
            col: Column index
            value: Value to assign (1-9)
        
        Returns:
            True if assignment doesn't violate any constraints
        """
        # TODO
        str_value = str(value)
        for c in range(self.size):
            if self.grid[row][c] == str_value and c != col:
                return False
        for r in range(self.size):
            if self.grid[r][col] == str_value and r != row:
                return False
        box_row_start = (row // 3) * 3
        box_col_start = (col // 3) * 3
        for r in range(box_row_start, box_row_start + 3):
            for c in range(box_col_start, box_col_start + 3):
                if self.grid[r][c] == str_value and (r, c) != (row, col):
                    return False
        return True
    
    def forward_check(self, row: int, col: int, value: int) -> Optional[Dict]:
        """
        Perform forward checking after assigning a value to a cell.
        
        Forward checking removes the assigned value from domains of all neighbors.
        If any neighbor's domain becomes empty, the assignment is invalid.
        
        Args:
            row: Row index of assigned variable
            col: Column index of assigned variable
            value: Assigned value
        
        Returns:
            Dictionary mapping (r,c) -> removed_values for rollback,
            or None if forward checking detects inconsistency
        """
        # TODO
        removed = {}
        for r, c in self.get_neighbors(row, col):
            if self.grid[r][c] == '0' and value in self.domains[r][c]:
                self.domains[r][c].remove(value)
                removed.setdefault((r, c), set()).add(value)

                if not self.domains[r][c]:
                    self.restore_domains(removed)
                    return None
        return removed

    
    def restore_domains(self, removed: Dict[Tuple[int, int], Set[int]]):
        """
        Restore domains after backtracking (undo forward checking).
        
        Args:
            removed: Dictionary from forward_check() mapping cells to removed values
        """
        # TODO
        for (r, c), value in removed.items():
            self.domains[r][c].update(value)
    
    def ac3(self) -> bool:
        """
        Apply AC-3 (Arc Consistency 3) algorithm for constraint propagation.
        
        AC-3 enforces arc consistency by iteratively removing values from domains
        that cannot be part of any solution. This is more powerful than forward
        checking as it propagates constraints through the entire network.
        
        Returns:
            True if the CSP is arc-consistent (may have solution),
            False if inconsistency detected (no solution possible)
        """
        # TODO
        queue = []
        for r in range(self.size):
            for c in range(self.size):
                for nr, nc in self.get_neighbors(r, c):
                    queue.append(((r, c), (nr, nc)))

        while queue:
            (xi, xj) = queue.pop(0)

            if self._revise(xi, xj):
                if not self.domains[xi[0]][xi[1]]:
                    return False
                
                for xk in self.get_neighbors(xi[0], xi[1]):
                    if xk != xj:
                        queue.append((xk, xi))

        return True
    
    def _revise(self, xi: Tuple[int, int], xj: Tuple[int, int]) -> bool:
        """
        Revise the domain of Xi to make it consistent with Xj.
        
        Remove values from Xi's domain that have no consistent value in Xj's domain.
        
        Args:
            xi: (row, col) of first variable
            xj: (row, col) of second variable
        
        Returns:
            True if Xi's domain was revised (values removed), False otherwise
        """
        # TODO
        revised = False
        xi_domain = self.domains[xi[0]][xi[1]]
        xj_domain = self.domains[xj[0]][xj[1]]

        if len(xj_domain) == 1:
            y = next(iter(xj_domain))
            if y in xi_domain:
                xi_domain.remove(y)
                revised = True

        return revised

    
    def backtrack(self, use_mrv: bool = False, use_lcv: bool = False, 
                    use_fc: bool = False) -> bool:
        """
        Recursive backtracking search to solve the CSP.
        
        Args:
            use_mrv: Use Minimum Remaining Values heuristic for variable selection
            use_lcv: Use Least Constraining Value heuristic for value ordering
            use_fc: Use Forward Checking for constraint propagation
        
        Returns:
            True if solution found, False otherwise
        """
        # TODO
        var = self.select_unassigned_variable(use_mrv)
        if var is None:
            return True
        
        row, col = var
        values = self.order_domain_values(row, col, use_lcv)

        for value in values:
            if self.is_consistent(row, col, value):
                self.grid[row][col] = str(value)

                removed = None
                if use_fc:
                    removed = self.forward_check(row, col, value)
                    if removed is None:
                        self.grid[row][col] = '0'
                        continue

                if self.backtrack(use_mrv, use_lcv, use_fc):
                    return True
                
                if use_fc and removed is not None:
                    self.restore_domains(removed)

                self.grid[row][col] = '0'

        return False


    
    def get_solution(self) -> List[List[str]]:
        """
        Get the current grid as the solution.
        
        Returns:
            9x9 grid as list of lists of strings
        """
        return [row[:] for row in self.grid]
    
"""
I asked Gemini:
1. How to levergae domains.
It answered:
In the context of this CSP solver, that's the perfect question. The self.domains variable isn't just data; it's the central engine for the entire search.

"Leveraging the domain" means using this "list of possibilities" for each cell to make the search smarter, faster, and more efficient. The entire goal is to prune these domains (remove impossible values) until every cell has only one value left.

Here’s exactly how each component you built leverages the domains:

1. 🧠 Heuristics (Making Smart Guesses)
Heuristics use domains to decide what to do next.

Minimum Remaining Values (MRV): (select_unassigned_variable)

How it's leveraged: This heuristic directly reads the size of each domain. It asks, "Which unassigned cell has the fewest possible values?" (i.e., len(self.domains[r][c])).

The Strategy: It picks the variable with the smallest domain. This is a "fail-fast" strategy. If a cell only has 1 or 2 options, it's a critical bottleneck. It's better to test that cell now. If it's a dead end, we find out immediately instead of after 1,000 recursive calls.

Least Constraining Value (LCV): (order_domain_values)

How it's leveraged: This heuristic looks inside the domains of neighbors. When deciding which value (like 5) to try for cell (r, c), it checks all its neighbors (nr, nc) and asks, "How many of you also have 5 in your domain (value in self.domains[nr][nc])?"

The Strategy: It orders the values to try the one that is least "popular" among its neighbors. By picking the value that least restricts the neighbors' domains, it leaves them with more options, making a solution more likely. It tries to "be a good neighbor."

2. propagation (Enforcing Logic)
Propagation algorithms use domains to actively remove impossible options.

Forward Checking (FC): (forward_check)

How it's leveraged: This is an active update. When you tentatively assign grid[r][c] = 5, Forward Checking immediately goes to all neighbors and removes 5 from their domains (self.domains[nr][nc].remove(5)).

The Strategy: This creates an immediate, local ripple effect. The key leverage is detecting failure early. If removing 5 from a neighbor's domain makes that neighbor's domain empty, we know the assignment was wrong instantly. We don't need to recurse. forward_check returns None, and backtrack tries the next value.

AC-3 (Arc Consistency): (ac3 and _revise)

How it's leveraged: This is the most powerful leverage. It looks at pairs of neighbors (an "arc" like (Xi, Xj)). The _revise function checks: "For every value x in Xi's domain, is there at least one value y in Xj's domain that is consistent (i.e., x != y)?"

The Strategy: If Xi has a value x (say, 5) but its neighbor Xj only has {5} in its domain, AC-3 knows that Xi cannot be 5. It revises Xi's domain and removes 5. This change then propagates, as all of Xi's other neighbors now need to be re-checked against its smaller domain. It's a global "logic bomb" that can solve large parts of the puzzle before the backtrack search even begins.

In short, the domains are the "state" of the puzzle. The heuristics read this state to make guesses, and the propagation algorithms write to this state to enforce logic.

2. How to do the forward_check?
It answered:
Propagation: forward_check
Forward Checking (FC) is a more powerful consistency check. When we assign value to (row, col), we immediately remove value from the domains of all its neighbors.

Logic (forward_check):

Create a dictionary removed_map to store all changes we make, so we can undo them.

Get all neighbors of (row, col).

For each neighbor (nr, nc):

If value is in self.domains[nr][nc]: a. Remove it: self.domains[nr][nc].remove(value). b. Record this change: removed_map.setdefault((nr, nc), set()).add(value). c. Crucial Check: If this removal makes the neighbor's domain empty (len(self.domains[nr][nc]) == 0), this assignment is invalid. Return None to signal failure.

If we get through all neighbors without creating an empty domain, return the removed_map.

Logic (restore_domains):

This function is the "undo" for forward_check.

It takes the removed_map as input.

Iterate through the (cell, values_to_restore) items in the map.

For each cell, add the values_to_restore back into its domain.

Implementation:

Python

def forward_check(self, row: int, col: int, value: int) -> Optional[Dict[Tuple[int, int], Set[int]]]:
    
    Perform forward checking after assigning a value to a cell.

    Removes 'value' from domains of all unassigned neighbors.

    Args:
        row: Row index of assigned variable
        col: Column index of assigned variable
        value: Assigned value

    Returns:
        Dictionary mapping (r,c) -> removed_values for rollback,
        or None if forward checking detects inconsistency
    
    removed_map = {}

    for nr, nc in self.get_neighbors(row, col):
        # Only affect unassigned neighbors
        if self.grid[nr][nc] == '0' and value in self.domains[nr][nc]:
            self.domains[nr][nc].remove(value)

            # Record what we did so we can undo it
            removed_map.setdefault((nr, nc), set()).add(value)

            # If we've emptied a domain, this assignment is invalid
            if not self.domains[nr][nc]:
                # Restore what we've done *so far* before failing
                self.restore_domains(removed_map)
                return None

    return removed_map


3. How does backtrack work?
It answered:
This is the heart of the solver. It's a recursive function that implements the "try, recurse, backtrack" logic.

Logic:

Base Case: Call select_unassigned_variable. If it returns None, it means all cells are assigned. We've found a solution! Return True.

Recursive Step: a. Get the (row, col) of the variable we just selected. b. Get the list of values to try for this cell using order_domain_values. c. Loop through each value in this list: d. Check Consistency: Call self.is_consistent(row, col, value). If it's False, this value conflicts with a pre-existing number, so skip it. (Note: This check is technically redundant if AC-3 or FC is used perfectly, but it's a good safeguard). e. Assign: If consistent, "try" this assignment: self.grid[row][col] = str(value). f. Forward Checking (if use_fc): i. Call removed = self.forward_check(row, col, value). ii. If removed is not None (meaning FC didn't find an empty domain): iii. Recurse: Call self.backtrack(...). If this returns True, a solution was found, so pass True all the way up. iv. Undo FC: If the recursive call failed, we must backtrack. Call self.restore_domains(removed) to put the values back. g. No Forward Checking: i. Recurse: Call self.backtrack(...). If it returns True, pass it up. h. Backtrack: If we're here, it means the value (or the subsequent recursion) failed. We must un-assign: self.grid[row][col] = '0'.

Failure: If the loop finishes without any value leading to a solution, return False.

Implementation:

Python

def backtrack(self, use_mrv: bool = False, use_lcv: bool = False, 
                use_fc: bool = False) -> bool:
    
    Recursive backtracking search to solve the CSP.

    Args:
        use_mrv: Use Minimum Remaining Values heuristic for variable selection
        use_lcv: Use Least Constraining Value heuristic for value ordering
        use_fc: Use Forward Checking for constraint propagation

    Returns:
        True if solution found, False otherwise
    
    # 1. Base Case: Check if all variables are assigned
    var = self.select_unassigned_variable(use_mrv)
    if var is None:
        return True  # Solution found!

    row, col = var

    # 2. Recursive Step: Try values for the selected variable
    values = self.order_domain_values(row, col, use_lcv)

    for value in values:
        # We only need to check consistency if we're *not* using
        # a perfect domain-maintaining algorithm (like FC or AC-3).
        # But, self.domains might not be perfectly pruned if FC isn't on.
        # is_consistent checks against the 'grid', so it's a safe check.
        if self.is_consistent(row, col, value):

            # Assign
            self.grid[row][col] = str(value)

            removed = None
            if use_fc:
                removed = self.forward_check(row, col, value)
                if removed is None:  # FC detected a failure
                    self.grid[row][col] = '0' # Un-assign
                    continue # Try next value

            # Recurse
            if self.backtrack(use_mrv, use_lcv, use_fc):
                return True  # Solution found

            # Backtrack: Un-do Forward Checking
            if use_fc and removed is not None:
                self.restore_domains(removed)

            # Backtrack: Un-assign
            self.grid[row][col] = '0'

    # 3. Failure: No value worked for this variable
    return False
"""