from scipy.optimize import linprog
from pulp import *
import numpy as np

# scipy.optimize.linprog can't solve integer variable problems
# In our sudoku formulation the variables are either 0 or 1

class sudoku_linprog():
    def __init__(self):
        self.n = 9
        self.m = int(np.sqrt(self.n))       # number of boxes across or down
        self.solution = None

    def convert_to_3d(self, array):
        """
        :param array: standard sudoku format (2d matrix) with initial constraints filled in.
                            Undetermined cells will be filled in with a zero
        :return: array3d: 3d cube array of zeros with a 1 in the depth index representing the value
                            for the (x,y) coordinate
        """

        # incoming array is square (e.g. 4x4 or 9x9)
        self.n, _ = array.shape
        r, c = np.nonzero(array)
        d = array[r, c] - 1
        array3d = np.zeros((self.n, self.n, self.n))
        array3d[r, c, d] = 1

        return array3d

    def convert_to_2d(self, array):
        """
        :param array: 3d cube array with 1 in the depth index representing the value for the (x,y) coordinate
                        (This is the solved puzzle, meaning that each (x,y) coordinate should have a value in the depth coord)
        :return: array2d: 2-dimensional representation of sudoku solution
        """
        n, _, _ = array.shape
        _, _, depth = np.nonzero(array)
        if depth.size == 81:
            array2d = depth.reshape((n, n))
        else:
            array2d = None

        return array2d

    def solve(self, array):
        """
        :param array: incomplete sudoku puzzle in 3d format
        :return: completed sudoku puzzle in 2d format
        """

        # Set-up binary linear programming problem
        model = LpProblem(name='sudoku-solver', sense=LpMaximize)
        nums = range(1, self.n + 1)
        var = LpVariable.matrix('v', (nums, nums, nums), cat='Binary')
        arrayV = np.array(var)

        # objective function
        model += 0      # we just want any correct solution

        # linear programming constraints
        # (1) row constraints
        row_constraints = arrayV.sum(0).reshape(-1)
        for i in range(len(row_constraints)):
            model += row_constraints[i] == 1

        # (2) column constraints
        col_constraints = arrayV.sum(1).reshape(-1)
        for j in range(len(col_constraints)):
            model += col_constraints[j] == 1

        # (3) depth constraints
        depth_constraints = arrayV.sum(2).reshape(-1)
        for k in range(len(depth_constraints)):
            model += depth_constraints[k] == 1

        # (4) box constraint
        for i in range(self.m):
            for j in range(self.m):
                for k in range(self.n):
                    box = [(l, m) for l in range(3*i, 3*(i+1)) for m in range(3*j, 3*(j+1))]
                    model += lpSum(var[i][j][k] for (i, j) in box) == 1

        # (5) initial constraints
        array_3d = self.convert_to_3d(array)
        rows, cols, depth = np.nonzero(array_3d)
        for i in range(rows.shape[0]):
            model += var[rows[i]][cols[i]][depth[i]] == 1

        # solve linear programming problem
        model.solve(PULP_CBC_CMD(msg=0))

        # Convert variable solutions to sudoku puzzle
        myarray = np.array([model.variables()[i].varValue for i in
                            range(1, len(model.variables()))]).reshape(self.n, self.n, self.n)

        solu = self.convert_to_2d(myarray)

        if solu is not None:
            self.solution = solu + 1


if __name__ == "__main__":
    solved_puzzle = np.array([[1, 2, 7, 6, 9, 5, 8, 3, 4], [9, 3, 8, 2, 7, 4, 5, 6, 1],
                   [5, 6, 4, 3, 8, 1, 9, 2, 7], [7, 1, 9, 8, 4, 2, 3, 5, 6],
                   [3, 4, 5, 9, 6, 7, 1, 8, 2], [2, 8, 6, 5, 1, 3, 7, 4, 9],
                   [4, 9, 3, 1, 5, 6, 2, 7, 8], [8, 7, 2, 4, 3, 9, 6, 1, 5],
                   [6, 5, 1, 7, 2, 8, 4, 9, 3]])

    initial_puzzle4 = np.array([[1, 5, 6, 3, 0, 4, 2, 0, 0],
                                [9, 0, 0, 1, 8, 0, 0, 7, 0],
                                [8, 0, 0, 0, 6, 0, 1, 4, 5],
                                [0, 8, 3, 0, 9, 6, 5, 0, 7],
                                [0, 0, 4, 0, 0, 0, 0, 0, 0],
                                [0, 1, 9, 7, 0, 3, 8, 0, 4],
                                [4, 0, 0, 5, 0, 1, 7, 6, 2],
                                [7, 0, 0, 0, 0, 0, 0, 0, 0],
                                [3, 0, 0, 0, 0, 7, 0, 0, 9]])

    # randomize initial constraints for new puzzle using solved puzzle
    percent = 0.35     # percentage of missing entries for a randomly generated new puzzle
    numbers = np.tile(np.arange(9)[:, None], 9)
    rows = numbers.reshape(-1)
    cols = numbers.T.reshape(-1)
    inds = [(row, col) for row, col in zip(rows, cols)]
    inds_remove = np.random.permutation(inds)[0:round(percent * len(inds))]
    new_puzzle = np.copy(solved_puzzle)
    new_puzzle[inds_remove[:, 0], inds_remove[:, 1]] = 0

    # solve new puzzle with random initial constraints
    solver = sudoku_linprog()
    solver.solve(initial_puzzle4)
    print(solver.solution)
