import unittest
from easy import gauss_solver

class TestGaussSolver(unittest.TestCase):

    def test_single_solution_homogeneous(self):
        """Единственное тривиальное решение однородной СЛАУ (все b = 0)"""
        A = [[1, 2], [3, 4]]
        b = [0, 0]
        expected = [[0, 0]]
        self.assertEqual(gauss_solver(A, b), expected)

    def test_single_solution_inhomogeneous(self):
        """Единственное решение неоднородной СЛАУ"""
        A = [[1, 2], [3, 4]]
        b = [5, 11]
        expected = [[1.0, 2.0]]

        result = gauss_solver(A, b)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0][0], expected[0][0], places=7)
        self.assertAlmostEqual(result[0][1], expected[0][1], places=7)

    def test_inf_solutions_homogeneous(self):
        """Бесконечное количество решений (однородная система)"""
        A = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        b = [0, 0, 0]

        result = gauss_solver(A, b)

        self.assertTrue(len(result) > 1)
        for x in result:
            for row in A:
                ans = sum(row[i] * x[i] for i in range(len(x)))
                self.assertAlmostEqual(ans, 0.0, places=7)

    def test_inf_solutions_inhomogeneous(self):
        """Бесконечное количество решений (неоднородная система)"""
        A = [[1, 1], [2, 2]]
        b = [2, 4]

        result = gauss_solver(A, b)

        self.assertTrue(len(result) >= 1)

        x_particular = result[0]
        self.assertAlmostEqual(x_particular[0] + x_particular[1], 2.0, places=7)

    def test_inconsistent_system_simple(self):
        """Система несовместна (нет решений)"""
        A = [[1, 1], [1, 1]]
        b = [1, 2]

        with self.assertRaises(ValueError):
            gauss_solver(A, b)

    def test_inconsistent_system_zeros_row(self):
        """Система несовместна (нулевая строка в матрице А равна ненулевому значению в b)"""
        A = [[1, 2, 3], [0, 0, 0], [4, 5, 6]]
        b = [1, 5, 2]

        with self.assertRaises(ValueError):
            gauss_solver(A, b)


if __name__ == "__main__":
    unittest.main()
