import unittest
from normal import find_eigenvalues, count_sign_changes, find_all_eigenvalues, explained_variance_ratio


class TestEigenMethods(unittest.TestCase):

    def test_find_eigenvalues_empty_and_scalar(self):
        """Проверка пустой матрицы и матрицы 1х1"""
        # Пустая матрица
        self.assertEqual(find_eigenvalues([]), [])

        # Матрица 1х1
        self.assertEqual(find_eigenvalues([[5.0]]), [5.0])

    def test_find_eigenvalues_diagonal(self):
        """Проверка диагональной матрицы 2х2"""
        C = [[2.0, 0.0], [0.0, 3.0]]
        res = find_eigenvalues(C)
        self.assertEqual(len(res), 2)
        self.assertAlmostEqual(res[0], 3.0, places=5)
        self.assertAlmostEqual(res[1], 2.0, places=5)

    def test_find_eigenvalues_symmetric_2x2(self):
        """Проверка симметричной матрицы 2х2 с известными аналитическими значениями"""
        C = [[2.0, 1.0], [1.0, 2.0]]
        res = find_eigenvalues(C)
        self.assertEqual(len(res), 2)
        self.assertAlmostEqual(res[0], 3.0, places=5)
        self.assertAlmostEqual(res[1], 1.0, places=5)

    def test_count_sign_changes_basic(self):
        """Проверка корректности подсчета смен знаков в последовательности Штурма"""
        diag = [2.0, 2.0]
        subdiag = [1.0]

        changes_low = count_sign_changes(diag, subdiag, 0.0)
        self.assertEqual(changes_low, 0)

        changes_mid = count_sign_changes(diag, subdiag, 2.0)
        self.assertEqual(changes_mid, 1)

        changes_high = count_sign_changes(diag, subdiag, 4.0)
        self.assertEqual(changes_high, 2)

    def test_find_all_eigenvalues_by_bisect(self):
        """Тестирование функции поиска значений по трехдиагональной форме"""
        diag = [4.0, 5.0]
        subdiag = [0.0]
        res = find_all_eigenvalues(diag, subdiag, tol=1e-6)
        self.assertAlmostEqual(res[0], 5.0, places=5)
        self.assertAlmostEqual(res[1], 4.0, places=5)

    def test_explained_variance_ratio(self):
        """Тестирование расчета доли объясненной дисперсии"""
        eigenvalues = [10.0, 5.0, 3.0, 2.0]

        ratio_k0 = explained_variance_ratio(eigenvalues, k=0)
        self.assertAlmostEqual(ratio_k0, 0.5)

        ratio_k1 = explained_variance_ratio(eigenvalues, k=1)
        self.assertAlmostEqual(ratio_k1, 0.75)

        ratio_all = explained_variance_ratio(eigenvalues, k=3)
        self.assertAlmostEqual(ratio_all, 1.0)


if __name__ == "__main__":
    unittest.main()
