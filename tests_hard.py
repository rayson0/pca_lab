import unittest
from unittest.mock import patch
from matplotlib.figure import Figure
from hard import pca, plot_pca_projection, reconstruction_error


class TestPCAModule(unittest.TestCase):

    @patch("hard.center_data")
    @patch("hard.covariance_matrix")
    @patch("hard.find_eigenvalues")
    @patch("hard.find_eigenvectors")
    @patch("hard.explained_variance_ratio")
    def test_pca_correctness(
        self, mock_ratio, mock_vectors, mock_values, mock_cov, mock_center
    ):
        """Проверка работы функции pca: логика перемножения, срезы и возвращаемые значения"""
        X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        k = 2

        # Имитируем, что центрированные данные совпали с X (для простоты расчета)
        mock_center.return_value = X
        mock_cov.return_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        mock_values.return_value = [10.0, 5.0, 1.0]

        mock_vectors.return_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        mock_ratio.return_value = 0.9375

        X_proj, share = pca(X, k)

        mock_center.assert_called_once_with(X)
        mock_cov.assert_called_once_with(X)

        mock_values.assert_called_once_with(mock_cov.return_value)
        mock_vectors.assert_called_once_with(mock_cov.return_value, [10.0, 5.0])
        mock_ratio.assert_called_once_with([10.0, 5.0], k)

        expected_proj = [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]

        self.assertEqual(X_proj, expected_proj)
        self.assertEqual(share, 0.9375)
        self.assertEqual(len(X_proj), 3, "Количество объектов не должно измениться")
        self.assertEqual(len(X_proj[0]), k, "Количество признаков должно быть равно k")

    def test_plot_pca_projection(self):
        """Проверка структуры и метаданных возвращаемого графика Matplotlib"""
        X_proj = [[1.0, 2.0], [3.0, 4.0]]

        fig = plot_pca_projection(X_proj)

        self.assertIsInstance(fig, Figure)

        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), "Первая главная компонента")
        self.assertEqual(ax.get_ylabel(), "Вторая главная компонента")
        self.assertEqual(
            ax.get_title(), "Проекция данных на первые две главные компоненты"
        )

    def test_reconstruction_error_perfect(self):
        """Проверка MSE при идеальном совпадении исходных и восстановленных данных"""
        X_orig = [[1.0, 2.0], [3.0, 4.0]]
        X_recon = [[1.0, 2.0], [3.0, 4.0]]

        error = reconstruction_error(X_orig, X_recon)
        self.assertAlmostEqual(error, 0.0, places=7)

    def test_reconstruction_error_calculation(self):
        """Проверка математического расчета формулы MSE"""
        X_orig = [[1.0, 2.0], [3.0, 4.0]]
        X_recon = [[1.5, 2.0], [2.0, 4.2]]

        # (1.0 - 1.5)^2 = 0.25
        # (2.0 - 2.0)^2 = 0.0
        # (3.0 - 2.0)^2 = 1.0
        # (4.0 - 4.2)^2 = 0.04
        # Сумма = 1.29. Всего элементов = 4. MSE = 1.29 / 4 = 0.3225
        expected_mse = 0.3225

        error = reconstruction_error(X_orig, X_recon)
        self.assertAlmostEqual(error, expected_mse, places=7)


if __name__ == "__main__":
    unittest.main()
