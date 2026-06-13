import math
import unittest
from unittest.mock import mock_open, patch
from expert import (
    add_noise_and_compare,
    apply_pca_to_dataset,
    auto_select_k,
    calculate_geometry_preservation,
    column_mean,
    handle_missing_values,
)

class TestPcaUtils(unittest.TestCase):

    # --- 1. Тесты для auto_select_k ---
    @patch("expert.explained_variance_ratio")
    def test_auto_select_k_normal(self, mock_evr):
        # Имитируем рост объясненной дисперсии с увеличением k
        mock_evr.side_effect = lambda ev, k: 0.5 if k == 1 else 0.96
        eigenvalues = [10.0, 5.0, 1.0]

        result = auto_select_k(eigenvalues, threshold=0.95)
        self.assertEqual(result, 2)

    @patch("expert.explained_variance_ratio")
    def test_auto_select_k_fallback(self, mock_evr):
        # Дисперсия никогда не превышает порог
        mock_evr.return_value = 0.80
        eigenvalues = [10.0, 5.0]

        result = auto_select_k(eigenvalues, threshold=0.95)
        self.assertEqual(result, 2)

    # --- 2. Тесты для column_mean ---
    def test_column_mean_with_nan(self):
        X = [[1.0, 10.0], [math.nan, 20.0], [3.0, 30.0]]
        self.assertAlmostEqual(column_mean(X, 0), 2.0)

    def test_column_mean_all_numbers(self):
        X = [[1.0], [2.0], [3.0]]
        self.assertAlmostEqual(column_mean(X, 0), 2.0)

    # --- 3. Тесты для handle_missing_values ---
    @patch("expert.column_mean")
    def test_handle_missing_values(self, mock_mean):
        mock_mean.return_value = 5.0
        X = [[1.0, math.nan], [math.nan, 2.0]]

        expected = [[1.0, 5.0], [5.0, 2.0]]
        result = handle_missing_values(X)

        self.assertEqual(result, expected)

    # --- 4. Тесты для add_noise_and_compare ---
    @patch("expert.random.gauss")
    @patch("expert.pca")
    @patch("expert.handle_missing_values")
    @patch("expert.reconstruction_error")
    def test_add_noise_and_compare(self, mock_recon, mock_handle, mock_pca, mock_gauss):
        mock_gauss.return_value = 0.1
        mock_pca.return_value = [[1.0]]
        mock_handle.return_value = [[1.0]]
        mock_recon.return_value = 0.05

        X = [[1.0, 2.0], [3.0, 4.0]]
        error = add_noise_and_compare(X, noise_level=0.1)

        self.assertEqual(error, 0.05)
        self.assertEqual(mock_pca.call_count, 2)
        
    # --- 5. Тесты для apply_pca_to_dataset ---
    @patch("expert.pca")
    @patch("expert.calculate_geometry_preservation")
    def test_apply_pca_to_dataset(self, mock_geo, mock_pca):
        # Имитируем структуру CSV файла (Парсинг: заголовок, затем строки данных)
        csv_data = "f1,f2,target\n1.0,2.0,0.0\n3.0,4.0,1.0"

        mock_pca.return_value = [[0.5], [1.5]]
        mock_geo.return_value = 0.99

        # Мокаем открытие файла через build-in open
        with patch("builtins.open", mock_open(read_data=csv_data)):
            projection, quality = apply_pca_to_dataset("fake_path.csv", k=1)

            self.assertEqual(projection, [[0.5], [1.5]])
            self.assertEqual(quality, 0.99)
            mock_pca.assert_called_once_with([[1.0, 2.0], [3.0, 4.0]], 1)

    # --- 6. Тесты для calculate_geometry_preservation ---
    def test_calculate_geometry_preservation_perfect(self):
        X_old = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        X_new = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

        res = calculate_geometry_preservation(X_old, X_new)
        self.assertAlmostEqual(res, 1.0)

    def test_calculate_geometry_preservation_zero_elements(self):
        # Если в матрице меньше 2 строк, попарных расстояний нет (m=0)
        X_old = [[1.0, 2.0]]
        X_new = [[0.5]]

        res = calculate_geometry_preservation(X_old, X_new)
        self.assertEqual(res, 0.0)


if __name__ == "__main__":
    unittest.main()
