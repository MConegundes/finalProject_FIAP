import unittest
from unittest.mock import MagicMock, patch
from src import preprocessing, data_loader, inferencia, train


class TestModelStorage(unittest.TestCase):
	def setUp(self):
		mock = MagicMock()
		mock.open.__enter__.return_value = True
		self.model_path = mock
		self.model = 'dummy_model'
		self.symbol = 'dummy_symbol'

	def test_save_and_load(self):
		self.assertTrue(preprocessing.save_scaler(self.model, self.symbol))

		self.assertTrue(preprocessing.load_scaler(self.symbol))


if __name__ == '__main__':
	unittest.main()
