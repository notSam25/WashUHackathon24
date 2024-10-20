import unittest
import pandas as pd
from model.src.predict import (
    predict,
    predict_from_dataframe,
)  # Adjust the import according to your module structure


class TestPredictFunction(unittest.TestCase):

    def setUp(self):
        # Load the input CSV directly
        self.input_data = pd.read_csv("data/test/test.csv")

    def test_predict_output_shape(self):
        # Call the predict function with the loaded input data
        esi_scores, probabilities = predict_from_dataframe(self.input_data)

        # Check if the output shape matches the input shape
        self.assertEqual(len(esi_scores), len(self.input_data))

    def test_predict_values(self):
        # Extract expected outputs from the input data
        expected_scores = self.input_data["expected_score_esi"].tolist()
        # Call the predict function
        esi_scores, probabilities = predict_from_dataframe(self.input_data)

        # Check if the predicted scores match the expected scores
        for expected, actual in zip(expected_scores, esi_scores):
            self.assertEqual(expected, actual)

    def tearDown(self):
        # No cleanup needed if using a pre-existing CSV
        pass


if __name__ == "__main__":
    unittest.main()
