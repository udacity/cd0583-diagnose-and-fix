import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import train


class DriftRegressionTests(unittest.TestCase):
    @staticmethod
    def sample(rows=30):
        return pd.DataFrame({
            feature: np.linspace(0, 1, rows)
            for feature in train.NUMERICAL_FEATURES
        })

    def test_dataset_share_uses_result_instead_of_configuration(self):
        reference = self.sample()
        for changed_columns, expected_ratio, expected_flag in [(0, 0, False), (2, .4, False), (3, .6, True)]:
            with self.subTest(changed_columns=changed_columns):
                current = reference.copy()
                current.iloc[:, :changed_columns] += 10
                self.assertAlmostEqual(
                    train.detect_dataset_drift(reference, current, return_ratio=True),
                    expected_ratio,
                )
                self.assertEqual(train.detect_dataset_drift(reference, current), expected_flag)

    def test_feature_flags_for_p_value_and_distance_methods(self):
        # Evidently automatically uses KS for small reference sets and Wasserstein
        # distance for large numerical reference sets; their score directions differ.
        for rows in (30, 1200):
            with self.subTest(rows=rows):
                reference = self.sample(rows)
                current = reference.copy()
                current['temp'] += 10
                flags = dict(train.detect_features_drift(reference, current))
                self.assertEqual(flags, {name: name == 'temp' for name in train.NUMERICAL_FEATURES})
                scores = dict(train.detect_features_drift(reference, current, return_scores=True))
                if rows == 30:
                    self.assertLess(scores['temp'], .05)
                    self.assertGreater(scores['hum'], .05)
                else:
                    self.assertGreater(scores['temp'], .1)
                    self.assertLess(scores['hum'], .1)

    def test_missing_drift_test_is_not_treated_as_no_drift(self):
        report = {'metrics': [{'id': 'example', 'value': .001}], 'tests': []}
        with patch.object(train, 'run_report', return_value=report):
            with self.assertRaisesRegex(ValueError, "Could not determine drift"):
                train.detect_features_drift(self.sample(), self.sample())


if __name__ == '__main__':
    unittest.main()
