"""Contract checks: monthly ranks, temporal leakage, and label independence."""
import unittest
import numpy as np
import pandas as pd
from src.investor_rf import load_track_features, select_investor_features, rank_transform_features, train_one_investor_gpu


class InvestorContractTests(unittest.TestCase):
    def test_manifest_selection(self):
        for track, count in [("accounting", 38), ("market", 26)]:
            features = load_track_features(track)
            chosen = select_investor_features(features, 1002)
            self.assertEqual(len(chosen), count)
            self.assertEqual(chosen, select_investor_features(features, 1002))
            self.assertTrue(set(chosen) <= set(features))

    def test_monthly_ranks_ties_nonfinite_and_singleton(self):
        df = pd.DataFrame({"eom": ["2010-01-31"] * 5 + ["2010-02-28"] * 2,
                           "x": [1, 1, 3, np.inf, np.nan, 9, np.nan]})
        ranked = rank_transform_features(df, ["x"])
        np.testing.assert_allclose(ranked.x, [-.5, -.5, 1, 0, 0, 0, 0])
        self.assertEqual(str(ranked.x.dtype), "float32")

    def test_future_labels_not_used_and_overlap_rejected(self):
        rng = np.random.default_rng(27)
        features = load_track_features("market")
        df = pd.DataFrame(rng.uniform(-1, 1, size=(200, len(features))), columns=features)
        df["id"] = np.tile(np.arange(100), 2)
        df["eom"] = [pd.Timestamp("2010-01-31")] * 100 + [pd.Timestamp("2010-02-28")] * 100
        df["ret_exc_lead1m"] = rng.normal(size=200)
        train, test = df.iloc[:100].copy(), df.iloc[100:].copy()
        kwargs = dict(track="market", device="cpu", n_estimators=10)
        first = train_one_investor_gpu(train, test, **kwargs)
        second = train_one_investor_gpu(train, test.drop(columns="ret_exc_lead1m"), **kwargs)
        np.testing.assert_array_equal(first["predictions"].prediction, second["predictions"].prediction)
        self.assertEqual(len(first["model"].get_booster().get_dump()), 10)
        self.assertEqual(first["predictions"].index.tolist(), test.index.tolist())
        with self.assertRaisesRegex(ValueError, "precede"):
            train_one_investor_gpu(train, train, **kwargs)
        with self.assertRaisesRegex(ValueError, "Missing prediction columns"):
            train_one_investor_gpu(train, test.drop(columns=features[0]), **kwargs)


if __name__ == "__main__":
    unittest.main()
