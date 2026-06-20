from typing import Optional, Tuple
import numpy as np
import pandas as pd


class Binning:
    @staticmethod
    def equal_width(
        data: np.ndarray,
        n_bins: int = 10,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = np.asarray(data).flatten()
        if min_value is None:
            min_value = np.min(data)
        if max_value is None:
            max_value = np.max(data)

        if min_value == max_value:
            edges = np.array([min_value, max_value + 1e-10])
            bin_indices = np.zeros(len(data), dtype=int)
        else:
            edges = np.linspace(min_value, max_value, n_bins + 1)
            bin_indices = np.digitize(data, edges, right=False)
            bin_indices = np.clip(bin_indices, 1, n_bins) - 1

        return bin_indices, edges

    @staticmethod
    def equal_frequency(
        data: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = np.asarray(data).flatten()
        sorted_data = np.sort(data)
        n = len(sorted_data)

        if n <= n_bins:
            n_bins = max(1, n - 1)

        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(sorted_data, percentiles)
        edges = np.unique(edges)

        if len(edges) < 2:
            edges = np.array([edges[0], edges[0] + 1e-10])

        bin_indices = np.digitize(data, edges, right=False)
        bin_indices = np.clip(bin_indices, 1, len(edges) - 1) - 1

        return bin_indices, edges

    @staticmethod
    def optimal_binning(
        data: np.ndarray,
        target: Optional[np.ndarray] = None,
        n_bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if target is None:
            return Binning.equal_frequency(data, n_bins)

        data = np.asarray(data).flatten()
        target = np.asarray(target).flatten()

        df = pd.DataFrame({"feature": data, "target": target})
        df = df.sort_values("feature").reset_index(drop=True)

        unique_vals = df["feature"].unique()
        if len(unique_vals) <= n_bins:
            edges = np.concatenate([[unique_vals[0] - 1e-10], unique_vals[1:], [unique_vals[-1] + 1e-10]])
        else:
            edges = Binning._tree_binning(df["feature"].values, df["target"].values, n_bins)

        bin_indices = np.digitize(data, edges, right=False)
        bin_indices = np.clip(bin_indices, 1, len(edges) - 1) - 1

        return bin_indices, edges

    @staticmethod
    def _tree_binning(
        feature: np.ndarray,
        target: np.ndarray,
        n_bins: int,
    ) -> np.ndarray:
        sorted_idx = np.argsort(feature)
        feature_sorted = feature[sorted_idx]
        target_sorted = target[sorted_idx]

        split_points = []

        def find_split(start: int, end: int) -> Tuple[int, float]:
            best_iv = -1
            best_split = start + 1

            for split in range(start + 1, end):
                left = target_sorted[start:split]
                right = target_sorted[split:end]

                if len(left) == 0 or len(right) == 0:
                    continue

                left_pos = np.sum(left)
                left_neg = len(left) - left_pos
                right_pos = np.sum(right)
                right_neg = len(right) - right_pos

                total_pos = left_pos + right_pos
                total_neg = left_neg + right_neg

                if total_pos == 0 or total_neg == 0:
                    continue

                left_pos_rate = left_pos / total_pos if total_pos > 0 else 0
                left_neg_rate = left_neg / total_neg if total_neg > 0 else 0
                right_pos_rate = right_pos / total_pos if total_pos > 0 else 0
                right_neg_rate = right_neg / total_neg if total_neg > 0 else 0

                eps = 1e-10
                iv_left = (left_pos_rate - left_neg_rate) * np.log(
                    (left_pos_rate + eps) / (left_neg_rate + eps)
                )
                iv_right = (right_pos_rate - right_neg_rate) * np.log(
                    (right_pos_rate + eps) / (right_neg_rate + eps)
                )
                iv = iv_left + iv_right

                if iv > best_iv:
                    best_iv = iv
                    best_split = split

            return best_split, feature_sorted[best_split]

        def recursive_split(start: int, end: int, depth: int):
            if depth <= 0 or end - start < 2:
                return

            split_idx, split_val = find_split(start, end)
            split_points.append(split_val)
            recursive_split(start, split_idx, depth - 1)
            recursive_split(split_idx, end, depth - 1)

        depth = int(np.log2(n_bins)) + 1
        recursive_split(0, len(feature_sorted), depth)

        split_points = sorted(set(split_points))
        edges = np.concatenate(
            [[feature_sorted[0] - 1e-10], split_points, [feature_sorted[-1] + 1e-10]]
        )

        return edges

    @staticmethod
    def get_bin_ratios(
        bin_indices: np.ndarray,
        n_bins: int,
    ) -> np.ndarray:
        counts = np.bincount(bin_indices, minlength=n_bins)
        total = len(bin_indices)
        if total == 0:
            return np.zeros(n_bins)
        return counts / total
