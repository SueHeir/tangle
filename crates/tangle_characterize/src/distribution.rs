//! Compact summaries of scalar distributions.
//!
//! A [`Distribution`] stores a sample's quantile function at evenly spaced
//! probabilities. That is enough to plot the distribution, read any quantile,
//! and compute the first Wasserstein (earth mover's) distance to another
//! distribution, which is how generated and CT-tracked structures are compared.

use std::cmp::Ordering;

/// Summary of a scalar sample: count, moments and quantile function.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Distribution {
    /// Number of finite values summarized.
    pub count: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Population standard deviation.
    pub standard_deviation: f64,
    /// Values at the probabilities `k / (quantiles.len() - 1)`, so the first
    /// entry is the minimum and the last is the maximum. Intermediate
    /// quantiles interpolate linearly between order statistics.
    pub quantiles: Vec<f64>,
}

impl Distribution {
    /// Summarizes the finite values of a sample with `quantile_count` evenly
    /// spaced quantiles (at least two). Returns `None` when no value is finite.
    pub fn from_values(
        values: impl IntoIterator<Item = f64>,
        quantile_count: usize,
    ) -> Option<Self> {
        let mut values: Vec<f64> = values.into_iter().filter(|v| v.is_finite()).collect();
        if values.is_empty() {
            return None;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let quantile_count = quantile_count.max(2);
        let quantiles = (0..quantile_count)
            .map(|k| sorted_quantile(&values, k as f64 / (quantile_count - 1) as f64))
            .collect();
        Some(Self {
            count,
            mean,
            standard_deviation: variance.sqrt(),
            quantiles,
        })
    }

    /// Summarizes a weighted sample. Values with a non-finite value or a
    /// non-positive or non-finite weight are ignored. The mean and standard
    /// deviation are weighted; quantiles interpolate between values placed at
    /// the midpoints of their cumulative weight, rescaled so the smallest
    /// value is the minimum and the largest the maximum. With equal weights
    /// this is exactly [`Distribution::from_values`].
    pub fn from_weighted_values(
        values: impl IntoIterator<Item = (f64, f64)>,
        quantile_count: usize,
    ) -> Option<Self> {
        let mut pairs: Vec<(f64, f64)> = values
            .into_iter()
            .filter(|(v, w)| v.is_finite() && w.is_finite() && *w > 0.0)
            .collect();
        if pairs.is_empty() {
            return None;
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        let count = pairs.len();
        let total: f64 = pairs.iter().map(|(_, w)| w).sum();
        let mean = pairs.iter().map(|(v, w)| v * w).sum::<f64>() / total;
        let variance = pairs
            .iter()
            .map(|(v, w)| w * (v - mean).powi(2))
            .sum::<f64>()
            / total;
        let quantile_count = quantile_count.max(2);
        let quantiles = if count == 1 {
            vec![pairs[0].0; quantile_count]
        } else {
            let first = 0.5 * pairs[0].1;
            let span = total - first - 0.5 * pairs[count - 1].1;
            let mut positions = Vec::with_capacity(count);
            let mut before = 0.0;
            for (_, weight) in &pairs {
                positions.push((before + 0.5 * weight - first) / span);
                before += weight;
            }
            (0..quantile_count)
                .map(|k| {
                    let p = k as f64 / (quantile_count - 1) as f64;
                    let upper = positions.partition_point(|t| *t < p).clamp(1, count - 1);
                    let (t0, t1) = (positions[upper - 1], positions[upper]);
                    let fraction = if t1 > t0 {
                        ((p - t0) / (t1 - t0)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    pairs[upper - 1].0 + fraction * (pairs[upper].0 - pairs[upper - 1].0)
                })
                .collect()
        };
        Some(Self {
            count,
            mean,
            standard_deviation: variance.sqrt(),
            quantiles,
        })
    }

    /// Value at probability `p` (clamped to `[0, 1]`), interpolated between the
    /// stored quantiles.
    pub fn quantile(&self, p: f64) -> f64 {
        sorted_quantile(&self.quantiles, p)
    }

    /// Median value.
    pub fn median(&self) -> f64 {
        self.quantile(0.5)
    }

    /// First Wasserstein distance, `∫₀¹ |F⁻¹(p) − G⁻¹(p)| dp`, evaluated with
    /// the trapezoid rule on the finer of the two quantile grids. It has the
    /// units of the values: shifting every value by `d` gives distance `|d|`.
    pub fn wasserstein_distance(&self, other: &Self) -> f64 {
        let points = self.quantiles.len().max(other.quantiles.len()).max(2);
        let step = 1.0 / (points - 1) as f64;
        let difference = |k: usize| {
            let p = k as f64 * step;
            (self.quantile(p) - other.quantile(p)).abs()
        };
        (0..points - 1)
            .map(|k| 0.5 * (difference(k) + difference(k + 1)) * step)
            .sum()
    }
}

/// Linear-interpolation quantile of ascending `values` at probability `p`.
fn sorted_quantile(values: &[f64], p: f64) -> f64 {
    match values.len() {
        0 => f64::NAN,
        1 => values[0],
        len => {
            let position = p.clamp(0.0, 1.0) * (len - 1) as f64;
            let lower = (position.floor() as usize).min(len - 2);
            let fraction = position - lower as f64;
            values[lower] + fraction * (values[lower + 1] - values[lower])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_of_an_evenly_spaced_sample_are_exact() {
        let distribution = Distribution::from_values((0..=100).rev().map(f64::from), 11).unwrap();
        assert_eq!(distribution.count, 101);
        assert!((distribution.mean - 50.0).abs() < 1e-12);
        let expected: Vec<f64> = (0..=10).map(|k| 10.0 * k as f64).collect();
        for (value, expected) in distribution.quantiles.iter().zip(&expected) {
            assert!((value - expected).abs() < 1e-12);
        }
        assert!((distribution.median() - 50.0).abs() < 1e-12);
        assert!((distribution.quantile(0.25) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn equal_weights_reproduce_the_unweighted_summary() {
        let values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6];
        let plain = Distribution::from_values(values, 11).unwrap();
        let weighted =
            Distribution::from_weighted_values(values.iter().map(|v| (*v, 2.0)), 11).unwrap();
        assert_eq!(plain.count, weighted.count);
        assert!((plain.mean - weighted.mean).abs() < 1e-12);
        assert!((plain.standard_deviation - weighted.standard_deviation).abs() < 1e-12);
        for (a, b) in plain.quantiles.iter().zip(&weighted.quantiles) {
            assert!((a - b).abs() < 1e-12, "{a} {b}");
        }
    }

    #[test]
    fn weights_shift_the_mean_and_zero_weights_are_ignored() {
        let weighted =
            Distribution::from_weighted_values([(0.0, 1.0), (1.0, 3.0), (5.0, 0.0)], 5).unwrap();
        assert_eq!(weighted.count, 2);
        assert!((weighted.mean - 0.75).abs() < 1e-12);
        assert_eq!(weighted.quantiles[0], 0.0);
        assert_eq!(weighted.quantiles[4], 1.0);
        assert!(Distribution::from_weighted_values([(1.0, 0.0)], 5).is_none());
    }

    #[test]
    fn non_finite_values_are_ignored_and_empty_samples_give_none() {
        assert!(Distribution::from_values([f64::NAN, f64::INFINITY], 5).is_none());
        let distribution = Distribution::from_values([2.0, f64::NAN, 2.0], 5).unwrap();
        assert_eq!(distribution.count, 2);
        assert_eq!(distribution.standard_deviation, 0.0);
        assert!(distribution.quantiles.iter().all(|q| *q == 2.0));
    }

    #[test]
    fn wasserstein_distance_of_a_shift_is_the_shift() {
        // Evenly spaced values, so a coarser quantile grid interpolates exactly.
        let first = Distribution::from_values((0..1000).map(|i| i as f64 / 999.0), 101).unwrap();
        let shifted =
            Distribution::from_values((0..1000).map(|i| i as f64 / 999.0 + 0.75), 51).unwrap();
        assert!((first.wasserstein_distance(&shifted) - 0.75).abs() < 1e-9);
        assert!((shifted.wasserstein_distance(&first) - 0.75).abs() < 1e-9);
        assert!(first.wasserstein_distance(&first).abs() < 1e-12);
    }

    #[test]
    fn wasserstein_distance_between_uniform_samples_matches_theory() {
        // U(0, 1) against U(0, 2): ∫ |p − 2p| dp = 1/2.
        let narrow = Distribution::from_values((0..=10_000).map(|i| i as f64 / 1e4), 201).unwrap();
        let wide = Distribution::from_values((0..=10_000).map(|i| i as f64 / 5e3), 201).unwrap();
        assert!((narrow.wasserstein_distance(&wide) - 0.5).abs() < 1e-6);
    }
}
