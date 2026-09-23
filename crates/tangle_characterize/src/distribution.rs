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
    fn non_finite_values_are_ignored_and_empty_samples_give_none() {
        assert!(Distribution::from_values([f64::NAN, f64::INFINITY], 5).is_none());
        let distribution = Distribution::from_values([2.0, f64::NAN, 2.0], 5).unwrap();
        assert_eq!(distribution.count, 2);
        assert_eq!(distribution.standard_deviation, 0.0);
        assert!(distribution.quantiles.iter().all(|q| *q == 2.0));
    }

    #[test]
    fn wasserstein_distance_of_a_shift_is_the_shift() {
        let first = Distribution::from_values((0..1000).map(|i| (i as f64).sqrt()), 101).unwrap();
        let shifted =
            Distribution::from_values((0..1000).map(|i| (i as f64).sqrt() + 0.75), 51).unwrap();
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
