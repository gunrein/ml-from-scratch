pub mod dataset;

use std::fmt::{Debug, Display, Formatter};

/// Compute a linear model from a list of data
pub fn compute_model(observations: Vec<Observation>) -> LinearModel {
    // Convenience function for squaring a number
    let square = |number| f32::powf(number, 2.0);

    let count = observations.len() as f32;

    // Compute the sum and average of the x values of the observations
    let sum_of_x = observations.iter().map(|sample| sample.x).sum::<f32>();
    let average_of_x = sum_of_x / count;

    // Compute the sum and average of the y values of the observations
    let sum_of_y = observations.iter().map(|sample| sample.y).sum::<f32>();
    let average_of_y = sum_of_y / count;

    // Calculate the Ordinary Least Squares linear regression for the training data
    // https://en.wikipedia.org/wiki/Ordinary_least_squares
    let numerator: f32 = observations
        .iter()
        .map(|observation| (observation.x - average_of_x) * (observation.y - average_of_y))
        .sum();
    let denominator: f32 = observations
        .iter()
        .map(|observation| square(observation.x - average_of_x))
        .sum();
    let beta = numerator / denominator;
    let alpha = average_of_y - (beta * average_of_x);
    let model = |x| x * beta + alpha;

    // How well did the model do? Check using Root Mean Square Error.
    // https://en.wikipedia.org/wiki/Root_mean_square_deviation
    let sum_of_squares_of_residuals = observations
        .iter()
        .map(|observation| square(observation.y - model(observation.x)))
        .sum::<f32>();
    let root_mean_square_error = f32::sqrt(sum_of_squares_of_residuals / count);

    let total_sum_of_squares = observations
        .iter()
        .map(|observation| square(observation.y - average_of_y))
        .sum::<f32>();
    let coefficient_of_determination = 1.0 - (sum_of_squares_of_residuals / total_sum_of_squares);

    LinearModel {
        alpha,
        beta,
        count,
        sum_of_x,
        average_of_x,
        sum_of_y,
        average_of_y,
        root_mean_square_error,
        coefficient_of_determination,
    }
}

/// Represents the model and statistics computed from linear regression
pub struct LinearModel {
    // Model parameter values
    pub alpha: f32,
    pub beta: f32,

    // Statistics
    // The count of observations used to train the model
    pub count: f32,
    // The sum and average of the x values of the observations
    pub sum_of_x: f32,
    pub average_of_x: f32,
    // The sum and average of the y values of the observations
    pub sum_of_y: f32,
    pub average_of_y: f32,
    // The RMSE (Root Mean Square Error) of the model
    pub root_mean_square_error: f32,
    // The R² (Coefficient of Determination) of the model
    pub coefficient_of_determination: f32,
}

impl LinearModel {
    /// Use the model to predict the y value for a given x
    #[allow(dead_code)]
    fn predict(&self, x: f32) -> f32 {
        x * self.beta + self.alpha
    }
}

/// Pretty printing of a model and its statistics
impl Display for LinearModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            r#"Model:     y_i = {} * x_i + {}
⍺:         {}
β:         {}
Count:     {}
x average: {} = {} / {}
y average: {} = {} / {}
RMSE:      {}
R²:        {}"#,
            self.beta,
            self.alpha,
            self.alpha,
            self.beta,
            self.count,
            self.average_of_x,
            self.sum_of_x,
            self.count,
            self.average_of_y,
            self.sum_of_y,
            self.count,
            self.root_mean_square_error,
            self.coefficient_of_determination
        )
    }
}

/// Represents a single observation or sample from an experiment
pub struct Observation {
    pub x: f32,
    pub y: f32,
}

/// Prettier, more conventional printing of Observations
impl Debug for Observation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

/// Prettier, more conventional printing of Observations
impl Display for Observation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::physics_ball_drop_experiment;
    use float_cmp::assert_approx_eq;

    #[test]
    fn round_trip() {
        let m = -2.0_f32;
        let b = -10.0_f32;
        let count = 201_f32;

        // Generate the training data in a fixed way
        let domain = 0..(count as usize);
        let training_data: Vec<Observation> = domain
            .map(|x| {
                let x_as_f32 = x as f32;
                Observation {
                    x: x_as_f32,
                    y: m * x_as_f32 + b,
                }
            })
            .collect();

        let model = compute_model(training_data);

        assert_approx_eq!(f32, b, model.alpha);
        assert_approx_eq!(f32, m, model.beta);
        assert_approx_eq!(f32, count, model.count);
        assert_approx_eq!(f32, 20100.0, model.sum_of_x);
        assert_approx_eq!(f32, 100.0, model.average_of_x);
        assert_approx_eq!(f32, -42210.0, model.sum_of_y);
        assert_approx_eq!(f32, -210.0, model.average_of_y);
        assert_approx_eq!(f32, 0.0, model.root_mean_square_error);
        assert_approx_eq!(f32, 1.0, model.coefficient_of_determination);
    }

    #[test]
    fn test_physics_ball_drop_experiment() {
        let model = compute_model(physics_ball_drop_experiment());

        assert_approx_eq!(f32, -0.3452387, model.alpha);
        assert_approx_eq!(f32, 0.8119048, model.beta);
        assert_approx_eq!(f32, 18.0, model.count);
        assert_approx_eq!(f32, 270.0, model.sum_of_x);
        assert_approx_eq!(f32, 15.0, model.average_of_x);
        assert_approx_eq!(f32, 213.0, model.sum_of_y);
        assert_approx_eq!(f32, 11.833333, model.average_of_y);
        assert_approx_eq!(f32, 0.41299996, model.root_mean_square_error);
        assert_approx_eq!(f32, 0.9783022, model.coefficient_of_determination);
    }
}
