use std::fmt::{Debug, Display, Formatter};

/// Represents a single observation or sample from an experiment
struct Observation {
    x: f32,
    y: f32,
}

/// Prettier, more conventional printing of Observations
impl Debug for Observation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl Display for Observation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

fn square(number: f32) -> f32 {
    f32::powf(number, 2.0)
}

fn main() {
    // Data from the drop bounce experiment. Units are in number of bricks.
    let physics_drop_bounce_experiment = [
        Observation { x: 10.0, y: 7.5 },
        Observation { x: 10.0, y: 7.5 },
        Observation { x: 10.0, y: 7.0 },
        Observation { x: 12.0, y: 10.0 },
        Observation { x: 12.0, y: 10.0 },
        Observation { x: 12.0, y: 9.75 },
        Observation { x: 14.0, y: 11.0 },
        Observation { x: 14.0, y: 11.0 },
        Observation { x: 14.0, y: 10.5 },
        Observation { x: 16.0, y: 12.75 },
        Observation { x: 16.0, y: 13.0 },
        Observation { x: 16.0, y: 13.0 },
        Observation { x: 18.0, y: 14.75 },
        Observation { x: 18.0, y: 14.0 },
        Observation { x: 18.0, y: 14.5 },
        Observation { x: 20.0, y: 15.25 },
        Observation { x: 20.0, y: 15.5 },
        Observation { x: 20.0, y: 16.0 },
    ];

    let domain = 0..physics_drop_bounce_experiment.len();

    println!("Data: {physics_drop_bounce_experiment:#?}");
    let count = physics_drop_bounce_experiment.len() as f32;
    let y_sum = physics_drop_bounce_experiment
        .iter()
        .map(|sample| sample.y)
        .sum::<f32>();
    let y_average = y_sum / count;
    println!("y average: {y_average} = {y_sum} / {count}");
    let x_sum = physics_drop_bounce_experiment
        .iter()
        .map(|sample| sample.x)
        .sum::<f32>();
    let x_average = x_sum / count;
    println!("x average: {x_average} = {x_sum} / {count}");

    // Calculate the Ordinary Least Squares linear regression for the training data
    // https://en.wikipedia.org/wiki/Ordinary_least_squares
    let indices: Vec<usize> = domain.clone().collect();
    let numerator: f32 = indices
        .iter()
        .map(|i| {
            (physics_drop_bounce_experiment.get(*i).unwrap().x - x_average)
                * (physics_drop_bounce_experiment.get(*i).unwrap().y - y_average)
        })
        .sum();
    let denominator: f32 = indices
        .iter()
        .map(|i| square(physics_drop_bounce_experiment.get(*i).unwrap().x - x_average))
        .sum();
    let beta = numerator / denominator;
    let alpha = y_average - (beta * x_average);
    let model = |x| x * beta + alpha;
    println!("Model:     y_i = {beta} * x_i + {alpha}");

    // How well did the model do? Check using Root Mean Square Error.
    // https://en.wikipedia.org/wiki/Root_mean_square_deviation
    let sum_of_squares_of_residuals = indices
        .iter()
        .map(|i| {
            let observation = physics_drop_bounce_experiment.get(*i).unwrap();
            let y_actual = observation.y;
            let y_predicted = model(observation.x);
            square(y_actual - y_predicted)
        })
        .sum::<f32>();
    let root_mean_square_error = f32::sqrt(sum_of_squares_of_residuals / count);
    println!("RMSE:      {root_mean_square_error}");

    let total_sum_of_squares = indices
        .iter()
        .map(|i| {
            let observation = physics_drop_bounce_experiment.get(*i).unwrap();
            let y_actual = observation.y;
            square(y_actual - y_average)
        })
        .sum::<f32>();
    let coefficient_of_determination = 1.0 - (sum_of_squares_of_residuals / total_sum_of_squares);
    println!("R²:        {coefficient_of_determination}");
}
