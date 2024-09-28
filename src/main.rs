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
        .map(|i| {
            f32::powf(
                physics_drop_bounce_experiment.get(*i).unwrap().x - x_average,
                2.0,
            )
        })
        .sum();
    let beta = numerator / denominator;
    let alpha = y_average - (beta * x_average);
    println!("Model: y_i = {beta} * x_i + {alpha}")
}
