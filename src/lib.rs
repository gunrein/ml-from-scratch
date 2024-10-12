pub mod dataset;

use std::fmt::{Debug, Display, Formatter};

/// Compute a linear model from a list of data
pub fn compute_model(observations: &[Observation]) -> LinearModel {
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

/// Generates an HTML+JavaScript plot of the data and model
pub fn plot_as_html(
    title: &str,
    caption: &str,
    y_axis_label: &str,
    x_axis_label: &str,
    observations: Vec<Observation>,
    linear_model: LinearModel,
) -> String {
    let min_max_round_and_cast =
        |min_max: (f32, f32)| (min_max.0.floor() as i32, min_max.1.ceil() as i32);
    let (min_x, max_x) = min_max_round_and_cast(
        observations
            .iter()
            .fold((0_f32, 0_f32), |min_max, observation| {
                (min_max.0.min(observation.x), min_max.1.max(observation.x))
            }),
    );

    // min_x - 1 since the range is inclusive and we want a prediction for at least 1 full unit below the lowest input
    // max_x + 2 since the range is exclusive and we want a prediction for at least 1 full unit above the highest input
    let predictions_as_js_objects = (min_x - 1..max_x + 2)
        .map(|x| {
            let x_f32 = x as f32;
            let prediction = linear_model.predict(x as f32);
            format!("{{x: {x_f32}, prediction: {prediction}}}")
        })
        .collect::<Vec<String>>()
        .join(",");
    let predictions_as_js_array = format!("[{predictions_as_js_objects}]");

    let observations_as_js_objects = observations
        .iter()
        .map(|observation| format!("{{x: {}, y: {}}}", observation.x, observation.y))
        .collect::<Vec<String>>()
        .join(",");

    let observations_as_js_array = format!("[{observations_as_js_objects}]");
    let plot_code = format!(
        r#"Plot.plot({{
        caption: "{caption}",
        grid: true,
        marks: [
            Plot.axisY({{label: "{y_axis_label}"}}),
            Plot.axisX({{label: "{x_axis_label}"}}),
            Plot.dot(data, {{x: "x", y: "y", tip: true}}),
            Plot.crosshair(data, {{x: "x", y: "y"}}),
            Plot.lineY(predictions, {{x: "x", y: "prediction", stroke: "blue", strokeOpacity: 0.5}}),
            Plot.ruleY([0]),
            Plot.ruleX([0])
        ]
    }})"#
    );

    let alpha = linear_model.alpha;
    let beta = linear_model.beta;
    let count = linear_model.count;
    let average_of_x = linear_model.average_of_x;
    let sum_of_x = linear_model.sum_of_x;
    let average_of_y = linear_model.average_of_y;
    let sum_of_y = linear_model.sum_of_y;
    let root_mean_square_error = linear_model.root_mean_square_error;
    let coefficient_of_determination = linear_model.coefficient_of_determination;

    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <link rel="stylesheet" href="https://cdn.simplecss.org/simple.min.css">
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        #model_info td:nth-child(2) {{
            text-align: right;
        }}
    </style>
</head>
<body>
    <section id="model_info">
        <h1>{title}</h1>
        <table>
            <tr>
                <td>Model</td>
                <td>\(y = {beta} x + ({alpha})\)</td>
            </tr>
            <tr>
                <td>\(\alpha\)</td>
                <td>\({alpha}\)</td>
            </tr>
            <tr>
                <td>\(\beta\)</td>
                <td>\({beta}\)</td>
            </tr>
            <tr>
                <td>Count</td>
                <td>\({count}\)</td>
            </tr>
            <tr>
                <td>Average of \(x\)</td>
                <td>\({average_of_x} = {sum_of_x} / {count}\)</td>
            </tr>
            <tr>
                <td>Average of \(y\)</td>
                <td>\({average_of_y} = {sum_of_y} / {count}\)</td>
            </tr>
            <tr>
                <td>\(RMSE\)</td>
                <td>\({root_mean_square_error}\)</td>
            </tr>
            <tr>
                <td>\(R^2\)</td>
                <td>\({coefficient_of_determination}\)</td>
            </tr>
        </table>
        <div id="the_plot"></div>
    </section>
</body>
<script type="module">
    import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";

    const data = {observations_as_js_array};
    const predictions = {predictions_as_js_array};
    const plot = {plot_code};
    const div = document.querySelector("#the_plot");
    div.append(plot);
</script>
</html>        
"##
    )
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
            r#"Model:     y = ({})x + ({})
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

        let model = compute_model(&training_data);

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
        let model = compute_model(&physics_ball_drop_experiment());

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
