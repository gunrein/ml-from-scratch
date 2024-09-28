use rand::distributions::{Distribution, Uniform};

fn main() {
    // let linear_function = |x, m, b| (m * x) + b;

    // This example is a bit funny because we're using a known linear function to generate
    // the data to train the model (i.e. find the model parameters `m` and `b`) so it is a closed
    // loop.


    // Grab a random number generator (rng)
    let mut random_number_generator = rand::thread_rng();

    // Create a distribution of [-99,100) to draw random samples from
    let distribution = Uniform::from(-99..100);
    let domain = 0..201;
    let y = 0..201;

    // Generate 100 random samples from the distribution and store them in a vector (i.e. array)
    let xs: Vec<i64> = domain.clone().collect();
    let training_data: Vec<i64> = xs.iter().map(|_x| distribution.sample(&mut random_number_generator)).collect();

    // Output the training data so we can see it
    println!("Training data: {training_data:?}");
    let y_sum = training_data.iter().sum::<i64>() as f64;
    let y_count = training_data.len() as f64;
    let y_average = y_sum / y_count;
    println!("y average: {y_average} = {y_sum} / {y_count}");
    let x_sum = xs.iter().sum::<i64>() as f64;
    let x_count = xs.len() as f64;
    let x_average = x_sum / x_count;
    println!("x average: {x_average} = {x_sum} / {x_count}");

    // Calculate the Ordinary Least Squares linear regression for the training data
    let indices: Vec<i64> = domain.clone().collect();
    let numerator: f64 = indices.iter().map(|i| (*xs.get(*i as usize).unwrap() as f64 - x_average) * (*training_data.get(*i as usize).unwrap() as f64 - y_average)).sum();
    let denominator: f64 = indices.iter().map(|i| f64::powf(*xs.get(*i as usize).unwrap() as f64 - x_average, 2.0_f64)).sum();
    let beta = numerator / denominator;
    let alpha = y_average - (beta * x_average);
    println!("Model: y_i = {beta} * x_i + {alpha}")
}
