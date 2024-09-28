use llm_from_scratch::compute_model;
use llm_from_scratch::dataset::physics_ball_drop_experiment;

fn main() {
    println!("{}", compute_model(physics_ball_drop_experiment()));
}
