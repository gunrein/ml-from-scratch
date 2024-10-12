use ml_from_scratch::dataset::physics_ball_drop_experiment;
use ml_from_scratch::{compute_model, plot_as_html};
use std::fs;

fn main() -> std::io::Result<()> {
    let data = physics_ball_drop_experiment();
    let model = compute_model(&data);
    println!("{}", model);

    let path = "examples/physics-ball-drop-experiment.html";
    println!("Plot rendered to {}", path);
    fs::write(
        path,
        plot_as_html(
            "Physics ball drop experiment", 
            "Physics experiment measuring bounce height of a rubber ball when dropped from a given height",
            "Bounce height measured in number of bricks",
            "Drop height measured in number of bricks",
            data,
            model
        ),
    )
}
