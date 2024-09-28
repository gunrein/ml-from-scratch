use llm_from_scratch::{compute_model, Observation};

fn main() {
    // Data from the drop bounce experiment. Units are in number of bricks.
    let physics_drop_bounce_experiment = vec![
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

    println!("{}", compute_model(physics_drop_bounce_experiment));
}
