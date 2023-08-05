/// An incredibly basic neural network that has 1 input, 1 hidden node and 1 output.
struct NeuralNetwork {
    /// Our weight
    w: f32,
    /// Our bias
    b: f32,
}

impl NeuralNetwork {
    fn new() -> Self {
        Self { w: 0., b: 0. }
    }

    fn out(&self, x: &f32) -> f32 {
        x * self.w + self.b
    }

    fn tweak(&mut self, training: &Vec<(f32, f32)>, learning_rate: &f32) {
        let stepw = ddw(&self, &training) * learning_rate;
        let stepb = ddb(&self, &training) * learning_rate;

        self.w -= stepw;
        self.b -= stepb;
    }
}

fn cost(nn: &NeuralNetwork, training: &Vec<(f32, f32)>) -> f32 {
    let mut cost = 0f32;
    for (x, y) in training {
        let out = nn.out(x);
        cost += (out - y).powi(2);
    }
    cost / training.len() as f32
}

fn ddw(nn: &NeuralNetwork, training: &Vec<(f32, f32)>) -> f32 {
    let mut cost = 0f32;
    for (x, y) in training {
        let out = nn.out(x);
        cost += 2. * x * (out - y);
    }
    cost / training.len() as f32
}

fn ddb(nn: &NeuralNetwork, training: &Vec<(f32, f32)>) -> f32 {
    let mut cost = 0f32;
    for (x, y) in training {
        let out = nn.out(x);
        cost += 2. * (out - y);
    }
    cost / training.len() as f32
}

/// The linear function we are mapping our Neural Network to. Change it and see how it *learns*!
fn y(x: &f32) -> f32 {
    -1. * x + 1.
}

// Change these!
const LEARNING_RATE: f32 = 1.1;
const TRAINING_SIZE: usize = 100;
const LEARNING_STEPS: usize = 600;

fn main() {
    let training: Vec<(f32, f32)> = (0..TRAINING_SIZE)
        .map(|x| {
            let r: f32 = x as f32 / TRAINING_SIZE as f32;
            (r, y(&r))
        })
        .collect();

    let mut neural_network = NeuralNetwork::new();

    let c = cost(&neural_network, &training);
    println!("=======BEFORE TWEAKS=======");
    println!("Cost: {c}\nW 0 B 0");
    println!("===========================");

    for i in 30..(LEARNING_STEPS + 30) {
        // This is my way of smoothing out the learning rate. It may not be the best, but it works fine.
        neural_network.tweak(&training, &(LEARNING_RATE / (i as f32).powf(0.5)));
        println!("W {:.5} B {:.5}", neural_network.w, neural_network.b);
    }

    let c = cost(&neural_network, &training);
    println!("=======AFTER  TWEAKS=======");
    println!("Cost: {c}\nW: {} B: {}", neural_network.w, neural_network.b);
    println!("===========================")
}
