use burn::nn::{
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    Dropout, DropoutConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pool: AdaptiveAvgPool1d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}
#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            pool: AdaptiveAvgPool1dConfig::new(8).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Values [batch_size, sample_size]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, values: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, length] = values.dims();

        // Create a channel at the second dimension.
        let x = values.reshape([batch_size, 1, length]);

        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }
}
