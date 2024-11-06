use crate::data::{ChopBatch, NUM_FEATURES};
use burn::{
    nn::{
        loss::{MseLoss, Reduction::Mean},
        Linear, LinearConfig, Lstm, LstmConfig, Relu,
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Module, Debug)]
pub struct RegressionModel<B: Backend> {
    input_layer: Linear<B>,
    lstm: Lstm<B>,
    output_layer: Linear<B>,
    activation: Relu,
}

#[derive(Config)]
pub struct RegressionModelConfig {
    #[config(default = 64)]
    pub hidden_size: usize,
    pub n_features: usize,
    pub n_targets: usize,
}

impl RegressionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RegressionModel<B> {
        let input_layer = LinearConfig::new(self.n_features, self.hidden_size)
            .with_bias(true)
            .init(device);
        let lstm = LstmConfig::new(self.hidden_size, self.hidden_size, true).init(device);
        let output_layer = LinearConfig::new(self.hidden_size, self.n_targets)
            .with_bias(true)
            .init(device);

        RegressionModel {
            input_layer,
            lstm,
            output_layer,
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> RegressionModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.input_layer.forward(input);
        let x = self.activation.forward(x);
        self.output_layer.forward(x)
    }

    pub fn forward_step(&self, item: ChopBatch<B>) -> RegressionOutput<B> {
        let targets: Tensor<B, 2> = item.targets.unsqueeze_dim(1);
        let output: Tensor<B, 2> = self.forward(item.inputs);

        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<ChopBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
    fn step(&self, item: ChopBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ChopBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
    fn step(&self, item: ChopBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}
