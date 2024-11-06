use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, SqliteDataset},
    },
    prelude::*,
    serde,
};

pub const NUM_FEATURES: usize = 8;

// Pre-computed statistics for the housing dataset features
const FEATURES_MIN: [f32; NUM_FEATURES] = [0.4999, 1., 0.8461, 0.375, 3., 0.6923, 32.54, -124.35];
const FEATURES_MAX: [f32; NUM_FEATURES] = [
    15., 52., 141.9091, 34.0667, 35682., 1243.3333, 41.95, -114.31,
];

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChopItem {
    pub features: Vec<f32>,
    pub targets: Vec<f32>,
}

/// Normalizer for the housing dataset.
#[derive(Clone, Debug)]
pub struct Normalizer<B: Backend> {
    pub min: Tensor<B, 2>,
    pub max: Tensor<B, 2>,
}

impl<B: Backend> Normalizer<B> {
    /// Creates a new normalizer.
    pub fn new(device: &B::Device, min: &[f32], max: &[f32]) -> Self {
        let min = Tensor::<B, 1>::from_floats(min, device).unsqueeze();
        let max = Tensor::<B, 1>::from_floats(max, device).unsqueeze();
        Self { min, max }
    }

    /// Normalizes the input image according to the housing dataset min/max.
    pub fn normalize(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        (input - self.min.clone()) / (self.max.clone() - self.min.clone())
    }
}

#[derive(Clone, Debug)]
pub struct ChopBatcher<B: Backend> {
    device: B::Device,
    normalizer: Normalizer<B>,
}

#[derive(Clone, Debug)]
pub struct ChopBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}

impl<B: Backend> ChopBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            device: device.clone(),
            normalizer: Normalizer::new(&device, &FEATURES_MIN, &FEATURES_MAX),
        }
    }
}

impl<B: Backend> Batcher<ChopItem, ChopBatch<B>> for ChopBatcher<B> {
    fn batch(&self, items: Vec<ChopItem>) -> ChopBatch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();

        for item in items.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats(item.features.as_slice(), &self.device);

            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0);
        let inputs = self.normalizer.normalize(inputs);

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats(item.targets.as_slice(), &self.device))
            .collect();

        let targets = Tensor::cat(targets, 0);

        ChopBatch { inputs, targets }
    }
}
