use std::fmt::Debug;

use crate::data::{ChopBatcher, ChopItem};
use crate::model::{RegressionModel, RegressionModelConfig};
use burn::config;
use burn::optim::AdamConfig;
use burn::train::renderer::{MetricState, MetricsRenderer, TrainingProgress};
use burn::train::Learner;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset, dataset::InMemDataset},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, LearnerBuilder},
};

#[derive(Config)]
pub struct ExpConfig {
    #[config(default = 100)]
    pub num_epochs: usize,

    #[config(default = 2)]
    pub num_workers: usize,

    #[config(default = 1337)]
    pub seed: u64,

    pub optimizer: AdamConfig,

    #[config(default = 256)]
    pub batch_size: usize,
}

impl Debug for ExpConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExpConfig")
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

struct ChopRenderer;

impl MetricsRenderer for ChopRenderer {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, item: TrainingProgress) {}

    fn render_valid(&mut self, item: TrainingProgress) {}
}

pub fn create_model<B: AutodiffBackend>(
    artifact_dir: &str,
    device: B::Device,
) -> RegressionModel<B> {
    create_artifact_dir(artifact_dir);

    // Config
    RegressionModelConfig::new(8, 8).init(&device)
}

pub fn train<B: AutodiffBackend>(
    device: B::Device,
    model: RegressionModel<B>,
    train_dataset: InMemDataset<ChopItem>,
    valid_dataset: InMemDataset<ChopItem>,
) -> RegressionModel<B> {
    create_artifact_dir("artifacts");
    let optimizer = AdamConfig::new();
    let config = ExpConfig::new(optimizer);
    B::seed(config.seed);

    println!("Train Dataset Size: {}", train_dataset.len());
    println!("Valid Dataset Size: {}", valid_dataset.len());

    let batcher_train = ChopBatcher::<B>::new(device.clone());

    let batcher_test = ChopBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);
    let learner = LearnerBuilder::new("artifacts")
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        // .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .renderer(ChopRenderer)
        .build(model, config.optimizer.init(), 1e-3);

    let model_trained = learner.fit(dataloader_train, dataloader_test);
    model_trained
}
