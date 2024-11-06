pub mod data;
pub mod model;
pub mod training;
use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    data::{dataloader::batcher::Batcher, dataset::InMemDataset},
    tensor::{backend::AutodiffBackend, Tensor},
};
use model::RegressionModel;
use td_rs_chop::param::MenuParam;
use td_rs_chop::*;
use td_rs_derive::{Param, Params};
use training::ExpConfig;

#[derive(Param, Default, Clone, Debug)]
enum Operation {
    #[default]
    Add,
    Multiply,
    Power,
}

#[derive(Params, Default, Clone, Debug)]
struct LittleLearnerChopParams {
    #[param(label = "Train", page = "Little Learner")]
    train: bool,
}

/// Struct representing our CHOP's state
#[derive(Debug)]
pub struct LittleLearnerChop {
    params: LittleLearnerChopParams,
    model: Option<RegressionModel<Autodiff<Wgpu>>>,
}

impl LittleLearnerChop {
    fn train(&mut self, inputs: &OperatorInputs<ChopInput>) {
        let _input_feats = inputs.input(0);
        let _input_targets = inputs.input(1);

        if _input_feats.is_none() || _input_targets.is_none() {
            return;
        }
        let input_feats = _input_feats.unwrap();
        let input_targets = _input_targets.unwrap();

        let mut items = vec![];
        let num_samples = input_feats.num_samples();
        for j in 0..num_samples {
            let mut targets_sample = vec![];
            let mut features_sample = vec![];
            for i in 0..input_feats.num_channels() {
                features_sample.push(input_feats[i][j]);
            }
            for i in 0..input_targets.num_channels() {
                targets_sample.push(input_targets[i][j]);
            }
            items.push(data::ChopItem {
                features: features_sample,
                targets: targets_sample,
            });
        }

        let train_dataset = InMemDataset::new(items.clone());
        let valid_dataset = InMemDataset::new(items);

        let model = self.model.take().unwrap();
        self.model = Some(training::train(
            WgpuDevice::default(),
            model,
            train_dataset,
            valid_dataset,
        ));
    }
}

/// Impl block providing default constructor for plugin
impl OpNew for LittleLearnerChop {
    fn new(_info: NodeInfo) -> Self {
        let model = training::create_model::<Autodiff<Wgpu>>("artifacts", WgpuDevice::default());
        Self {
            params: LittleLearnerChopParams { train: false },
            model: Some(model),
        }
    }
}

impl OpInfo for LittleLearnerChop {
    const OPERATOR_LABEL: &'static str = "Little Learner";
    const OPERATOR_TYPE: &'static str = "Littlelearner";
}

impl Op for LittleLearnerChop {
    fn params_mut(&mut self) -> Option<Box<&mut dyn OperatorParams>> {
        Some(Box::new(&mut self.params))
    }
}

impl Chop for LittleLearnerChop {
    #[tracing::instrument]
    fn execute(&mut self, output: &mut ChopOutput, inputs: &OperatorInputs<ChopInput>) {
        let params = inputs.params();
        params.enable_param("Train", self.params.train);
        let is_training = self.params.train;

        if is_training {
            self.train(inputs);
        } else {
            let input_feats = inputs.input(0).unwrap();
            let num_samples = input_feats.num_samples();
            let num_channels = input_feats.num_channels();
            let mut samples = vec![];
            let device = WgpuDevice::default();
            for j in 0..num_samples {
                let mut features_sample = vec![];
                for i in 0..num_channels {
                    features_sample.push(input_feats[i][j]);
                }
                samples.push(Tensor::from_floats(features_sample.as_slice(), &device));
            }
            let input = Tensor::cat(samples, 0);
            let output_tensor = self.model.as_ref().unwrap().forward(input);
            for (i, item) in output_tensor.into_data().iter().enumerate() {
                output[i][0] = item;
            }
        }
    }

    fn general_info(&self, _inputs: &OperatorInputs<ChopInput>) -> ChopGeneralInfo {
        ChopGeneralInfo {
            cook_every_frame: false,
            cook_every_frame_if_asked: false,
            timeslice: false,
            input_match_index: 0,
        }
    }

    fn channel_name(&self, index: usize, _inputs: &OperatorInputs<ChopInput>) -> String {
        format!("chan{}", index)
    }

    fn output_info(&self, inputs: &OperatorInputs<ChopInput>) -> Option<ChopOutputInfo> {
        let input_targets = inputs.input(1)?;
        Some(ChopOutputInfo {
            num_channels: input_targets.num_channels() as u32,
            num_samples: input_targets.num_samples() as u32,
            start_index: 0,
            ..Default::default()
        })
    }
}

chop_plugin!(LittleLearnerChop);
