use burn::{data::dataloader::batcher::Batcher, prelude::*};

#[derive(Clone)]
pub struct ChopBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> ChopBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Debug, Clone)]
pub struct Values {
    pub values: Vec<f32>,
    pub label: f32,
}

#[derive(Clone, Debug)]
pub struct ChopBatch<B: Backend> {
    pub values: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<Values, ChopBatch<B>> for ChopBatcher<B> {
    fn batch(&self, items: Vec<Values>) -> ChopBatch<B> {
        let values = items
            .iter()
            .map(|item| TensorData::from(item.values.as_slice()).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            .map(|tensor| tensor.reshape([1, 28]))
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    [(item.label as i64).elem::<B::IntElem>()],
                    &self.device,
                )
            })
            .collect();

        let values = Tensor::cat(values, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        ChopBatch { values, targets }
    }
}
