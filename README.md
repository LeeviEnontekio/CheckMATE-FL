# CheckMATE-FL

This repository contains code for CheckMATE distillation and Federated Learning simulation.

Project structure\
├── Distillation\
└── Simulation

## Distillation

CheckMATE distillation for UCF101 and HMDB51 datasets can be found in their respective Jupyter Notebooks.

## Simulation

The simulation folder contents can be pasted into flwr 
quickstart project. 

Please refer to [Flwr Tensorflow Quickstart Guide](https://flower.ai/docs/framework/tutorial-quickstart-tensorflow.html)

# Model and Datasets

ConvNext model weights can be downloaded from [here](https://github.com/leondgarse/keras_cv_attention_models?tab=readme-ov-file#convnext)

# Dependencies

- flwr
- Tensorflow (should be installed automatically with flwr by following the quickstart guide)
- OpenCV 
- Ray (required for flwr simulation as flwr sometimes struggles to install it by itself)



Dataset downloads: 
- [UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/)
- [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)



# Note!

- `TF_FORCE_GPU_ALLOW_GROWTH=true` environment variable must be set when using Flwr framework. 

- Sometimes when training with Flwr, out of memory error appears around 10 epoch mark. Unknown if the error stems from Flwr or Tensorflow model leaks memory.