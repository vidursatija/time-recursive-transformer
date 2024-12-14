# Time Recursive Transformer

## Abstract
Inspired by Google's [Relaxed Recursive Transformer](https://arxiv.org/pdf/2410.20672), I want to compress the transformer even further. Diffusion models often run the same network across multiple time steps. Using both of these ideas, I came up with time recursive transformers. The model is pretty straightforward. Rather than having N layers with different weights, we have 1 input layer, k repeating middle layers with shared weights, 1 output layer. To differentiate between the k middle layers, we add time embeddings to the input.

## Results
I fine-tuned ViT-base-patch16-224 due to compute restrictions.

### ImageNet-1k performance on base model
Accuracy is 80.27% on the validation set.

### ImageNet-1k performance on new model after being initialized using base model
Accuracy is 55.83% on the validation set after training for 4 epochs. The loss had almost stopped decreasing after 4 epochs.


Due to the poor performance, no further fine-tuning was performed. The notebook to reproduce the results are in train_re_transformer.ipynb file.