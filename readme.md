## Contrastive Learning using Triplet Loss and Contrastive Loss on the MNIST dataset.


This is a Comprehensive notebook showcasing some of the differences between Contrastive Learning via Contrastive Loss and Triplet Loss





first of all, due to the simplicity of the dataset, the differences may not be obviously clear, but we'll show as much differences as we can.



first, the network architecture:
the network is a simple resnet with 4 ResBlocks[conv + batch norm + leaky relu] and 2 fully connected layers

Input -> ResBlock -> ResBlock -> Pooling -> ResBlock -> Pooling -> ResBlock -> GAP -> linear -> bn -> lrelu -> linear



the network should the exact same results on both loss functions
Contrastive:
17:54:29 | INFO | Epoch 25/30 | Train Contrastive Loss: 0.0009 | Val Contrastive Loss: 0.0027 | LR: 0.002500
17:54:29 | INFO |   KNN (k=5) -> Acc: 0.9950 | Precision (macro): 0.9950 | Recall (macro): 0.9949 | F1 (macro): 0.9950
17:54:29 | INFO |   -> New best model (val_loss=0.0027) saved to checkpoints/best_model.pt

Triplet:
17:09:48 | INFO | Epoch 26/30 | Train Triplet Loss: 0.0032 | Val Triplet Loss: 0.0031 | LR: 0.002500
17:09:48 | INFO |   KNN (k=5) -> Acc: 0.9944 | Precision (macro): 0.9944 | Recall (macro): 0.9943 | F1 (macro): 0.9944
17:09:48 | INFO |   -> New best model (val_loss=0.0031) saved to checkpoints/best_model.pt


the slight differences show up in the t-SNE and UMAP visualizations


