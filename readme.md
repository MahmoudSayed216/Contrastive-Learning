## Contrastive Learning using Triplet Loss and Contrastive Loss on the MNIST dataset.


This is a Comprehensive notebook showcasing some of the differences between Contrastive Learning via Contrastive Loss and Triplet Loss





first of all, due to the simplicity of the dataset, the differences may not be obviously clear, but we'll show as much differences as we can.



first, the network architecture:
the network is a simple resnet with 4 ResBlocks[conv + batch norm + leaky relu] and 2 fully connected layers

Input -> ResBlock -> ResBlock -> Pooling -> ResBlock -> Pooling -> ResBlock -> GAP -> linear -> bn -> lrelu -> linear



the network showed the exact same results on both loss functions
here are the metrics i used to evaluate both networks:
- Triplet Loss/Contrastive Loss
- KNN Accuracy/Precision/Recall [fitted a KNN on the embeddings of the training data, and evaluated on the embeddings of the test set]
- Precision and Recall on the models predictions 
- ROC AUC
- Precision-Recall Curve


Contrastive:
21:22:36 | INFO | Best validation contrastive loss: 0.0029
21:22:36 | INFO | Running verification evaluation (Accuracy / ROC / Precision-Recall) on best checkpoint...
21:22:41 | INFO | Verification -> Accuracy: 0.9606 | Precision: 0.9297 | Recall: 0.9978 | ROC AUC: 0.9990 | Average Precision: 0.9991
21:22:41 | INFO | ROC curve data exported to checkpoints/roc_curve.csv
21:22:41 | INFO | Precision-Recall curve data exported to checkpoints/precision_recall_curve.csv

21:22:36 | INFO | Epoch 30/30 | Train Contrastive Loss: 0.0004 | Val Contrastive Loss: 0.0029 | LR: 0.001250
21:22:36 | INFO |   KNN (k=5) -> Acc: 0.9948 | Precision (macro): 0.9948 | Recall (macro): 0.9947 | F1 (macro): 0.9947
21:22:36 | INFO |   -> New best model (val_loss=0.0029) saved to checkpoints/best_model.pt

Triplet:
21:29:10 | INFO | Training complete.
21:29:10 | INFO | Best validation triplet loss: 0.0021
21:29:10 | INFO | Running verification evaluation (Accuracy / ROC / Precision-Recall) on best checkpoint...
21:29:17 | INFO | Verification -> Accuracy: 0.5154 | Precision: 1.0000 | Recall: 0.0308 | ROC AUC: 0.9997 | Average Precision: 0.9997
21:29:17 | INFO | ROC curve data exported to checkpoints/roc_curve.csv
21:29:17 | INFO | Precision-Recall curve data exported to checkpoints/precision_recall_curve.csv

21:26:28 | INFO | Epoch 27/30 | Train Triplet Loss: 0.0030 | Val Triplet Loss: 0.0021 | LR: 0.001250
21:26:28 | INFO |   KNN (k=5) -> Acc: 0.9935 | Precision (macro): 0.9935 | Recall (macro): 0.9934 | F1 (macro): 0.9935
21:26:28 | INFO |   -> New best model (val_loss=0.0021) saved to checkpoints/best_model.pt

the slight differences show up in the t-SNE and UMAP visualizations

both have

