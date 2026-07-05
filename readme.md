## Contrastive Learning using Triplet Loss and Contrastive Loss on the MNIST dataset.


This is a Comprehensive notebook showcasing some of the differences between Contrastive Learning via Contrastive Loss and Triplet Loss





first of all, due to the simplicity of the dataset, the differences may not be obviously clear, but we'll show as much differences as we can.



first, the network architecture:
the network is a simple resnet with 4 ResBlocks[conv + batch norm + leaky relu] and 2 fully connected layers

Input -> ResBlock -> ResBlock -> Pooling -> ResBlock -> Pooling -> ResBlock -> GAP -> linear -> bn -> lrelu -> linear



the network showed the some interesting results for both loss functions
but it seems like the Triplet Network had better verification metrics values overall 
here are the metrics i used to evaluate both networks:
- Triplet Loss/Contrastive Loss
- KNN Accuracy/Precision/Recall [fitted a KNN on the embeddings of the training data, and evaluated on the embeddings of the test set]
- Precision and Recall on the contrastive-models predictions
- ROC AUC
- Precision-Recall Curve



## Loss values

| Metric | Contrastive | Triplet |
|---|---|---|
| Train Loss | 0.0004 | 0.0034 |
| Val Loss | 0.0029 | 0.0029 |
| LR | 0.001250 | 0.002500 |


## Verification Metrics

| Metric | Contrastive | Triplet |
|---|---|---|
| Accuracy | 0.9606 | **0.9918** |
| Precision | 0.9297 | **0.9920** |
| Recall | 0.9978 | **0.9917** |
| ROC AUC | 0.9990 | **0.9996** |
| Average Precision | 0.9991 | **0.9996** |


## KNN (k=5) Evaluation

| Metric | Contrastive | Triplet |
|---|---|---|
| Accuracy | 0.9948 | **0.9949** |
| Precision (macro) | 0.9948 | **0.9949** |
| Recall (macro) | 0.9947 | **0.9948** |
| F1 (macro) | 0.9947 | **0.9949** |


there are several reasons Triplets performed better than Contrastive most importantly is that Triplets aims to ensure that d(a,p) is less then d(a,n) without exceeding a specific margin. whereas Contrastive loss only performs on pairs, either getting similar pairs closer together or getting dissimilar pairs further apart. the first approach gives each item a sense of where it stand relative to other classes, whereas the contrastive operates in a way that makes it unaware of other classes, there's no relativity involved here, only absolute distances between pairs. the triplets approach should in theory result in better separation between different classes.


The visualizations below [using t-SNE and UMAP] show how each method separate between different classes
It's very clear that Triplets have a far better separation then Contrastive and Contrastive tends to have some tails dragged towards other classes.

Also, due to the way Triplets work, which is having both a positive and a negative sample in consideration while training, you can see that similar classes are closer together [like 1 and 7, 3 and 8, 9 and 4], you can hardly observe that in contrastive due to its nature of only comparing between a pair at a time.

Finally, a weird observation in Contrastive is that when they are drawn in 3D, they look kind of flat or really really thin, I don't have any explanation to that, but on the contrary, the 3D graph of the Triplets didn't show this phenomena, each cluster has a chunky shape that's not thin or flat.

below is some the graphs made by t-SNE and UMAP showing the separation of each method between different classes 

# Embedding Visualizations: Contrastive vs Triplet Loss

> Place this file in the same parent folder as your `pics/` directory so the image paths resolve correctly.

## t-SNE — 2D

| Contrastive | Triplet |
|---|---|
| ![t-SNE Contrastive 2D](pics/t-SNE%20CONTRASTIVE%202D.png) | ![t-SNE Triplet 2D](pics/t-SNE%20TRIPLET%202D.png) |
| **t-SNE CONTRASTIVE 2D.png** | **t-SNE TRIPLET 2D.png** |

## t-SNE — 3D (1)

| Contrastive | Triplet |
|---|---|
| ![t-SNE Contrastive 3D 1](pics/t-SNE%20CONTRASTIVE%203D%201.png) | ![t-SNE Triplet 3D 1](pics/t-SNE%20TRIPLET%203D%201.png) |
| **t-SNE CONTRASTIVE 3D 1.png** | **t-SNE TRIPLET 3D 1.png** |

## t-SNE — 3D (2)

| Contrastive | Triplet |
|---|---|
| ![t-SNE Contrastive 3D 2](pics/t-SNE%20CONTRASTIVE%203D%202.png) | ![t-SNE Triplet 3D 2](pics/t-SNE%20TRIPLET%203D%202.png) |
| **t-SNE CONTRASTIVE 3D 2.png** | **t-SNE TRIPLET 3D 2.png** |

## t-SNE — 3D (3)

| Contrastive | Triplet |
|---|---|
| ![t-SNE Contrastive 3D 3](pics/t-SNE%20CONTRASTIVE%203D%203.png) | ![t-SNE Triplet 3D 3](pics/t-SNE%20TRIPLET%203D%203.png) |
| **t-SNE CONTRASTIVE 3D 3.png** | **t-SNE TRIPLET 3D 3.png** |

## UMAP — 2D

| Contrastive | Triplet |
|---|---|
| ![UMAP Contrastive 2D](pics/UMAP%20CONTRASTIVE%202D%20.png) | ![UMAP Triplet 2D](pics/UMAP%20TRIPLET%202D.png) |
| **UMAP CONTRASTIVE 2D .png** | **UMAP TRIPLET 2D.png** |

## UMAP — 3D (1)

| Contrastive | Triplet |
|---|---|
| ![UMAP Contrastive 3D 1](pics/UMAP%20CONTRASTIVE%203D%201.png) | ![UMAP Triplet 3D 1](pics/UMAP%20TRIPLET%203D%201.png) |
| **UMAP CONTRASTIVE 3D 1.png** | **UMAP TRIPLET 3D 1.png** |

## UMAP — 3D (2)

| Contrastive | Triplet |
|---|---|
| ![UMAP Contrastive 3D 2](pics/UMAP%20CONTRASTIVE%203D%202.png) | ![UMAP Triplet 3D 2](pics/UMAP%20TRIPLET%203D%202.png) |
| **UMAP CONTRASTIVE 3D 2.png** | **UMAP TRIPLET 3D 2.png** |

## UMAP — 3D (3)

| Contrastive | Triplet |
|---|---|
| ![UMAP Contrastive 3D 3](pics/UMAP%20CONTRASTIVE%203D%203.png) | ![UMAP Triplet 3D 3](pics/UMAP%20TRIPLET%203D%203.png) |
| **UMAP CONTRASTIVE 3D 3.png** | **UMAP TRIPLET 3D 3.png** |