# 👗 Fashion Compatibility Learning

> Deep Learning course project — University of Bern / Neuchâtel  
> Professor: **Paolo Favaro**

---

## Overview

A major challenge when building an outfit is knowing how well different clothing items go together. This project proposes a way to **measure compatibility between clothing items** (tops, bottoms, shoes) using deep learning and metric learning techniques.

We frame compatibility as a **metric learning problem**: given images of individual clothing items, we learn visual embeddings such that compatible items (from the same curated outfit) are mapped nearby in the embedding space, while incompatible items are pushed apart.

---

## Research Question

> Can modern general-purpose vision backbones (Swin Transformer, ConvNeXt) learn sufficient compatibility signals **without explicit type conditioning**, reducing the need for type-aware supervision as in prior work?

---

## Repository Structure

```
Deep-Learning-Project/
│
├── data_pipeline.ipynb        # Sandra — dataset loading, exploration & triplet sampling
├── model_training.ipynb       # Allizha — model architecture, training & experiments
├── references.bib             # BibTeX references
├── main.tex                   # Project report (LaTeX, NeurIPS style)
└── README.md
```

---

## Dataset

We use the **[Marqo/polyvore](https://huggingface.co/datasets/Marqo/polyvore)** dataset from Hugging Face — a subset of the publicly available Polyvore benchmark.

| Property | Value |
|---|---|
| Total items | 94,096 |
| Split | `data` (single split) |
| Columns | `image`, `category`, `text`, `item_ID` |
| Outfit grouping | Via `item_ID` prefix (e.g. `100002074_1` → outfit `100002074`) |

Outfit groupings are used to construct **compatible pairs** (anchor + positive from the same outfit) and **incompatible pairs** (negative from a different outfit).

---

## Method

### Backbones (pretrained on ImageNet-1K)

| Model | Type | Params |
|---|---|---|
| **ConvNeXt-T** | Modern CNN | ~28M |
| **Swin-T** | Vision Transformer | ~28M |

Both backbones are followed by a **2-layer MLP projection head** producing L2-normalized embeddings of dimension 128.

### Training Strategy

Two-phase fine-tuning to handle GPU memory constraints:

- **Phase 1 (3 epochs):** Backbone frozen → only projection head trained  
- **Phase 2 (up to 30 epochs):** Full fine-tuning with differential learning rates  
  - Backbone: `lr = 1e-5`  
  - Projection head: `lr = 1e-4`

### Loss Functions

**Triplet Margin Loss:**
$$\mathcal{L} = \max(0,\ d(a,p) - d(a,n) + \alpha)$$

**Contrastive Loss:**
$$\mathcal{L} = d_{pos}^2 + \max(0,\ m - d_{neg})^2$$

### Retrieval

At inference: cosine-similarity **k-NN** over all test embeddings.

---

## Experiments

We run **4 configurations** comparing backbones and loss functions:

| Run | Backbone | Loss | Best Val Loss |
|---|---|---|---|
| 1 | ConvNeXt-T | Triplet | 0.1984 ✅ |
| 2 | ConvNeXt-T | Contrastive | pending |
| 3 | Swin-T | Triplet | pending |
| 4 | Swin-T | Contrastive | pending |

### Evaluation Metrics

- **AUC** — binary compatibility prediction (compatible vs. random pair)
- **FITB** — Fill-in-the-Blank accuracy (select correct item from 4 candidates)
- **Recall@10** — fraction of true compatible items in top-10 retrieved results

---

## Setup & Usage

### 1. Requirements

```bash
pip install datasets torch torchvision Pillow numpy pandas matplotlib tqdm
```

### 2. Run the data pipeline

Open `data_pipeline.ipynb` in Google Colab with a **T4 GPU** runtime.  
Loads the Polyvore dataset, builds the `PolyvoreDataset` class with triplet sampling, and creates DataLoaders.

### 3. Run model training

Open `model_training.ipynb` in Google Colab with a **T4 GPU** runtime.  
In Cell 8, change these 2 lines to switch between the 4 experiments:

```python
BACKBONE = 'convnext_tiny'   # or 'swin_t'
LOSS_FN  = 'triplet'         # or 'contrastive'
```

Checkpoints are saved automatically to Google Drive.

> ⚠️ Use `BATCH_SIZE = 16` to avoid CUDA out-of-memory errors on a T4 GPU.

---

## Authors & Collaboration

- Allizha Theiventhiram — University of Neuchâtel  
- Sandra Nikoloska — University of Bern  
- Boris Verdecia Echarte — University of Neuchâtel
---

## References

1. Tan et al. — *Learning Similarity Conditions Without Explicit Supervision*, ICCV 2019 — [arXiv](https://arxiv.org/pdf/1908.08589)  
2. Sarkar et al. — *OutfitTransformer: Learning Outfit Representations for Fashion Recommendation*, WACV 2023 — [arXiv](https://arxiv.org/pdf/2204.04812)  
3. Vasileva et al. — *Learning Type-Aware Embeddings for Fashion Compatibility*, ECCV 2018 — [arXiv](https://arxiv.org/pdf/1803.09196)  
4. Liu et al. — *Swin Transformer*, ICCV 2021  
5. Liu et al. — *A ConvNet for the 2020s*, CVPR 2022 
