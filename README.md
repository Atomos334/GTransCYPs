# GTransCYPs: an improved graph transformer neural network with attention pooling for reliably predicting CYP450 inhibitors

## Introduction
<p align="justify">
State‑of‑the‑art medical studies proved that predicting CYP450 enzyme inhibitors is beneficial in the early stage of drug discovery. However, accurate machine learning-based (ML) in silico methods for predicting CYP450 inhibitors remains challenging. Here, we introduce GTransCYPs, an improved graph neural network (GNN) with a transformer mechanism for predicting CYP450 inhibitors. This model significantly enhances the discrimination between inhibitors and non-inhibitors for five major CYP450 isozymes: 1A2, 2C9, 2C19, 2D6, and 3A4. GTransCYPs learns information patterns from molecular graphs by aggregating node and edge representations using a transformer. The GTransCYPs model utilizes transformer convolution layers to process features, followed by a global attention-pooling technique to synthesize the graph-level information. This information is then fed through successive linear layers for final output generation. Experimental results demonstrate that the GTransCYPs model achieved high performance, outperforming other state-of-the-art methods in CYP450 prediction.</p>

<p align="center">
 <img width="800" alt="image" src="img/GTransCYPs.png">
</p>

## Dependencies

- Python 3.8.0
- PyTorch 2.0.1
- Pytorch-geometric 2.3.1
- Pytorch-lightning 2.0.2
- Deepchem 2.7.1
- RDKit 2023.03.1

## Citation
If you find the code helpful in your resarch or work, please cite our paper: 
```
@article{zonyfar2024gtranscyps,
  title={GTransCYPs: an improved graph transformer neural network with attention pooling for reliably predicting CYP450 inhibitors},
  author={Zonyfar, Candra and Ngnamsie Njimbouom, Soualihou and Mosalla, Sophia and Kim, Jeong-Dong},
  journal={Journal of Cheminformatics},
  volume={16},
  number={1},
  pages={119},
  year={2024},
  doi={10.1186/s13321-024-00915-z},
  url={https://doi.org/10.1186/s13321-024-00915-z}
}
```
