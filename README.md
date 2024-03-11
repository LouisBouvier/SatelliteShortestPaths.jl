# Shortest Paths on satellite images

This repository gathers some code to learn how to compute shortest paths from top left to bottom right corners of satellite images.

1. Before start, the following datasets must be dowloaded:

- Kaggle dataset of satellite images with road masks: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset?resource=download. 
- Zenodo-hosted dataset of terrain satellite images: https://zenodo.org/records/7711810#.ZAm3k-zMKEA.

2. For now, dataset creation is done in Python.

3. Once the dataset is created, run the Julia notebook (Pluto notebook) as follows:

```julia
using Pluto
Pluto.run()
```
A browser window opens, select the `satellite_TP.jl` notebook.