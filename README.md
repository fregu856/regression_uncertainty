# regression_uncertainty

Work in progress, code will be uploaded soon!




***

![overview image](regression_uncertainty.png)

Official implementation (PyTorch) of the paper: \
**How Reliable is Your Regression Model's Uncertainty Under Real-World Distribution Shifts?**, 2023 [[arXiv (TODO!)]]() [[project (TODO!)]](). \
[Fredrik K. Gustafsson](http://www.fregu856.com/), [Martin Danelljan](https://martin-danelljan.github.io/), [Thomas B. Schön](http://user.it.uu.se/~thosc112/). \
_TODO!_

If you find this work useful, please consider citing:
```
TODO!
```




***
***

## Datasets:

#### Cells:
- TODO!
- Run Cells/create_datasets.py to generate the train, val and test splits.

#### Cells-Tails:
- TODO!
- Run Cells-Tails/create_datasets.py to generate the train, val and test splits.

#### Cells-Gap:
- TODO!
- Run Cells-Gap/create_datasets.py to generate the train, val and test splits.

#### ChairAngle:
- TODO!
- Run ChairAngle/create_datasets.py to generate the train, val and test splits.

#### ChairAngle-Tails:
- TODO!
- Run ChairAngle-Tails/create_datasets.py to generate the train, val and test splits.

#### ChairAngle-Gap:
- TODO!
- Run ChairAngle-Gap/create_datasets.py to generate the train, val and test splits.

#### AssetWealth:
- $ pip install wilds
- Run AssetWealth/create_datasets.py to download the dataset (13 GB).

#### VentricularVolume:
- TODO!
- Run VentricularVolume/create_datasets.py to generate the train, val and test splits.

#### BrainTumourPixels:
- TODO!
- Run BrainTumourPixels/create_datasets.py to generate the train, val and test splits.

#### SkinLesionPixels:
- TODO!
- Run SkinLesionPixels/create_datasets.py to generate the train, val and test splits.

#### HistologyNucleiPixels:
- Download consep.zip from https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/ and extract.
- Download kumar.zip and tnbc.zip from https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK and extract.
- Create the folder regression_uncertainty/datasets/HistologyNucleiPixels.
- Move the resulting CoNSeP, kumar and tnbc folders to regression_uncertainty/datasets/HistologyNucleiPixels.
- Run HistologyNucleiPixels/create_datasets.py to generate the train, val and test splits.

#### AerialBuildingPixels:
- Download the data from https://project.inria.fr/aerialimagelabeling/ and extract everything.
- Move the resulting AerialImageDataset folder to regression_uncertainty/datasets.
- Run AerialBuildingPixels/create_datasets.py to generate the train, val and test splits.
