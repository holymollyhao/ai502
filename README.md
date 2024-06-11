# CLIP VSR
I clean up the repository, based on the eval_002.py and data_002.py
## Instalation
- Pillow
- matplotlib
- opencv_python
- torch, torchvision
- tqdm
- pandas
- transformers
- numpy

## Data
See [`data/`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data) folder's readme. Images should be saved under `data/images/`.

## How to use
```
python src/eval.py
```

### Run VP and RA-aug at once
```
. src/run_exp.sh
```

#### Download images
See [`data/`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data) folder's readme. Images should be saved under `data/images/`.


### Citation
If you use the VSR dataset please site the orginal authors:
```bibtex
@article{Liu2022VisualSR,
  title={Visual Spatial Reasoning},
  author={Fangyu Liu and Guy Edward Toh Emerson and Nigel Collier},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.00363}
}
```

### License
This project is licensed under the [Apache-2.0 License](https://github.com/cambridgeltl/visual-spatial-reasoning/blob/master/LICENSE).
