# CLIP VSR
This is the code repository for 2024 Spring KAIST AI502 Deep Learning final project by Nayoung Oh, Taewon Kim, Anh Nguyen Thi Chung.

## Instalation
Recommnend, generating anaconda env with python==3.8
- Pillow
- matplotlib
- opencv_python
- torch, torchvision
- tqdm
- pandas
- transformers
- numpy

Tested with torch==2.0.1 and 2.3.0 / transformers=4.41.x

## Data
See [`data/`](https://github.com/holymollyhao/ai502/tree/master/data) folder's readme.

### Run VP and RA-aug at once
```bash
  sh src/run_exp.sh
```

### Visualize attention
```bash
  sh src/run_visualize.sh
```

### Reference
Some repositories used to develop
- [VSR](https://github.com/cambridgeltl/visual-spatial-reasoning)
- [CLIP VSR](https://github.com/Sohojoe/CLIP_visual-spatial-reasoning)
- [Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability)
