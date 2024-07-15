# Realistic AI watermark util

Adds and reads watermark on image

## Quick Start
```bash
pip install opencv-python numpy PyWavelets
```

## Run
```bash
python ./embed.py --img x22.png --pwd 3465 --out x22w.png --wm a61d358eaff59724fdf5b3ce06447dc9
python ./unembed.py --img x22w.png --pwd 3465
```

## Documentation
```bash
python ./embed.py -h
python ./unembed.py -h
```

## Based on
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9477625/
