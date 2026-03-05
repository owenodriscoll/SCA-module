![Continuous Integration build in GitHub Actions](https://github.com/owenodriscoll/SCA-module/actions/workflows/main.yaml/badge.svg?branch=main)

## Data preparation
1. Download SWB data from [data.4tu.nl](https://data.4tu.nl/private_datasets/wbchBwb0jtmStAHf7U5M5B6xjYBeugZoKB-kgiqfxtI). This is needed for wind-field information.

2. Download SWB lookup tbales from [https://harmony.tudelft.nl](https://harmony.tudelft.nl/?page_id=128).

3. Run the notebooks under `notebooks`

## Environment preparation
Create a new environment and activate

```bash
conda create -n ENV_NAME python==3.12
conda activate ENV_NAME
```

Conda install GDAL (not enabled from pip)
```bash
conda install GDAL
```

Clone environment
```bash
git clone git@github.com:owenodriscoll/SCA-module.git
```

Navigate to installed directory and pip install other requirements
```bash
pip install -e .
```
