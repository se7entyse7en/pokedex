#!/bin/bash
set -e

mkdir -p data/images
KAGGLE_CONFIG_DIR=. kaggle datasets download vishalsubbiah/pokemon-images-and-types --unzip -p data
unzip data/images.zip -d data && rm data/images.zip
