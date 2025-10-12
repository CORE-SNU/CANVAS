#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. download videos
gdown --folder https://drive.google.com/drive/folders/16AbCZJeNXqwQVyKIqT2uCbiqB396RZMo -O "$SCRIPT_DIR/videos"

# 2. download datasets
gdown --folder https://drive.google.com/drive/folders/1rSkSCy6fvV357DQmf63mXqskbDzdwhSg -O "$SCRIPT_DIR/datasets"

# 3. download pretrained models
# 3-1. Trajectron++
gdown --folder https://drive.google.com/drive/folders/18A3V0FQo7yXmlSdsAcrwrvq80C_KK9Pq -O "$SCRIPT_DIR/models/trajectron"

