#!/bin/bash
source /opt/conda/bin/activate base
data='train'
data='validation'
data='test'
python download_cath.py $data