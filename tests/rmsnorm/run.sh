#!/bin/bash

BATCH_SIZE=2
SEQ_LEN=4
EMBED_DIM=8
python test_rmsnorm.py -bs $BATCH_SIZE -sl $SEQ_LEN -ed $EMBED_DIM -v --save