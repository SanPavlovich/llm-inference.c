#!/bin/bash

PROGRAM="test_rope.py"
BATCH_SIZE="2"
SEQ_LEN="12"
EMBED_DIM="8"
NUM_HEADS="2"

python $PROGRAM -bs $BATCH_SIZE -sl $SEQ_LEN -ed $EMBED_DIM -nh $NUM_HEADS -v --save
