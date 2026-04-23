#!/bin/bash

# PROGRAM="test_rope.py"
PROGRAM="test_llama_attention.py"
BATCH_SIZE="2"
SEQ_LEN="12"
EMBED_DIM="8"
# NUM_HEADS="2"
NUM_HEADS="4"

python $PROGRAM -bs $BATCH_SIZE -sl $SEQ_LEN -ed $EMBED_DIM -nh $NUM_HEADS -v --save