#!/bin/bash
# fix /r windows: sed -i 's/\r$//' run.sh

# PROGRAM="test_rope.py"
# PROGRAM="test_llama_attention.py"
# PROGRAM="test_llama_decoder.py"
PROGRAM="test_llama_forward.py"
BATCH_SIZE="2"
SEQ_LEN="12"
EMBED_DIM="768"
INTERMEDIATE_DIM="2048"
# NUM_HEADS="2"
NUM_HEADS="12"
VOCAB_SIZE="1024"
NUM_LAYERS="12"

python $PROGRAM -bs $BATCH_SIZE -sl $SEQ_LEN -ed $EMBED_DIM -id $INTERMEDIATE_DIM -nh $NUM_HEADS -vs $VOCAB_SIZE -nl $NUM_LAYERS -v --save