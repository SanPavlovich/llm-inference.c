CC = gcc
# CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm
TARGET = test_forward
SRCS = test_forward.c utils.c ops.c
# ARGS = 2 12 8 4
ARGS = --model-path /root/llm-inference/data/forward/bs_2_sl_12_ed_768_nh_12

compile:
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

compile_sanitize:
	$(CC) -fsanitize=address -g $(SRCS) -o $(TARGET) $(LDFLAGS)

run:
	./$(TARGET) $(ARGS)

clear:
	rm -f $(TARGET)