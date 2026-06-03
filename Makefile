# Stride -- SIMD-accelerated numerical optimisers
#
# Targets:
#   make              build/libstride.a
#   make test         gradient + solver tests
#   make check-saddle bit-exact comparison against ../Saddle's C Adam
#   make clean

CC       ?= gcc
CFLAGS   ?= -O2 -g -Wall -Wextra -std=c11
CPPFLAGS  = -Iinclude
LDLIBS    = -lm

NASM     ?= nasm
NASMFLAGS ?= -f elf64 -g -F dwarf

BUILD := build

LIB_SRCS := src/cpu/cpu.c \
            src/kernels/kernels.c \
            src/blas/gemv.c \
            src/objectives/objectives.c \
            src/objectives/logreg.c \
            src/solvers/solvers.c
LIB_OBJS := $(LIB_SRCS:%.c=$(BUILD)/%.o)

# hand written asm kernels, only on x86_64
ASM_SRCS := src/kernels/x86/adam_avx2.asm \
            src/kernels/x86/blas1_avx2.asm
ASM_OBJS := $(ASM_SRCS:%.asm=$(BUILD)/%.o)
LIB_OBJS += $(ASM_OBJS)

LIB := $(BUILD)/libstride.a

TEST_SRCS := tests/test_gradients.c tests/test_solvers.c tests/test_logreg.c
TEST_BINS := $(TEST_SRCS:tests/%.c=$(BUILD)/%)

all: $(LIB)

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD)/%.o: %.asm
	@mkdir -p $(dir $@)
	$(NASM) $(NASMFLAGS) $< -o $@

$(LIB): $(LIB_OBJS)
	ar rcs $@ $^

$(BUILD)/%: tests/%.c $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

test: $(TEST_BINS) $(BUILD)/checkasm
	@for t in $(TEST_BINS); do echo "== $$t"; $$t || exit 1; done
	@echo "== build/checkasm --selftest"
	@$(BUILD)/checkasm --selftest

# checkasm harness, correctness checks plus kernel benchmarks
CHECKASM_SRCS := tests/checkasm/checkasm.c tests/checkasm/check_kernels.c \
                 tests/checkasm/check_blas.c
CHECKASM_HDRS := tests/checkasm/checkasm.h tests/checkasm/check_kernels_tmpl.h \
                 tests/checkasm/check_blas_tmpl.h

$(BUILD)/checkasm: $(CHECKASM_SRCS) $(CHECKASM_HDRS) $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(CHECKASM_SRCS) $(LIB) $(LDLIBS) -o $@

checkasm: $(BUILD)/checkasm
	$(BUILD)/checkasm --selftest

bench: $(BUILD)/checkasm
	$(BUILD)/checkasm --bench

# Cross-check against Saddle's C Adam (needs the sibling Saddle checkout).
SADDLE_CSRC ?= ../Saddle/backend/csrc

$(BUILD)/test_saddle: tests/test_saddle.c $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(SADDLE_CSRC) $< \
	    $(SADDLE_CSRC)/adam.c $(SADDLE_CSRC)/surfaces.c $(LIB) $(LDLIBS) -o $@

check-saddle: $(BUILD)/test_saddle
	$(BUILD)/test_saddle

clean:
	rm -rf $(BUILD)

# Hand-maintained header dependencies; this stays manageable until the M1
# harness, at which point -MMD takes over if it gets unwieldy.
$(LIB_SRCS:%.c=$(BUILD)/%.o): include/stride/cpu.h include/stride/kernels.h \
    include/stride/objective.h include/stride/solver.h
$(BUILD)/src/kernels/kernels.o: src/kernels/kernels_tmpl.h src/kernels/x86/kernels_x86.h
$(BUILD)/src/blas/gemv.o: src/blas/gemv_tmpl.h
$(BUILD)/src/objectives/objectives.o: src/objectives/objectives_tmpl.h
$(BUILD)/src/objectives/logreg.o: src/objectives/logreg_tmpl.h
$(BUILD)/src/solvers/solvers.o: src/solvers/solvers_tmpl.h
$(TEST_BINS) $(BUILD)/test_saddle: tests/test_util.h include/stride/stride.h \
    include/stride/kernels.h include/stride/objective.h include/stride/solver.h

.PHONY: all test check-saddle checkasm bench clean
