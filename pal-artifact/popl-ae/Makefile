# Push-button reproduction of the paper's evaluation. Run inside the artifact environment
# (`nix develop .#artifact` from the repository root), from this directory.
#
#   make setup            download Alea.jl's packages (once, needs the network)
#   make smoke            a quick pass over every experiment (about 5 minutes)
#   make full             the full evaluation, with sizes and timeouts capped (a few hours)
#   make full GPU=0       run PAL/XLA, memo and natlist on GPU 0 (nvidia-smi's numbering)
#   make markov | bayes | grid | natlist    one experiment (MODE=smoke or MODE=full)
#   make report           the figures and tables, rendered with LaTeX: $(RESULTS)/report.pdf
#   make check            answers agree across configurations
#
# Results go to $(RESULTS) (results/<mode> by default).

MODE    ?= full
GPU     ?=
RESULTS ?= results/$(MODE)
PYTHON  ?= python
MEM_GB  ?= 64

export PYTHONPATH := $(abspath ..)$(if $(PYTHONPATH),:$(PYTHONPATH))

GPU_ARG  := $(if $(GPU),--gpu $(GPU))
COMMA    := ,
JAX_ENV  := $(if $(GPU),JAX_PLATFORMS=cuda$(COMMA)cpu CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(GPU),JAX_PLATFORMS=cpu)
upper     = $(shell echo $(1) | tr a-z- A-Z_)
BENCH    := $(PYTHON) bench.py --mem-limit-gb $(MEM_GB)
BENCH_PAL   = $(BENCH) --trials $(PAL_TRIALS) --timeout $(PAL_TIMEOUT)
BENCH_DICE  = $(BENCH) --trials $(DICE_TRIALS) --timeout $(DICE_TIMEOUT)

ifeq ($(MODE),smoke)
PAL_TRIALS   := 1
PAL_TIMEOUT  := 300
DICE_TRIALS  := 1
DICE_TIMEOUT := 10
BAYES_DICE_TIMEOUT := 60
L_CAESAR   := 1,10,100,1000
L_DIAMOND  := 1,100,1000,10000
L_LADDER   := 1,100,1000,10000
L_DICEFIG1 := 1,100,1000,10000
NETWORKS   := cancer,survey,alarm,insurance,hepar2,pigs
L_GRID_XLA    := 2,3,4,5,6,8
L_GRID_NOMARG := 2,3,4,5,6
L_GRID_NUMPY  := 2,3,4,5,6,8
L_GRID_DICE   := 2,3,4,5,6,8
NATLIST_EPOCHS := 100
else
PAL_TRIALS   := 3
PAL_TIMEOUT  := 600
DICE_TRIALS  := 3
DICE_TIMEOUT := 60
BAYES_DICE_TIMEOUT := 600
L_CAESAR   := 1,2,5,10,25,50,100,250,500,1000,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000
L_DIAMOND  := 1,2,5,10,25,50,100,250,500,1000,2000,3000,4000,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,120000,140000,160000,180000,200000
L_LADDER   := 1,2,5,10,25,50,100,250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,15000,20000,25000,30000
L_DICEFIG1 := 1,2,5,10,25,50,100,250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000
NETWORKS   := cancer,survey,alarm,insurance,hepar2,pigs,water,munin,andes,link,diabetes,hailfinder
L_GRID_XLA    := 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28
L_GRID_NOMARG := 2,3,4,5,6,7,8,9,10,11,12
L_GRID_NUMPY  := 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25
L_GRID_DICE   := 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
NATLIST_EPOCHS := 1000
endif

MARKOV := caesar diamond ladder dicefig1
GRIDS  := grid-link grid-router

.PHONY: setup smoke full markov bayes grid natlist report check clean $(MARKOV) $(GRIDS)

# Downloads and precompiles Alea.jl's Julia packages (into ~/.cache/pal-dev/alea-jl), so
# that no later step needs the network.
setup:
	alea-jl -e 'using Dice; println("Alea.jl is ready.")'

smoke:
	$(MAKE) -k MODE=smoke markov bayes grid natlist report check

full:
	$(MAKE) -k MODE=full markov bayes grid natlist report check

# --- Markov models: Dice, PAL/XLA (cold and warm), memo on dicefig1, and the
# no-marginalization ablation on caesar.
markov: $(MARKOV)

caesar diamond ladder dicefig1:
	$(BENCH_PAL) --config pal-xla $(GPU_ARG) --output-dir $(RESULTS)/$@ $@ -L $(L_$(call upper,$@))
	$(BENCH_DICE) --config dice --output-dir $(RESULTS)/$@ $@ -L $(L_$(call upper,$@))
	$(if $(filter caesar,$@),$(BENCH_PAL) --config pal-xla-nomarg $(GPU_ARG) --output-dir $(RESULTS)/$@ $@ -L $(L_CAESAR))
	$(if $(filter dicefig1,$@),$(BENCH_DICE) --config memo $(GPU_ARG) --output-dir $(RESULTS)/$@ $@ -L $(L_DICEFIG1))

# --- Bayesian networks: PAL/NumPy with and without eager marginalization, and Dice.
bayes:
	$(BENCH_PAL) --config pal-numpy --output-dir $(RESULTS)/bayes bayes --networks $(NETWORKS)
	$(BENCH_PAL) --config pal-numpy-nomarg --output-dir $(RESULTS)/bayes bayes --networks $(NETWORKS)
	$(BENCH) --trials $(DICE_TRIALS) --timeout $(BAYES_DICE_TIMEOUT) --config dice \
	    --output-dir $(RESULTS)/bayes bayes --networks $(NETWORKS)

# --- Network reliability: Dice, PAL/XLA with and without eager marginalization, PAL/NumPy.
grid: $(GRIDS)

grid-link grid-router:
	$(BENCH_PAL) --config pal-xla $(GPU_ARG) --output-dir $(RESULTS)/$@ $@ -L $(L_GRID_XLA)
	$(BENCH_PAL) --config pal-xla-nomarg $(GPU_ARG) --output-dir $(RESULTS)/$@ $@ -L $(L_GRID_NOMARG)
	$(BENCH_PAL) --config pal-numpy --output-dir $(RESULTS)/$@ $@ -L $(L_GRID_NUMPY)
	$(BENCH_DICE) --config dice --output-dir $(RESULTS)/$@ $@ -L $(L_GRID_DICE)

# --- Tuning a random list generator: PAL (Adam) and Dice (Alea.jl), N = 100.
natlist:
	mkdir -p $(RESULTS)/natlist
	$(JAX_ENV) $(PYTHON) natlist/natlist.py --n 100 --epochs $(NATLIST_EPOCHS) \
	    --output-csv $(RESULTS)/natlist/pal_loss.csv | tee $(RESULTS)/natlist/pal.log
	alea-jl natlist/natlist.jl --n 100 --epochs $(NATLIST_EPOCHS) \
	    --output-csv $(abspath $(RESULTS))/natlist/dice_loss.csv | tee $(RESULTS)/natlist/dice.log

report:
	$(PYTHON) report/make_report.py $(RESULTS)
	cd $(RESULTS)/report && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex > latexmk.log
	cp $(RESULTS)/report/report.pdf $(RESULTS)/report.pdf

check:
	$(PYTHON) report/check.py $(RESULTS)

clean:
	rm -rf results/smoke results/full
