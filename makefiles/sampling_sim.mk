# Makefile for generating the sampling_sim figure (fig2)
# including running simulations etc.

# can provide make -f makefiles/sampling_sim.mk SLIM_PATH="" to use msprime
SLIM_PATH := $(shell which slim)
EQ = =

# Check if SAMPLE_SIZE was provided on command line e.g. via
# make -f sampling_sim.mk SAMPLE_SIZE=100
# The 'origin' function returns how a variable got its value
# 'command line' means it was set via command line
# 'file' means it was set in the Makefile
SAMPLE_SIZE_ORIGIN := $(origin SAMPLE_SIZE)
ifeq ($(SAMPLE_SIZE_ORIGIN),undefined)
    SAMPLE_SIZE = 60000
    $(info Using default SAMPLE_SIZE=$(SAMPLE_SIZE))
else
    N = -n $(SAMPLE_SIZE)
    $(info Using provided SAMPLE_SIZE=$(SAMPLE_SIZE))
endif

# Directories
SCRIPT_DIR = scripts
DATA_DIR = data
FIG_DIR = figures

# Target files for unbalanced simulation
UNBAL_SIM = $(DATA_DIR)/sampling_sim_unbalanced+$(SAMPLE_SIZE).tsz
UNBAL_INF = $(DATA_DIR)/sampling_sim_unbalanced+$(SAMPLE_SIZE).inferred.tsz
UNBAL_DATED = $(DATA_DIR)/sampling_sim_unbalanced+$(SAMPLE_SIZE).dated.tsz

# Target files for balanced simulation
BAL_SIM = $(DATA_DIR)/sampling_sim_balanced+$(SAMPLE_SIZE).tsz
BAL_INF = $(DATA_DIR)/sampling_sim_balanced+$(SAMPLE_SIZE).inferred.tsz
BAL_DATED = $(DATA_DIR)/sampling_sim_balanced+$(SAMPLE_SIZE).dated.tsz

# Final output files
DATA_OUT = $(DATA_DIR)/sampling_sim+$(SAMPLE_SIZE)_data.csv.gz
FIG_OUT = $(FIG_DIR)/sampling_sim+$(SAMPLE_SIZE).pdf

all: $(FIG_OUT)

# Unbalanced simulation pipeline
$(UNBAL_SIM):
	python $(SCRIPT_DIR)/sampling_sim_stdpopsim.py unbalanced --slim_path$(EQ)$(SLIM_PATH) $(N)

$(UNBAL_INF): $(UNBAL_SIM)
	python $(SCRIPT_DIR)/sampling_sim_tsinfer.py unbalanced $(N)

$(UNBAL_DATED): $(UNBAL_INF)
	python $(SCRIPT_DIR)/sampling_sim_tsdate.py unbalanced $(N)

# Balanced simulation pipeline
$(BAL_SIM):
	python $(SCRIPT_DIR)/sampling_sim_stdpopsim.py balanced --slim_path$(EQ)$(SLIM_PATH) $(N)

$(BAL_INF): $(BAL_SIM)
	python $(SCRIPT_DIR)/sampling_sim_tsinfer.py balanced $(N)

$(BAL_DATED): $(BAL_INF)
	python $(SCRIPT_DIR)/sampling_sim_tsdate.py balanced $(N)

# Generate data file from simulation results
$(DATA_OUT): $(UNBAL_SIM) $(UNBAL_DATED) $(BAL_SIM) $(BAL_DATED)
	python $(SCRIPT_DIR)/make_figure_data.py sampling_sim $(N)

# Generate final figure
$(FIG_OUT): $(DATA_OUT) | $(FIG_DIR)
	python $(SCRIPT_DIR)/plot.py sampling_sim $(N)

.PHONY: all clean

clean:
	rm -f $(UNBAL_SIM) $(UNBAL_INF) $(UNBAL_DATED)
	rm -f $(BAL_SIM) $(BAL_INF) $(BAL_DATED)
	rm -f $(DATA_OUT)
	rm -f $(FIG_OUT)


