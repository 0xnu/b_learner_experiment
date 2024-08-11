# 401k_verification, the C program simulates and analyses a 401K dataset,
# implementing the B-Learner method to estimate bounds for the Conditional
# Average Treatment Effect (CATE). It generates synthetic data, calculates
# true CATEs, and outputs results for various log-gamma values, including
# average bounds, coverage, and the percentage of negative lower bounds,
# all formatted as CSV for further analysis.
#
# Copyright (c) 2024 Finbarrs Oketunji
# Written by Finbarrs Oketunji <f@finbarrs.eu>
#
# This file is part of 401k_verification.
#
# 401k_verification is an open-source software: you are free to redistribute
# and/or modify it under the terms specified in version 3 of the GNU
# General Public License, as published by the Free Software Foundation.
#
# 401k_verification is is made available with the hope that it will be beneficial,
# but it comes with NO WARRANTY whatsoever. This includes, but is not limited
# to, any implied warranties of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE. For more detailed information, please refer to the
# GNU General Public License.
#
# You should have received a copy of the GNU General Public License
# along with 401k_verification.  If not, visit <http://www.gnu.org/licenses/>.

CC = gcc
CFLAGS = -lm
TARGET = 401k_verification
OUTPUT = 401k_verification_results.csv

all: $(TARGET) ## Compile the target executable

$(TARGET): $(TARGET).c
	$(CC) -o $@ $^ $(CFLAGS)

run: $(TARGET) ## Run the program and output results to CSV
	./$(TARGET) > $(OUTPUT)

clean: ## Remove compiled executable and output file
	rm -f $(TARGET) $(OUTPUT)

help: ## Display Help Message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: all run clean help
.DEFAULT_GOAL := help