#!/usr/bin/python3 -u
# ==============================================================================
# Data Preparation
# ==============================================================================
import numpy as np
########################################################################################
# We have 3 cases the first is non-overlapping without stateful = False = case_1       #
# second, non-overlapping with stateful = True, case_2, in this case the timesteps     #
# and batchsize need to divide with number with module =0                              #
# third, overlapping with time step =1, case_3  with stateful=true                     #
########################################################################################
############################Case1#######################################################
