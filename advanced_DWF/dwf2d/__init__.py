"""Joint master + slave directional water-filling (DWF), split by concern.

    channels      optical (Beer-Lambert) and THz channel models
    params        JointParams, JointResult
    solver        joint_dwf
    physical      physical receive chain post-processing and beta calibration
    verification  check_kkt_joint, reference_solution_joint
    plotting      one function per figure
    demos         legacy_demo, physical_demo

build_2D_dwf.py re-exports everything (including private helpers) for backward compatibility;
the theory and notation are in its module docstring.
"""
from .channels import beer_lambert, thz_absorption, thz_alpha_cm, thz_alpha_m, thz_freq, thz_gain
from .params import JointParams, JointResult
from .solver import InfeasibleSlaveBudget, joint_dwf
from .physical import physical_stored_power, PhysicalStorageResult, solve_with_physical_chain
from .verification import check_kkt_joint, reference_solution_joint
from .plotting import *  # noqa: F403  (the plot_* functions)
from .demos import legacy_demo, physical_demo


# claude --resume c6463b24-bd91-4096-8ee0-1d0ea317641b