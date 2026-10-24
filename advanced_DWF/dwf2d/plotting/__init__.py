"""Plotting for the joint DWF: each plot_* function saves exactly one PNG.

    style         palette and shared helpers (_new_axis, _finish, ...)
    joint         panels of the joint solution
    physical      solver vs physical receive chain, loss breakdown
    slave_panels  slave-load panels and the slave energy CSV
"""
from .style import BITS_PER_NAT, C_LINK, C_MASTER, C_SLAVE, EDGE_TOL, GRID, INK, INK2
from .joint import (
    plot_cumulative_energy, plot_dwf_allocation, plot_harvested_energy, plot_link_power, plot_master_battery, plot_sharing_energy,
    plot_slave_battery,
    plot_throughput_cumulative, plot_throughput_per_slot, plot_transmit_power,
)
from .physical import (
    energy_loss_breakdown, plot_loss_histogram, plot_physical_slave_battery, plot_physical_stored_power,
)
from .slave_panels import (
    C_MCU, C_MEM, C_MPPT, LOAD_COLORS, LOAD_LABELS, plot_slave_battery_window, plot_slave_cumulative_energy,
    plot_slave_energy_breakdown, plot_slave_load_per_slot, slave_load_components, write_refusal_history_csv,
    write_slave_energy_csv,
)
