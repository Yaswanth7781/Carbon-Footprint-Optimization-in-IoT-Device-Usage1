# ────────────────────────────────────────────────────────────
#  Carbon Footprint Optimization — Utility Helpers
# ────────────────────────────────────────────────────────────

from typing import Optional, List

# Average CO₂ emission factor (kg CO₂ per kWh)
EMISSION_FACTOR = 0.82

# Peak-hour range (6 PM – 11 PM)
PEAK_HOURS = range(18, 23)

# Thresholds
HIGH_DURATION_THRESHOLD = 4        # hours
HIGH_ENERGY_THRESHOLD = 5.0        # kWh


def calculate_co2(energy_kwh: float) -> float:
    """Convert energy consumption (kWh) to estimated CO₂ (kg)."""
    return round((energy_kwh / 1000) * EMISSION_FACTOR, 6)


def get_optimization_suggestions(
    duration: float,
    hour: int,
    energy: Optional[float] = None,
) -> List[str]:
    """Return a list of actionable optimisation suggestions."""

    suggestions: List[str] = []

    if duration > HIGH_DURATION_THRESHOLD:
        suggestions.append(
            f"Reduce device usage duration (currently {duration:.1f} h, "
            f"threshold is {HIGH_DURATION_THRESHOLD} h)."
        )

    if hour in PEAK_HOURS:
        suggestions.append(
            "Shift usage to off-peak hours (before 6 PM or after 11 PM) "
            "to lower grid carbon intensity."
        )

    if energy is not None and energy > HIGH_ENERGY_THRESHOLD:
        suggestions.append(
            f"Energy consumption is high ({energy:.2f} kWh). "
            "Consider using an energy-efficient device or reducing load."
        )

    if not suggestions:
        suggestions.append("Usage looks efficient — no changes recommended. 🌿")

    return suggestions