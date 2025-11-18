import pickle
from typing import Any


def print_readings(readings):
    # Accepts a 3xN array-like: [deg; deg/s; mA]
    q_deg, qd_dps, I_mA = readings
    q_str = ",".join(f"{x:5.1f}" for x in q_deg)
    qd_str = ",".join(f"{x:5.1f}" for x in qd_dps)
    I_str = ",".join(f"{x:5.0f}" for x in I_mA)
    print(f"q(deg): [{q_str}] | qdot(deg/s): [{qd_str}] | I(mA): [{I_str}]")


def pretty_print(obj: Any) -> None:
    """
    Prints objects similar to Jupyter's display functionality.
    Falls back to different string representations based on availability.
    """
    try:
        # Try __repr__ first (preferred detailed representation)
        if hasattr(obj, "__repr__"):
            print(obj.__repr__())
        # Fall back to __str__ if __repr__ isn't available
        elif hasattr(obj, "__str__"):
            print(obj.__str__())
        else:
            # Last resort if neither method is available
            print(obj)
    except Exception as e:
        print(f"Error displaying object: {e}")


def save_to_pickle(data: dict, filename: str):
    """Save run data to a pickle file."""
    print(f"Data saved to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_from_pickle(filename: str):
    """Load saved run data."""
    print(f"Data loaded from {filename}")
    with open(filename, "rb") as f:
        return pickle.load(f)
