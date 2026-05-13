import numpy as np


KBAR_TO_PA = 1e8
M_PER_KM = 1000.0
DEFAULT_G = 9.81


def kbar_to_km(pressure_kbar, density_kg_m3=2700.0, g=DEFAULT_G):
    """
    Convert lithostatic pressure from kbar to depth in km for a single-density crust.

    Parameters
    ----------
    pressure_kbar : float or array-like
        Pressure in kbar.
    density_kg_m3 : float, default 2700.0
        Crustal density in kg/m^3.
    g : float, default 9.81
        Gravitational acceleration in m/s^2.

    Returns
    -------
    float or np.ndarray
        Depth in km. Returns a float if input is scalar; otherwise returns an array.

    Notes
    -----
    Formula:
        pressure = density * g * depth

    Therefore:
        depth_km = pressure_kbar * 1e8 / (density_kg_m3 * g) / 1000
    """
    pressure_arr = np.asarray(pressure_kbar, dtype=float)

    if np.any(pressure_arr < 0):
        raise ValueError("pressure_kbar must be non-negative.")

    if density_kg_m3 <= 0:
        raise ValueError("density_kg_m3 must be positive.")

    depth_km = pressure_arr * KBAR_TO_PA / (density_kg_m3 * g) / M_PER_KM

    if np.isscalar(pressure_kbar):
        return float(depth_km)

    return depth_km


def kbar_to_km_multilayer(
    pressure_kbar,
    densities_kg_m3,
    layer_boundaries_km,
    g=DEFAULT_G,
):
    """
    Convert lithostatic pressure from kbar to depth in km for a layered crust.

    Parameters
    ----------
    pressure_kbar : float or array-like
        Pressure in kbar.

    densities_kg_m3 : array-like
        Density of each layer in kg/m^3.

        Example:
            densities_kg_m3 = [2242, 2900]

        means:
            layer 1 density = 2242 kg/m^3
            layer 2 density = 2900 kg/m^3

    layer_boundaries_km : array-like
        Depths of layer boundaries in km.

        Length must be len(densities_kg_m3) - 1.

        Example:
            densities_kg_m3 = [2242, 2900]
            layer_boundaries_km = [10]

        means:
            0–10 km: 2242 kg/m^3
            >10 km: 2900 kg/m^3

        Example:
            densities_kg_m3 = [2700, 3000, 3300]
            layer_boundaries_km = [10, 20]

        means:
            0–10 km: 2700 kg/m^3
            10–20 km: 3000 kg/m^3
            >20 km: 3300 kg/m^3

    g : float, default 9.81
        Gravitational acceleration in m/s^2.

    Returns
    -------
    float or np.ndarray
        Depth in km. Returns a float if input is scalar; otherwise returns an array.

    Notes
    -----
    This function assumes the last density layer extends indefinitely downward.
    """
    pressure_arr = np.asarray(pressure_kbar, dtype=float)
    scalar_input = np.isscalar(pressure_kbar)

    densities = np.asarray(densities_kg_m3, dtype=float)
    boundaries = np.asarray(layer_boundaries_km, dtype=float)

    if np.any(pressure_arr < 0):
        raise ValueError("pressure_kbar must be non-negative.")

    if np.any(densities <= 0):
        raise ValueError("All densities_kg_m3 values must be positive.")

    if np.any(boundaries <= 0):
        raise ValueError("All layer_boundaries_km values must be positive.")

    if len(boundaries) != len(densities) - 1:
        raise ValueError(
            "layer_boundaries_km must have length len(densities_kg_m3) - 1."
        )

    if len(boundaries) > 1 and np.any(np.diff(boundaries) <= 0):
        raise ValueError("layer_boundaries_km must be strictly increasing.")

    # Convert finite layer boundaries into layer thicknesses.
    # Example:
    # boundaries = [10, 20]
    # layer_tops = [0, 10]
    # finite_thicknesses = [10, 10]
    layer_tops_km = np.concatenate(([0.0], boundaries[:-1]))
    finite_thicknesses_km = boundaries - layer_tops_km

    # Pressure at each finite boundary in kbar.
    # Only finite layers are used here; the last layer extends downward indefinitely.
    finite_layer_pressures_kbar = (
        densities[:-1] * g * finite_thicknesses_km * M_PER_KM / KBAR_TO_PA
    )

    boundary_pressures_kbar = np.cumsum(finite_layer_pressures_kbar)

    # Find which layer each pressure falls into.
    # layer_index = 0 means first layer.
    # layer_index = len(densities) - 1 means deepest layer.
    layer_indices = np.searchsorted(
        boundary_pressures_kbar,
        pressure_arr,
        side="left",
    )

    depth_km = np.empty_like(pressure_arr, dtype=float)

    for i, pressure in np.ndenumerate(pressure_arr):
        layer_index = layer_indices[i]

        if layer_index == 0:
            previous_depth_km = 0.0
            previous_pressure_kbar = 0.0
        else:
            previous_depth_km = boundaries[layer_index - 1]
            previous_pressure_kbar = boundary_pressures_kbar[layer_index - 1]

        remaining_pressure_kbar = pressure - previous_pressure_kbar
        remaining_depth_km = (
            remaining_pressure_kbar * KBAR_TO_PA
            / (densities[layer_index] * g)
            / M_PER_KM
        )

        depth_km[i] = previous_depth_km + remaining_depth_km

    if scalar_input:
        return float(depth_km)

    return depth_km

def km_to_kbar(depth_km, density_kg_m3=2700.0, g=DEFAULT_G):
    """
    Convert depth in km to lithostatic pressure in kbar for a single-density crust.

    Parameters
    ----------
    depth_km : float or array-like
        Depth in km.
    density_kg_m3 : float, default 2700.0
        Crustal density in kg/m^3.
    g : float, default 9.81
        Gravitational acceleration in m/s^2.

    Returns
    -------
    float or np.ndarray
        Pressure in kbar. Returns a float if input is scalar; otherwise returns an array.

    Notes
    -----
    Formula:
        pressure = density * g * depth

    Therefore:
        pressure_kbar = density_kg_m3 * g * depth_km * 1000 / 1e8
    """
    depth_arr = np.asarray(depth_km, dtype=float)

    if np.any(depth_arr < 0):
        raise ValueError("depth_km must be non-negative.")

    if density_kg_m3 <= 0:
        raise ValueError("density_kg_m3 must be positive.")

    pressure_kbar = density_kg_m3 * g * depth_arr * M_PER_KM / KBAR_TO_PA

    if np.isscalar(depth_km):
        return float(pressure_kbar)

    return pressure_kbar

def km_to_kbar_multilayer(
    depth_km,
    densities_kg_m3,
    layer_boundaries_km,
    g=DEFAULT_G,
):
    """
    Convert depth in km to lithostatic pressure in kbar for a layered crust.

    Parameters
    ----------
    depth_km : float or array-like
        Depth in km.

    densities_kg_m3 : array-like
        Density of each layer in kg/m^3.

    layer_boundaries_km : array-like
        Depths of layer boundaries in km.

        Length must be len(densities_kg_m3) - 1.

    g : float, default 9.81
        Gravitational acceleration in m/s^2.

    Returns
    -------
    float or np.ndarray
        Pressure in kbar. Returns a float if input is scalar; otherwise returns an array.

    Notes
    -----
    This function assumes the last density layer extends indefinitely downward.
    """
    depth_arr = np.asarray(depth_km, dtype=float)
    scalar_input = np.isscalar(depth_km)

    densities = np.asarray(densities_kg_m3, dtype=float)
    boundaries = np.asarray(layer_boundaries_km, dtype=float)

    if np.any(depth_arr < 0):
        raise ValueError("depth_km must be non-negative.")

    if np.any(densities <= 0):
        raise ValueError("All densities_kg_m3 values must be positive.")

    if np.any(boundaries <= 0):
        raise ValueError("All layer_boundaries_km values must be positive.")

    if len(boundaries) != len(densities) - 1:
        raise ValueError(
            "layer_boundaries_km must have length len(densities_kg_m3) - 1."
        )

    if len(boundaries) > 1 and np.any(np.diff(boundaries) <= 0):
        raise ValueError("layer_boundaries_km must be strictly increasing.")

    layer_tops_km = np.concatenate(([0.0], boundaries[:-1]))
    finite_thicknesses_km = boundaries - layer_tops_km

    finite_layer_pressures_kbar = (
        densities[:-1] * g * finite_thicknesses_km * M_PER_KM / KBAR_TO_PA
    )

    boundary_pressures_kbar = np.cumsum(finite_layer_pressures_kbar)

    layer_indices = np.searchsorted(boundaries, depth_arr, side="left")
    pressure_kbar = np.empty_like(depth_arr, dtype=float)

    for i, depth in np.ndenumerate(depth_arr):
        layer_index = layer_indices[i]

        if layer_index == 0:
            previous_depth_km = 0.0
            previous_pressure_kbar = 0.0
        else:
            previous_depth_km = boundaries[layer_index - 1]
            previous_pressure_kbar = boundary_pressures_kbar[layer_index - 1]

        remaining_depth_km = depth - previous_depth_km
        remaining_pressure_kbar = (
            densities[layer_index] * g * remaining_depth_km * M_PER_KM / KBAR_TO_PA
        )

        pressure_kbar[i] = previous_pressure_kbar + remaining_pressure_kbar

    if scalar_input:
        return float(pressure_kbar)

    return pressure_kbar

if __name__ == "__main__":
    pressures_kbar = [1, 2, 3, 4, 5, 6]

    depths_km = kbar_to_km_multilayer(
        pressure_kbar=pressures_kbar,
        densities_kg_m3=[2242, 2900],
        layer_boundaries_km=[10],
    )

    print(depths_km)
