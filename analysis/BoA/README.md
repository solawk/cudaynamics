# Basins of Attraction (BoA)

## Purpose

The analysis assigns every initial-condition variation to a basin by clustering
the final values of exactly two selected analysis features. The default feature
pair is `Mean peak` and `Mean interval`.

## Settings

`Basins of Attraction` has four serialized settings:

1. first `AnalysisIndex`;
2. second, distinct `AnalysisIndex`;
3. DBSCAN epsilon in normalized two-dimensional feature space;
4. DBSCAN minimum-points count (the point itself is included).

Example for a system `.txt` file:

```text
analysis Basins of Attraction settings 6 7 0.05 4
```

The UI always displays two selectors. Selecting the same feature twice is
disabled, and imported settings with missing, extra, invalid, or duplicate
features are rejected. Enabling BoA automatically enables the two source
analysis ports for the computation without changing their persistent UI
checkboxes.

## Two-stage algorithm

1. CUDAynamics integrates every trajectory and computes the two configured
   feature maps using the existing analysis functions.
2. Once every point in the full initial-condition grid is available, BoA
   normalizes both features by their finite min/max ranges and runs exact
   two-dimensional DBSCAN. The result is written to the `IND_BOA` map port.

In CUDA mode the DBSCAN neighbourhood scan, core-point detection, connected
component label propagation, and border assignment run on the GPU. Only the two
feature arrays are copied to the host to obtain normalization ranges; labels
stay on the device until the normal CUDAynamics map transfer. The implementation
uses O(N) auxiliary memory and O(N^2) distance work, avoiding an O(N^2)
adjacency matrix.

In OpenMP mode the same stages and tie-breaking rules run in parallel on the
CPU. Both backends then compact positive component roots to deterministic basin
IDs `1..K`, ordered by their feature-space centroids. Noise is `-1`; a point
with a non-finite feature is `-2`.

## Display and export

Choose `BoA (Basins of Attraction)` in Graph Builder. Its heatmap uses a
categorical generated palette, so it is not limited to two attractors. The
window's `File -> Export to .csv` command exports the basin-ID grid.

BoA intentionally requires the complete grid and is not available in the
chunked Hi-Res path. For systems with long transients, including Thomas, allow
one or two normal continuous-computation buffers before interpreting or
exporting the map.
