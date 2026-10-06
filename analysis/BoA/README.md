# Basins of Attraction (BoA)

## Purpose

The analysis assigns every initial-condition variation to a basin by clustering
the final values of exactly two selected analysis features. The default feature
pair is `Mean peak` and `Mean interval`.

## Settings

`Basins of Attraction` has six serialized settings:

1. first `AnalysisIndex`;
2. second, distinct `AnalysisIndex`;
3. DBSCAN epsilon in normalized two-dimensional feature space;
4. DBSCAN minimum-points count (the point itself is included).
5. parameter-sweep flag (`0` or `1`);
6. swept parameter index (`-1` when none is selected).

Example for a system `.txt` file:

```text
analysis Basins of Attraction settings 6 7 0.05 4
```

The legacy four-field form remains valid and loads with parameter sweep off.
A saved sweep configuration looks like `6 7 0.05 4 1 0`.

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

## Parameter sweep

Enable `Parameter sweep` in Analysis Settings and select a physical parameter
that has at least two ranged values. The numerical integration and the two
feature analyses keep their existing CUDA/OpenMP parallelism. BoA then runs an
independent DBSCAN for every parameter layer; the CUDA backend processes points
from all layers in parallel while restricting neighbour searches to the same
layer. Its work is `O(L * M^2)` for `L` layers of `M` points instead of applying
one global `O((L*M)^2)` clustering.

After the labels return to the host, clusters in adjacent layers are matched by
a deterministic score combining normalized feature-centroid distance and the
intersection-over-union of their initial-condition cells. A matched cluster
inherits its track ID, and therefore its heatmap colour; unmatched clusters get
new IDs. Splits and merges retain one best matching track and assign new IDs to
the remaining branches.

The BoA plot window repeats the sweep-parameter selector and shows the value of
the displayed layer. Changing the selector is a pending computation setting and
the window asks for `Compute` before using it. The selected sweep dimension is
not available as a heatmap axis; it is selected with the normal ranging control,
so the two axes remain available for the initial-condition plane.

## Display and export

Choose `BoA (Basins of Attraction)` in Graph Builder. Its heatmap uses a
categorical generated palette, so it is not limited to two attractors. The
window's `File -> Export to .csv` command exports the basin-ID grid.

BoA intentionally requires the complete grid and is not available in the
chunked Hi-Res path. For systems with long transients, including Thomas, allow
one or two normal continuous-computation buffers before interpreting or
exporting the map.
