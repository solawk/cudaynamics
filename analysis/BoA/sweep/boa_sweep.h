#pragma once

#include "../../../computation_struct.h"

// Parameter-sweep BoA stage 2. Clustering is local to each parameter layer;
// tracking then replaces layer-local cluster roots with stable global ids.
void FinalizeBOASweepOpenMP(Computation* data);
cudaError_t FinalizeBOASweepCUDA(Computation* data, numb* cudaMaps, uint64_t variations);
void TrackBOASweepLabels(Computation* data);

