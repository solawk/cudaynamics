#include "boa.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

#include "../../index2port.h"

namespace
{
    constexpr int BOA_INVALID = -2;
    constexpr int BOA_NOISE = -1;

    struct FeatureRanges
    {
        numb minimum[2];
        numb inverseRange[2];
    };

    bool getOffsets(const Computation* data, unsigned int offsets[3])
    {
        const BOA_Settings& settings = data->marshal.kernel.analyses.BOA;
        const Port* f0 = index2port(const_cast<AnalysesSettings&>(data->marshal.kernel.analyses), settings.features[0]);
        const Port* f1 = index2port(const_cast<AnalysesSettings&>(data->marshal.kernel.analyses), settings.features[1]);
        if (!f0 || !f1 || !settings.basinId.used || !f0->used || !f1->used) return false;
        offsets[0] = f0->offset;
        offsets[1] = f1->offset;
        offsets[2] = settings.basinId.offset;
        return true;
    }

    bool calculateRanges(const numb* feature0, const numb* feature1, uint64_t count, FeatureRanges& ranges)
    {
        ranges.minimum[0] = ranges.minimum[1] = std::numeric_limits<numb>::infinity();
        numb maximum[2] = { -std::numeric_limits<numb>::infinity(), -std::numeric_limits<numb>::infinity() };
        bool found = false;
        for (uint64_t i = 0; i < count; ++i)
        {
            if (!std::isfinite((double)feature0[i]) || !std::isfinite((double)feature1[i])) continue;
            found = true;
            ranges.minimum[0] = std::min(ranges.minimum[0], feature0[i]);
            ranges.minimum[1] = std::min(ranges.minimum[1], feature1[i]);
            maximum[0] = std::max(maximum[0], feature0[i]);
            maximum[1] = std::max(maximum[1], feature1[i]);
        }
        if (!found) return false;
        for (int d = 0; d < 2; ++d)
        {
            const numb span = maximum[d] - ranges.minimum[d];
            ranges.inverseRange[d] = span > (numb)0 ? (numb)1 / span : (numb)0;
        }
        return true;
    }

    inline bool cpuNeighbor(const numb* f0, const numb* f1, uint64_t a, uint64_t b,
        const FeatureRanges& ranges, numb epsilonSquared)
    {
        const numb ax = f0[a], ay = f1[a], bx = f0[b], by = f1[b];
        if (!std::isfinite((double)ax) || !std::isfinite((double)ay) ||
            !std::isfinite((double)bx) || !std::isfinite((double)by)) return false;
        const numb dx = (ax - bx) * ranges.inverseRange[0];
        const numb dy = (ay - by) * ranges.inverseRange[1];
        return dx * dx + dy * dy <= epsilonSquared;
    }

    __device__ bool gpuFinite(numb value)
    {
        return isfinite(value) != 0;
    }

    __device__ bool gpuNeighbor(const numb* f0, const numb* f1, uint64_t a, uint64_t b,
        numb min0, numb min1, numb inv0, numb inv1, numb epsilonSquared)
    {
        const numb ax = f0[a], ay = f1[a], bx = f0[b], by = f1[b];
        if (!gpuFinite(ax) || !gpuFinite(ay) || !gpuFinite(bx) || !gpuFinite(by)) return false;
        const numb dx = ((ax - min0) * inv0) - ((bx - min0) * inv0);
        const numb dy = ((ay - min1) * inv1) - ((by - min1) * inv1);
        return dx * dx + dy * dy <= epsilonSquared;
    }

    __global__ void classifyCoreKernel(const numb* f0, const numb* f1, uint64_t count,
        numb min0, numb min1, numb inv0, numb inv1, numb epsilonSquared, int minimumPoints,
        unsigned char* valid, unsigned char* core, int* labels)
    {
        const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        valid[i] = gpuFinite(f0[i]) && gpuFinite(f1[i]);
        if (!valid[i]) { core[i] = 0; labels[i] = BOA_INVALID; return; }
        int neighbors = 0;
        for (uint64_t j = 0; j < count && neighbors < minimumPoints; ++j)
            if (gpuNeighbor(f0, f1, i, j, min0, min1, inv0, inv1, epsilonSquared)) ++neighbors;
        core[i] = neighbors >= minimumPoints;
        labels[i] = core[i] ? (int)i + 1 : BOA_NOISE;
    }

    __global__ void propagateLabelsKernel(const numb* f0, const numb* f1, uint64_t count,
        numb min0, numb min1, numb inv0, numb inv1, numb epsilonSquared,
        const unsigned char* core, const int* input, int* output, int* changed)
    {
        const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        if (!core[i]) { output[i] = input[i]; return; }
        int best = input[i];
        for (uint64_t j = 0; j < count; ++j)
            if (core[j] && input[j] > 0 && input[j] < best &&
                gpuNeighbor(f0, f1, i, j, min0, min1, inv0, inv1, epsilonSquared))
                best = input[j];
        output[i] = best;
        if (best != input[i]) atomicExch(changed, 1);
    }

    __global__ void assignBorderKernel(const numb* f0, const numb* f1, uint64_t count,
        numb min0, numb min1, numb inv0, numb inv1, numb epsilonSquared,
        const unsigned char* valid, const unsigned char* core, const int* labels, numb* output)
    {
        const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        if (!valid[i]) { output[i] = (numb)BOA_INVALID; return; }
        if (core[i]) { output[i] = (numb)labels[i]; return; }
        int best = INT_MAX;
        for (uint64_t j = 0; j < count; ++j)
            if (core[j] && labels[j] > 0 && labels[j] < best &&
                gpuNeighbor(f0, f1, i, j, min0, min1, inv0, inv1, epsilonSquared))
                best = labels[j];
        output[i] = best == INT_MAX ? (numb)BOA_NOISE : (numb)best;
    }
}

__host__ __device__ void BOA(Computation* data, uint64_t variation)
{
    const BOA_Settings settings = CUDA_kernel.analyses.BOA;
    CUDA_marshal.maps[indexPosition(settings.basinId.offset, 0)] = (numb)BOA_NOISE;
}

void FinalizeBOAOpenMP(Computation* data)
{
    if (!data || data->isHires || !data->marshal.kernel.analyses.BOA.toCompute) return;
    unsigned int offsets[3];
    if (!getOffsets(data, offsets)) return;
    const uint64_t count = data->marshal.totalVariations;
    numb* maps = data->marshal.maps;
    const numb* f0 = maps + (uint64_t)offsets[0] * count;
    const numb* f1 = maps + (uint64_t)offsets[1] * count;
    numb* output = maps + (uint64_t)offsets[2] * count;
    FeatureRanges ranges;
    if (!calculateRanges(f0, f1, count, ranges))
    {
        std::fill(output, output + count, (numb)BOA_INVALID);
        return;
    }

    const BOA_Settings& settings = data->marshal.kernel.analyses.BOA;
    const numb epsilonSquared = settings.epsilon * settings.epsilon;
    std::vector<unsigned char> valid(count), core(count);
    std::vector<int> labels(count), next(count);

#pragma omp parallel for
    for (long long i = 0; i < (long long)count; ++i)
    {
        valid[i] = std::isfinite((double)f0[i]) && std::isfinite((double)f1[i]);
        if (!valid[i]) { core[i] = 0; labels[i] = BOA_INVALID; continue; }
        int neighbors = 0;
        for (uint64_t j = 0; j < count && neighbors < settings.minimumPoints; ++j)
            if (cpuNeighbor(f0, f1, i, j, ranges, epsilonSquared)) ++neighbors;
        core[i] = neighbors >= settings.minimumPoints;
        labels[i] = core[i] ? (int)i + 1 : BOA_NOISE;
    }

    for (uint64_t iteration = 0; iteration < count; ++iteration)
    {
        int changed = 0;
#pragma omp parallel for reduction(|:changed)
        for (long long i = 0; i < (long long)count; ++i)
        {
            if (!core[i]) { next[i] = labels[i]; continue; }
            int best = labels[i];
            for (uint64_t j = 0; j < count; ++j)
                if (core[j] && labels[j] > 0 && labels[j] < best &&
                    cpuNeighbor(f0, f1, i, j, ranges, epsilonSquared)) best = labels[j];
            next[i] = best;
            changed |= best != labels[i];
        }
        labels.swap(next);
        if (!changed) break;
    }

#pragma omp parallel for
    for (long long i = 0; i < (long long)count; ++i)
    {
        if (!valid[i]) { output[i] = (numb)BOA_INVALID; continue; }
        if (core[i]) { output[i] = (numb)labels[i]; continue; }
        int best = std::numeric_limits<int>::max();
        for (uint64_t j = 0; j < count; ++j)
            if (core[j] && labels[j] > 0 && labels[j] < best &&
                cpuNeighbor(f0, f1, i, j, ranges, epsilonSquared)) best = labels[j];
        output[i] = best == std::numeric_limits<int>::max() ? (numb)BOA_NOISE : (numb)best;
    }
}

cudaError_t FinalizeBOACUDA(Computation* data, numb* cudaMaps, uint64_t count)
{
    if (!data || data->isHires || !data->marshal.kernel.analyses.BOA.toCompute) return cudaSuccess;
    unsigned int offsets[3];
    if (!getOffsets(data, offsets)) return cudaErrorInvalidValue;

    const numb* deviceF0 = cudaMaps + (uint64_t)offsets[0] * count;
    const numb* deviceF1 = cudaMaps + (uint64_t)offsets[1] * count;
    numb* deviceOutput = cudaMaps + (uint64_t)offsets[2] * count;
    std::vector<numb> hostF0(count), hostF1(count);
    cudaError_t status = cudaMemcpy(hostF0.data(), deviceF0, count * sizeof(numb), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return status;
    status = cudaMemcpy(hostF1.data(), deviceF1, count * sizeof(numb), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return status;
    FeatureRanges ranges;
    if (!calculateRanges(hostF0.data(), hostF1.data(), count, ranges))
    {
        std::vector<numb> invalid(count, (numb)BOA_INVALID);
        return cudaMemcpy(deviceOutput, invalid.data(), count * sizeof(numb), cudaMemcpyHostToDevice);
    }

    unsigned char* valid = nullptr;
    unsigned char* core = nullptr;
    int* labels = nullptr;
    int* next = nullptr;
    int* changed = nullptr;
    if ((status = cudaMalloc(&valid, count)) != cudaSuccess) goto Cleanup;
    if ((status = cudaMalloc(&core, count)) != cudaSuccess) goto Cleanup;
    if ((status = cudaMalloc(&labels, count * sizeof(int))) != cudaSuccess) goto Cleanup;
    if ((status = cudaMalloc(&next, count * sizeof(int))) != cudaSuccess) goto Cleanup;
    if ((status = cudaMalloc(&changed, sizeof(int))) != cudaSuccess) goto Cleanup;

    {
        const int threads = 256;
        const int blocks = (int)((count + threads - 1) / threads);
        const BOA_Settings& settings = data->marshal.kernel.analyses.BOA;
        const numb epsilonSquared = settings.epsilon * settings.epsilon;
        classifyCoreKernel<<<blocks, threads>>>(deviceF0, deviceF1, count,
            ranges.minimum[0], ranges.minimum[1], ranges.inverseRange[0], ranges.inverseRange[1],
            epsilonSquared, settings.minimumPoints, valid, core, labels);
        if ((status = cudaGetLastError()) != cudaSuccess) goto Cleanup;

        for (uint64_t iteration = 0; iteration < count; ++iteration)
        {
            int hostChanged = 0;
            if ((status = cudaMemset(changed, 0, sizeof(int))) != cudaSuccess) goto Cleanup;
            propagateLabelsKernel<<<blocks, threads>>>(deviceF0, deviceF1, count,
                ranges.minimum[0], ranges.minimum[1], ranges.inverseRange[0], ranges.inverseRange[1],
                epsilonSquared, core, labels, next, changed);
            if ((status = cudaGetLastError()) != cudaSuccess) goto Cleanup;
            if ((status = cudaMemcpy(&hostChanged, changed, sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) goto Cleanup;
            std::swap(labels, next);
            if (!hostChanged) break;
        }

        assignBorderKernel<<<blocks, threads>>>(deviceF0, deviceF1, count,
            ranges.minimum[0], ranges.minimum[1], ranges.inverseRange[0], ranges.inverseRange[1],
            epsilonSquared, valid, core, labels, deviceOutput);
        if ((status = cudaGetLastError()) != cudaSuccess) goto Cleanup;
        status = cudaDeviceSynchronize();
    }

Cleanup:
    if (valid) cudaFree(valid);
    if (core) cudaFree(core);
    if (labels) cudaFree(labels);
    if (next) cudaFree(next);
    if (changed) cudaFree(changed);
    return status;
}

void CompactBOALabels(Computation* data)
{
    if (!data || data->isHires || !data->marshal.kernel.analyses.BOA.toCompute) return;
    unsigned int offsets[3];
    if (!getOffsets(data, offsets)) return;
    const uint64_t count = data->marshal.totalVariations;
    numb* output = data->marshal.maps + (uint64_t)offsets[2] * count;
    const numb* f0 = data->marshal.maps + (uint64_t)offsets[0] * count;
    const numb* f1 = data->marshal.maps + (uint64_t)offsets[1] * count;

    struct Summary { double x = 0, y = 0; uint64_t count = 0; };
    std::map<int, Summary> summaries;
    for (uint64_t i = 0; i < count; ++i)
    {
        const int root = (int)output[i];
        if (root <= 0) continue;
        Summary& summary = summaries[root];
        summary.x += f0[i]; summary.y += f1[i]; ++summary.count;
    }
    std::vector<std::pair<int, Summary>> ordered(summaries.begin(), summaries.end());
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b)
    {
        const double ax = a.second.x / a.second.count, bx = b.second.x / b.second.count;
        if (ax != bx) return ax < bx;
        return a.second.y / a.second.count < b.second.y / b.second.count;
    });
    std::map<int, int> dense;
    for (size_t i = 0; i < ordered.size(); ++i) dense[ordered[i].first] = (int)i + 1;
    for (uint64_t i = 0; i < count; ++i)
        if (output[i] > 0) output[i] = (numb)dense[(int)output[i]];
}
