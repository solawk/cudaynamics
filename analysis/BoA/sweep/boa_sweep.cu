#include "boa_sweep.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

#include "../../../index2port.h"

namespace boa_sweep_detail
{
    constexpr int INVALID = -2;
    constexpr int NOISE = -1;

    struct Ranges
    {
        numb minimum[2];
        numb inverse[2];
    };

    bool getLayout(Computation* data, unsigned int offsets[3], uint64_t& stride,
        uint64_t& layers, uint64_t& pointsPerLayer)
    {
        if (!data) return false;
        BOA_Settings& settings = data->marshal.kernel.analyses.BOA;
        Kernel& kernel = data->marshal.kernel;
        if (settings.sweepParameter < 0 || settings.sweepParameter >= kernel.PARAM_COUNT) return false;
        if (kernel.stepType == ST_Parameter && settings.sweepParameter == kernel.PARAM_COUNT - 1) return false;
        layers = (uint64_t)kernel.parameters[settings.sweepParameter].TrueStepCount();
        if (layers < 2 || data->marshal.totalVariations % layers != 0) return false;

        Port* f0 = index2port(kernel.analyses, settings.features[0]);
        Port* f1 = index2port(kernel.analyses, settings.features[1]);
        if (!f0 || !f1 || !settings.basinId.used || !f0->used || !f1->used) return false;
        offsets[0] = f0->offset;
        offsets[1] = f1->offset;
        offsets[2] = settings.basinId.offset;

        stride = 1;
        for (int v = 0; v < kernel.VAR_COUNT; ++v)
            stride *= (uint64_t)kernel.variables[v].TrueStepCount();
        for (int p = 0; p < settings.sweepParameter; ++p)
            stride *= (uint64_t)kernel.parameters[p].TrueStepCount();
        pointsPerLayer = data->marshal.totalVariations / layers;
        return stride > 0 && pointsPerLayer > 0;
    }

    __host__ __device__ uint64_t globalIndex(uint64_t orderedIndex, uint64_t stride,
        uint64_t layers, uint64_t pointsPerLayer)
    {
        const uint64_t layer = orderedIndex / pointsPerLayer;
        const uint64_t local = orderedIndex - layer * pointsPerLayer;
        const uint64_t low = local % stride;
        const uint64_t high = local / stride;
        return low + layer * stride + high * stride * layers;
    }

    bool calculateRanges(const numb* f0, const numb* f1, uint64_t count, Ranges& ranges)
    {
        ranges.minimum[0] = ranges.minimum[1] = std::numeric_limits<numb>::infinity();
        numb maximum[2] = { -std::numeric_limits<numb>::infinity(), -std::numeric_limits<numb>::infinity() };
        bool found = false;
        for (uint64_t i = 0; i < count; ++i)
        {
            if (!std::isfinite((double)f0[i]) || !std::isfinite((double)f1[i])) continue;
            found = true;
            ranges.minimum[0] = std::min(ranges.minimum[0], f0[i]);
            ranges.minimum[1] = std::min(ranges.minimum[1], f1[i]);
            maximum[0] = std::max(maximum[0], f0[i]);
            maximum[1] = std::max(maximum[1], f1[i]);
        }
        if (!found) return false;
        for (int d = 0; d < 2; ++d)
        {
            const numb span = maximum[d] - ranges.minimum[d];
            ranges.inverse[d] = span > (numb)0 ? (numb)1 / span : (numb)0;
        }
        return true;
    }

    inline bool hostNeighbor(const numb* f0, const numb* f1, uint64_t a, uint64_t b,
        const Ranges& ranges, numb epsilonSquared)
    {
        if (!std::isfinite((double)f0[a]) || !std::isfinite((double)f1[a]) ||
            !std::isfinite((double)f0[b]) || !std::isfinite((double)f1[b])) return false;
        const numb dx = (f0[a] - f0[b]) * ranges.inverse[0];
        const numb dy = (f1[a] - f1[b]) * ranges.inverse[1];
        return dx * dx + dy * dy <= epsilonSquared;
    }

    __device__ bool deviceFinite(numb value) { return isfinite(value) != 0; }

    __device__ bool deviceNeighbor(const numb* f0, const numb* f1, uint64_t a, uint64_t b,
        numb inv0, numb inv1, numb epsilonSquared)
    {
        if (!deviceFinite(f0[a]) || !deviceFinite(f1[a]) ||
            !deviceFinite(f0[b]) || !deviceFinite(f1[b])) return false;
        const numb dx = (f0[a] - f0[b]) * inv0;
        const numb dy = (f1[a] - f1[b]) * inv1;
        return dx * dx + dy * dy <= epsilonSquared;
    }

    __global__ void classifyKernel(const numb* f0, const numb* f1, uint64_t count,
        uint64_t stride, uint64_t layers, uint64_t pointsPerLayer, numb inv0, numb inv1,
        numb epsilonSquared, int minimumPoints, unsigned char* valid, unsigned char* core, int* labels)
    {
        const uint64_t q = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (q >= count) return;
        const uint64_t i = globalIndex(q, stride, layers, pointsPerLayer);
        valid[q] = deviceFinite(f0[i]) && deviceFinite(f1[i]);
        if (!valid[q]) { core[q] = 0; labels[q] = INVALID; return; }
        const uint64_t first = (q / pointsPerLayer) * pointsPerLayer;
        int neighbors = 0;
        for (uint64_t local = 0; local < pointsPerLayer && neighbors < minimumPoints; ++local)
        {
            const uint64_t other = first + local;
            const uint64_t j = globalIndex(other, stride, layers, pointsPerLayer);
            if (deviceNeighbor(f0, f1, i, j, inv0, inv1, epsilonSquared)) ++neighbors;
        }
        core[q] = neighbors >= minimumPoints;
        labels[q] = core[q] ? (int)q + 1 : NOISE;
    }

    __global__ void propagateKernel(const numb* f0, const numb* f1, uint64_t count,
        uint64_t stride, uint64_t layers, uint64_t pointsPerLayer, numb inv0, numb inv1,
        numb epsilonSquared, const unsigned char* core, const int* input, int* output, int* changed)
    {
        const uint64_t q = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (q >= count) return;
        if (!core[q]) { output[q] = input[q]; return; }
        const uint64_t i = globalIndex(q, stride, layers, pointsPerLayer);
        const uint64_t first = (q / pointsPerLayer) * pointsPerLayer;
        int best = input[q];
        for (uint64_t local = 0; local < pointsPerLayer; ++local)
        {
            const uint64_t other = first + local;
            if (!core[other] || input[other] <= 0 || input[other] >= best) continue;
            const uint64_t j = globalIndex(other, stride, layers, pointsPerLayer);
            if (deviceNeighbor(f0, f1, i, j, inv0, inv1, epsilonSquared)) best = input[other];
        }
        output[q] = best;
        if (best != input[q]) atomicExch(changed, 1);
    }

    __global__ void borderKernel(const numb* f0, const numb* f1, numb* output, uint64_t count,
        uint64_t stride, uint64_t layers, uint64_t pointsPerLayer, numb inv0, numb inv1,
        numb epsilonSquared, const unsigned char* valid, const unsigned char* core, const int* labels)
    {
        const uint64_t q = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (q >= count) return;
        const uint64_t i = globalIndex(q, stride, layers, pointsPerLayer);
        if (!valid[q]) { output[i] = (numb)INVALID; return; }
        if (core[q]) { output[i] = (numb)labels[q]; return; }
        const uint64_t first = (q / pointsPerLayer) * pointsPerLayer;
        int best = INT_MAX;
        for (uint64_t local = 0; local < pointsPerLayer; ++local)
        {
            const uint64_t other = first + local;
            if (!core[other] || labels[other] <= 0 || labels[other] >= best) continue;
            const uint64_t j = globalIndex(other, stride, layers, pointsPerLayer);
            if (deviceNeighbor(f0, f1, i, j, inv0, inv1, epsilonSquared)) best = labels[other];
        }
        output[i] = best == INT_MAX ? (numb)NOISE : (numb)best;
    }

    struct Cluster
    {
        int root = 0;
        int track = 0;
        double x = 0.0;
        double y = 0.0;
        std::vector<uint64_t> members;
    };

    double intersectionOverUnion(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b)
    {
        size_t ia = 0, ib = 0, intersection = 0;
        while (ia < a.size() && ib < b.size())
        {
            if (a[ia] == b[ib]) { ++intersection; ++ia; ++ib; }
            else if (a[ia] < b[ib]) ++ia;
            else ++ib;
        }
        const size_t unionSize = a.size() + b.size() - intersection;
        return unionSize ? (double)intersection / (double)unionSize : 0.0;
    }
}

void FinalizeBOASweepOpenMP(Computation* data)
{
    using namespace boa_sweep_detail;
    if (!data || data->isHires || !data->marshal.kernel.analyses.BOA.toCompute) return;
    unsigned int offsets[3];
    uint64_t stride, layers, pointsPerLayer;
    if (!getLayout(data, offsets, stride, layers, pointsPerLayer)) return;
    const uint64_t count = data->marshal.totalVariations;
    numb* maps = data->marshal.maps;
    const numb* f0 = maps + (uint64_t)offsets[0] * count;
    const numb* f1 = maps + (uint64_t)offsets[1] * count;
    numb* output = maps + (uint64_t)offsets[2] * count;
    Ranges ranges;
    if (!calculateRanges(f0, f1, count, ranges)) { std::fill(output, output + count, (numb)INVALID); return; }

    const BOA_Settings& settings = data->marshal.kernel.analyses.BOA;
    const numb epsilonSquared = settings.epsilon * settings.epsilon;
    std::vector<unsigned char> valid(count), core(count);
    std::vector<int> labels(count), next(count);

#pragma omp parallel for
    for (long long qSigned = 0; qSigned < (long long)count; ++qSigned)
    {
        const uint64_t q = (uint64_t)qSigned;
        const uint64_t i = globalIndex(q, stride, layers, pointsPerLayer);
        valid[q] = std::isfinite((double)f0[i]) && std::isfinite((double)f1[i]);
        if (!valid[q]) { core[q] = 0; labels[q] = INVALID; continue; }
        const uint64_t first = (q / pointsPerLayer) * pointsPerLayer;
        int neighbors = 0;
        for (uint64_t local = 0; local < pointsPerLayer && neighbors < settings.minimumPoints; ++local)
        {
            const uint64_t other = first + local;
            if (hostNeighbor(f0, f1, i, globalIndex(other, stride, layers, pointsPerLayer), ranges, epsilonSquared)) ++neighbors;
        }
        core[q] = neighbors >= settings.minimumPoints;
        labels[q] = core[q] ? (int)q + 1 : NOISE;
    }

    for (uint64_t iteration = 0; iteration < pointsPerLayer; ++iteration)
    {
        int changed = 0;
#pragma omp parallel for reduction(|:changed)
        for (long long qSigned = 0; qSigned < (long long)count; ++qSigned)
        {
            const uint64_t q = (uint64_t)qSigned;
            if (!core[q]) { next[q] = labels[q]; continue; }
            const uint64_t i = globalIndex(q, stride, layers, pointsPerLayer);
            const uint64_t first = (q / pointsPerLayer) * pointsPerLayer;
            int best = labels[q];
            for (uint64_t local = 0; local < pointsPerLayer; ++local)
            {
                const uint64_t other = first + local;
                if (core[other] && labels[other] > 0 && labels[other] < best &&
                    hostNeighbor(f0, f1, i, globalIndex(other, stride, layers, pointsPerLayer), ranges, epsilonSquared))
                    best = labels[other];
            }
            next[q] = best;
            changed |= best != labels[q];
        }
        labels.swap(next);
        if (!changed) break;
    }

#pragma omp parallel for
    for (long long qSigned = 0; qSigned < (long long)count; ++qSigned)
    {
        const uint64_t q = (uint64_t)qSigned;
        const uint64_t i = globalIndex(q, stride, layers, pointsPerLayer);
        if (!valid[q]) { output[i] = (numb)INVALID; continue; }
        if (core[q]) { output[i] = (numb)labels[q]; continue; }
        const uint64_t first = (q / pointsPerLayer) * pointsPerLayer;
        int best = INT_MAX;
        for (uint64_t local = 0; local < pointsPerLayer; ++local)
        {
            const uint64_t other = first + local;
            if (core[other] && labels[other] > 0 && labels[other] < best &&
                hostNeighbor(f0, f1, i, globalIndex(other, stride, layers, pointsPerLayer), ranges, epsilonSquared))
                best = labels[other];
        }
        output[i] = best == INT_MAX ? (numb)NOISE : (numb)best;
    }
}

cudaError_t FinalizeBOASweepCUDA(Computation* data, numb* cudaMaps, uint64_t count)
{
    using namespace boa_sweep_detail;
    if (!data || data->isHires || !data->marshal.kernel.analyses.BOA.toCompute) return cudaSuccess;
    unsigned int offsets[3];
    uint64_t stride, layers, pointsPerLayer;
    if (!getLayout(data, offsets, stride, layers, pointsPerLayer) || count != data->marshal.totalVariations)
        return cudaErrorInvalidValue;

    const numb* deviceF0 = cudaMaps + (uint64_t)offsets[0] * count;
    const numb* deviceF1 = cudaMaps + (uint64_t)offsets[1] * count;
    numb* deviceOutput = cudaMaps + (uint64_t)offsets[2] * count;
    std::vector<numb> hostF0(count), hostF1(count);
    cudaError_t status = cudaMemcpy(hostF0.data(), deviceF0, count * sizeof(numb), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return status;
    status = cudaMemcpy(hostF1.data(), deviceF1, count * sizeof(numb), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return status;
    Ranges ranges;
    if (!calculateRanges(hostF0.data(), hostF1.data(), count, ranges))
    {
        std::vector<numb> invalid(count, (numb)INVALID);
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
        classifyKernel<<<blocks, threads>>>(deviceF0, deviceF1, count, stride, layers, pointsPerLayer,
            ranges.inverse[0], ranges.inverse[1], epsilonSquared, settings.minimumPoints, valid, core, labels);
        if ((status = cudaGetLastError()) != cudaSuccess) goto Cleanup;

        for (uint64_t iteration = 0; iteration < pointsPerLayer; ++iteration)
        {
            int hostChanged = 0;
            if ((status = cudaMemset(changed, 0, sizeof(int))) != cudaSuccess) goto Cleanup;
            propagateKernel<<<blocks, threads>>>(deviceF0, deviceF1, count, stride, layers, pointsPerLayer,
                ranges.inverse[0], ranges.inverse[1], epsilonSquared, core, labels, next, changed);
            if ((status = cudaGetLastError()) != cudaSuccess) goto Cleanup;
            if ((status = cudaMemcpy(&hostChanged, changed, sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) goto Cleanup;
            std::swap(labels, next);
            if (!hostChanged) break;
        }

        borderKernel<<<blocks, threads>>>(deviceF0, deviceF1, deviceOutput, count, stride, layers,
            pointsPerLayer, ranges.inverse[0], ranges.inverse[1], epsilonSquared, valid, core, labels);
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

void TrackBOASweepLabels(Computation* data)
{
    using namespace boa_sweep_detail;
    if (!data || data->isHires || !data->marshal.kernel.analyses.BOA.toCompute) return;
    unsigned int offsets[3];
    uint64_t stride, layers, pointsPerLayer;
    if (!getLayout(data, offsets, stride, layers, pointsPerLayer)) return;
    const uint64_t count = data->marshal.totalVariations;
    numb* output = data->marshal.maps + (uint64_t)offsets[2] * count;
    const numb* f0 = data->marshal.maps + (uint64_t)offsets[0] * count;
    const numb* f1 = data->marshal.maps + (uint64_t)offsets[1] * count;
    Ranges ranges;
    if (!calculateRanges(f0, f1, count, ranges)) return;

    std::vector<std::vector<Cluster>> layerClusters((size_t)layers);
    for (uint64_t layer = 0; layer < layers; ++layer)
    {
        std::map<int, size_t> rootToCluster;
        std::vector<Cluster>& clusters = layerClusters[(size_t)layer];
        for (uint64_t local = 0; local < pointsPerLayer; ++local)
        {
            const uint64_t q = layer * pointsPerLayer + local;
            const uint64_t i = globalIndex(q, stride, layers, pointsPerLayer);
            const int root = (int)output[i];
            if (root <= 0) continue;
            auto inserted = rootToCluster.emplace(root, clusters.size());
            if (inserted.second) { clusters.push_back(Cluster()); clusters.back().root = root; }
            Cluster& cluster = clusters[inserted.first->second];
            cluster.x += ((double)f0[i] - ranges.minimum[0]) * ranges.inverse[0];
            cluster.y += ((double)f1[i] - ranges.minimum[1]) * ranges.inverse[1];
            cluster.members.push_back(local);
        }
        for (Cluster& cluster : clusters)
        {
            cluster.x /= (double)cluster.members.size();
            cluster.y /= (double)cluster.members.size();
        }
        std::sort(clusters.begin(), clusters.end(), [](const Cluster& a, const Cluster& b)
        {
            if (a.x != b.x) return a.x < b.x;
            if (a.y != b.y) return a.y < b.y;
            return a.root < b.root;
        });
    }

    int nextTrack = 1;
    if (!layerClusters.empty())
        for (Cluster& cluster : layerClusters[0]) cluster.track = nextTrack++;

    const double matchThreshold = std::max(0.35,
        std::min(0.75, (double)data->marshal.kernel.analyses.BOA.epsilon * 6.0));
    struct Candidate { double cost; size_t previous; size_t current; };
    for (size_t layer = 1; layer < layerClusters.size(); ++layer)
    {
        std::vector<Cluster>& previous = layerClusters[layer - 1];
        std::vector<Cluster>& current = layerClusters[layer];
        std::vector<Candidate> candidates;
        for (size_t p = 0; p < previous.size(); ++p)
            for (size_t c = 0; c < current.size(); ++c)
            {
                const double dx = previous[p].x - current[c].x;
                const double dy = previous[p].y - current[c].y;
                const double featureDistance = std::sqrt(dx * dx + dy * dy);
                const double overlap = intersectionOverUnion(previous[p].members, current[c].members);
                candidates.push_back({ 0.65 * featureDistance + 0.35 * (1.0 - overlap), p, c });
            }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b)
        {
            if (a.cost != b.cost) return a.cost < b.cost;
            if (a.previous != b.previous) return a.previous < b.previous;
            return a.current < b.current;
        });
        std::vector<unsigned char> usedPrevious(previous.size(), 0), usedCurrent(current.size(), 0);
        for (const Candidate& candidate : candidates)
        {
            if (candidate.cost > matchThreshold) break;
            if (usedPrevious[candidate.previous] || usedCurrent[candidate.current]) continue;
            current[candidate.current].track = previous[candidate.previous].track;
            usedPrevious[candidate.previous] = usedCurrent[candidate.current] = 1;
        }
        for (Cluster& cluster : current) if (!cluster.track) cluster.track = nextTrack++;
    }

    for (uint64_t layer = 0; layer < layers; ++layer)
        for (const Cluster& cluster : layerClusters[(size_t)layer])
            for (uint64_t local : cluster.members)
            {
                const uint64_t q = layer * pointsPerLayer + local;
                output[globalIndex(q, stride, layers, pointsPerLayer)] = (numb)cluster.track;
            }
}

