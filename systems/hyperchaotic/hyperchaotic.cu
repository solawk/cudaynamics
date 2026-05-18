#include "hyperchaotic.h"
#define name hyperchaotic

namespace attributes
{
enum variables { x, y, z, w };
enum parameters { a, b, c, d, method, COUNT };
enum methods { ExplicitEuler, SemiExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, ExplicitDormandPrince8};
}

__global__ void gpu_wrapper_(name)(Computation* data, uint64_t variation)
{
    kernelProgram_(name)(data, (blockIdx.x* blockDim.x) + threadIdx.x);
}
__host__ __device__ void kernelProgram_(name)(Computation* data, uint64_t variation)
{
    if (variation >= CUDA_marshal.totalVariations) return;   // Shutdown thread if there isn't a variation to compute
    uint64_t stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(false);
    // Custom area (usually) starts here
    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_(name));
    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_(name)(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis
    AnalysisLobby(data, &finiteDifferenceScheme_(name), variation);
}
__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters, PerThread* pt)
{

    const int Number = 4;
    numb v[Number] = {V(x), V(y), V(z), V(w)};
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + H * (P(a)*V(x)-V(y)*V(z)+V(w));
        Vnext(y) = V(y) + H * (V(x)*V(z)-P(b)*V(y));
        Vnext(z) = V(z) + H * (V(x)*V(y)-P(c)*V(z));
        Vnext(w) = V(w) + H * (-V(y)+P(d));
    }

    ifMETHOD(P(method), SemiExplicitEuler)
    {
        Vnext(x) = V(x) + H * (P(a)*V(x)-V(y)*V(z)+V(w));
        Vnext(y) = V(y) + H * (Vnext(x)*V(z)-P(b)*V(y));
        Vnext(z) = V(z) + H * (Vnext(x)*Vnext(y)-P(c)*V(z));
        Vnext(w) = V(w) + H * (-Vnext(y)+P(d));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + (numb)0.5 * H * (P(a)*V(x)-V(y)*V(z)+V(w));
        numb ymp = V(y) + (numb)0.5 * H * (V(x)*V(z)-P(b)*V(y));
        numb zmp = V(z) + (numb)0.5 * H * (V(x)*V(y)-P(c)*V(z));
        numb wmp = V(w) + (numb)0.5 * H * (-V(y)+P(d));
        Vnext(x) = V(x) + H * (P(a)*xmp-ymp*zmp+wmp);
        Vnext(y) = V(y) + H * (xmp*zmp-P(b)*ymp);
        Vnext(z) = V(z) + H * (xmp*ymp-P(c)*zmp);
        Vnext(w) = V(w) + H * (-ymp+P(d));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(a)*V(x)-V(y)*V(z)+V(w);
        numb ky1 = V(x)*V(z)-P(b)*V(y);
        numb kz1 = V(x)*V(y)-P(c)*V(z);
        numb kw1 = -V(y)+P(d);

        numb xmp = V(x) + (numb)0.5 * H * kx1;
        numb ymp = V(y) + (numb)0.5 * H * ky1;
        numb zmp = V(z) + (numb)0.5 * H * kz1;
        numb wmp = V(w) + (numb)0.5 * H * kw1;

        numb kx2 = P(a)*xmp-ymp*zmp+wmp;
        numb ky2 = xmp*zmp-P(b)*ymp;
        numb kz2 = xmp*ymp-P(c)*zmp;
        numb kw2 = -ymp+P(d);

        xmp = V(x) + (numb)0.5 * H * kx2;
        ymp = V(y) + (numb)0.5 * H * ky2;
        zmp = V(z) + (numb)0.5 * H * kz2;
        wmp = V(w) + (numb)0.5 * H * kw2;

        numb kx3 = P(a)*xmp-ymp*zmp+wmp;
        numb ky3 = xmp*zmp-P(b)*ymp;
        numb kz3 = xmp*ymp-P(c)*zmp;
        numb kw3 = -ymp+P(d);

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;
        wmp = V(w) + H * kw3;

        numb kx4 = P(a)*xmp-ymp*zmp+wmp;
        numb ky4 = xmp*zmp-P(b)*ymp;
        numb kz4 = xmp*ymp-P(c)*zmp;
        numb kw4 = -ymp+P(d);

        Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;
        Vnext(y) = V(y) + H * (ky1 + (numb)2.0 * ky2 + (numb)2.0 * ky3 + ky4) / (numb)6.0;
        Vnext(z) = V(z) + H * (kz1 + (numb)2.0 * kz2 + (numb)2.0 * kz3 + kz4) / (numb)6.0;
        Vnext(w) = V(w) + H * (kw1 + (numb)2.0 * kw2 + (numb)2.0 * kw3 + kw4) / (numb)6.0;
    }

    ifMETHOD(P(method), ExplicitDormandPrince8)
    {
			const numb M[13][12] = { {(numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.05555555555556, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.02083333333333, (numb)0.0625, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.03125, (numb)0.0, (numb)0.09375, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.3125, (numb)0.0, -(numb)1.171875, (numb)1.171875, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.0375, (numb)0.0, (numb)0.0, (numb)0.1875, (numb)0.15, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.04791013711111, (numb)0.0, (numb)0.0, (numb)0.1122487127778, -(numb)0.02550567377778, (numb)0.01284682388889, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.01691798978729, (numb)0.0, (numb)0.0, (numb)0.387848278486, (numb)0.0359773698515, (numb)0.1969702142157, -(numb)0.1727138523405, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.06909575335919, (numb)0.0, (numb)0.0, -(numb)0.6342479767289, -(numb)0.1611975752246, (numb)0.1386503094588, (numb)0.9409286140358, (numb)0.2116363264819, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.183556996839, (numb)0.0, (numb)0.0, -(numb)2.468768084316, -(numb)0.2912868878163, -(numb)0.02647302023312, (numb)2.847838764193, (numb)0.2813873314699, (numb)0.1237448998633, (numb)0.0, (numb)0.0, (numb)0.0},
								{-(numb)1.215424817396, (numb)0.0, (numb)0.0, (numb)16.67260866595, (numb)0.9157418284168, -(numb)6.056605804357, -(numb)16.00357359416, (numb)14.8493030863, -(numb)13.37157573529, (numb)5.13418264818, (numb)0.0, (numb)0.0},
								{(numb)0.2588609164383, (numb)0.0, (numb)0.0, -(numb)4.774485785489, -(numb)0.435093013777, -(numb)3.049483332072, (numb)5.577920039936, (numb)6.155831589861, -(numb)5.062104586737, (numb)2.193926173181, (numb)0.1346279986593, (numb)0.0},
								{(numb)0.8224275996265, (numb)0.0, (numb)0.0, -(numb)11.65867325728, -(numb)0.7576221166909, (numb)0.7139735881596, (numb)12.07577498689, -(numb)2.12765911392, (numb)1.990166207049, -(numb)0.234286471544, (numb)0.1758985777079, (numb)0.0} };

			const numb b[13] = { (numb)0.04174749114153, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, -(numb)0.05545232861124, (numb)0.2393128072012, (numb)0.7035106694034, -(numb)0.7597596138145, (numb)0.6605630309223, (numb)0.1581874825101, -(numb)0.2381095387529, (numb)0.25 };
          numb y[Number], X1[Number], X2[Number];
          numb k[Number][13];
          int i = 0, j = 0, l = 0;
        for (i = 0; i < Number; i++){
            X1[i] = v[i];
        }
        for (i = 0; i < 13; i++){
            k[0][i] = P(a)*X1[0]-X1[1]*X1[2]+X1[3];
            k[1][i] = X1[0]*X1[2]-P(b)*X1[1];
            k[2][i] = X1[0]*X1[1]-P(c)*X1[2];
            k[3][i] = -X1[1]+P(d);
				for (l = 0; l < Number; l++)
 					X2[l] = 0;

				for (j = 0; j < i + 1; j++)
					for (l = 0; l < Number; l++)
						X2[l] += M[i + 1][j] * k[l][j];

				for (l = 0; l < Number; l++)
					X1[l] = v[l] + H * X2[l];

          }

			for (l = 0; l < Number; l++)
				X2[l] = 0;

			for (i = 0; i < 13; i++)
				for (l = 0; l < Number; l++)
					X2[l] += b[i] * k[l][i];

			for (l = 0; l < Number; l++)
				y[l] = v[l] + H * X2[l];
        Vnext(x) = y[0]; 
        Vnext(y) = y[1]; 
        Vnext(z) = y[2]; 
        Vnext(w) = y[3]; 
    }

}
