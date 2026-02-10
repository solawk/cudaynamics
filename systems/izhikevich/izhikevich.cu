#include "izhikevich.h"

#define name izhikevich

namespace attributes
{
    enum variables { v, u, i, t };
    enum parameters { a, b, c, d, p0, p1, p2, p3, Idc, Iamp, Ifreq, Idel, Idf, symmetry, signal, method, COUNT };
	enum waveforms { square };
    enum methods { ExplicitEuler, SemiExplicitEuler, ImplicitEuler, ExplicitMidpoint, ImplicitMidpoint, ExplicitRungeKutta4, ExplicitDormandPrince8, VariableSymmetryCD };
}

__global__ void gpu_wrapper_(name)(Computation* data, uint64_t variation)
{
    kernelProgram_(name)(data, (blockIdx.x * blockDim.x) + threadIdx.x);
}

__host__ __device__ void kernelProgram_(name)(Computation* data, uint64_t variation)
{
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
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

__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters)
{	
    ifSIGNAL(P(signal), square)
	{
		ifMETHOD(P(method), ExplicitEuler)
		{
			Vnext(i) = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
			Vnext(t) = V(t) + H;
			Vnext(v) = V(v) + H * (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + Vnext(i));
			Vnext(u) = V(u) + H * (P(a) * (P(b) * V(v) - V(u)));

			if (Vnext(v) >= P(p3))
			{
				Vnext(v) = P(c);
				Vnext(u) = Vnext(u) + P(d);
			}
		}

		ifMETHOD(P(method), SemiExplicitEuler)
		{
			Vnext(t) = V(t) + H;
			Vnext(i) = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
			Vnext(v) = V(v) + H * (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + Vnext(i));
			Vnext(u) = V(u) + H * (P(a) * (P(b) * Vnext(v) - V(u)));
			if (Vnext(v) >= P(p3))
			{
				Vnext(v) = P(c);
				Vnext(u) = Vnext(u) + P(d);
			}
		}

        ifMETHOD(P(method), ImplicitEuler)
        {
            Vnext(i) = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            Vnext(t) = V(t) + H;

            numb v_guess = V(v) + H * (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + Vnext(i));
            numb u_guess = V(u) + H * (P(a) * (P(b) * Vnext(v) - V(u)));

            v_guess = V(v) + H * (P(p0) * v_guess * v_guess + P(p1) * v_guess + P(p2) - u_guess + Vnext(i));
            v_guess = V(v) + H * (P(p0) * v_guess * v_guess + P(p1) * v_guess + P(p2) - u_guess + Vnext(i));
            v_guess = V(v) + H * (P(p0) * v_guess * v_guess + P(p1) * v_guess + P(p2) - u_guess + Vnext(i));
            v_guess = V(v) + H * (P(p0) * v_guess * v_guess + P(p1) * v_guess + P(p2) - u_guess + Vnext(i));

            u_guess = V(u) + H * (P(a) * (P(b) * v_guess - u_guess));
            u_guess = V(u) + H * (P(a) * (P(b) * v_guess - u_guess));
            u_guess = V(u) + H * (P(a) * (P(b) * v_guess - u_guess));
            u_guess = V(u) + H * (P(a) * (P(b) * v_guess - u_guess));

            if (v_guess >= P(p3))
            {
                v_guess = P(c);
                u_guess = u_guess + P(d);
            }

            Vnext(v) = v_guess;
            Vnext(u) = u_guess;
        }

        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            numb tmp = V(t) + H * (numb)0.5;
            numb vmp = V(v) + H * (numb)0.5 * (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + Vnext(i));
            numb ump = V(u) + H * (numb)0.5 * (P(a) * (P(b) * Vnext(v) - V(u)));

            Vnext(i) = P(Idc) + (fmod((tmp - P(Idel)) > (numb)0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            Vnext(t) = V(t) + H;
            Vnext(v) = V(v) + H * (P(p0) * vmp * vmp + P(p1) * vmp + P(p2) - ump + Vnext(i));
            Vnext(u) = V(u) + H * (P(a) * (P(b) * vmp - ump));

            if (Vnext(v)  >= P(p3))
            {
                Vnext(v) = P(c);
                Vnext(u) = Vnext(u) + P(d);
            }
        }

        ifMETHOD(P(method), ImplicitMidpoint)
        {
            //не квадратный сигнал? опечатка?
            numb imp = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor(((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0)) * pow((numb)(-1), floor(((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0)));
            numb tmp = V(t) + (numb)0.5 * H;
            numb v_mid_guess = V(v) + H * (numb)0.5 * (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + imp);
            numb u_mid_guess = V(u) + H * (numb)0.5 * (P(a) * (P(b) * Vnext(v) - V(u)));

            v_mid_guess = V(v) + H * (numb)0.5 * (P(p0) * v_mid_guess * v_mid_guess + P(p1) * v_mid_guess + P(p2) - u_mid_guess + imp);
            v_mid_guess = V(v) + H * (numb)0.5 * (P(p0) * v_mid_guess * v_mid_guess + P(p1) * v_mid_guess + P(p2) - u_mid_guess + imp);
            v_mid_guess = V(v) + H * (numb)0.5 * (P(p0) * v_mid_guess * v_mid_guess + P(p1) * v_mid_guess + P(p2) - u_mid_guess + imp);
            v_mid_guess = V(v) + H * (numb)0.5 * (P(p0) * v_mid_guess * v_mid_guess + P(p1) * v_mid_guess + P(p2) - u_mid_guess + imp);

            u_mid_guess = V(u) + H * (numb)0.5 * (P(a) * (P(b) * v_mid_guess - u_mid_guess));
            u_mid_guess = V(u) + H * (numb)0.5 * (P(a) * (P(b) * v_mid_guess - u_mid_guess));
            u_mid_guess = V(u) + H * (numb)0.5 * (P(a) * (P(b) * v_mid_guess - u_mid_guess));
            u_mid_guess = V(u) + H * (numb)0.5 * (P(a) * (P(b) * v_mid_guess - u_mid_guess));

            Vnext(t) = V(t) + H;
            Vnext(i) = P(Idc) + (fmod((tmp - P(Idel)) > (numb)0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            Vnext(v) = V(v) + H * (P(p0) * v_mid_guess * v_mid_guess + P(p1) * v_mid_guess + P(p2) - u_mid_guess + Vnext(i));
            Vnext(u) = V(u) + H * (P(a) * (P(b) * v_mid_guess - u_mid_guess));

            if (Vnext(v) >= P(p3))
            {
                Vnext(v) = P(c);
                Vnext(u) = Vnext(u) + P(d);
            }
        }

        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            numb i1 = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            numb kt1 = V(t) + (numb)0.5 * H;

            numb kv1 = (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + i1);
            numb ku1 = (P(a) * (P(b) * V(v) - V(u)));

            numb vmp = V(v) + (numb)0.5 * H * kv1;
            numb ump = V(u) + (numb)0.5 * H * ku1;

            numb i2 = P(Idc) + (fmod((kt1 - P(Idel)) > (numb)0 ? (kt1 - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - kt1), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            numb kv2 = (P(p0) * vmp * vmp + P(p1) * vmp + P(p2) - ump + i2);
            numb ku2 = (P(a) * (P(b) * vmp - ump));

            vmp = V(v) + (numb)0.5 * H * kv2;
            ump = V(u) + (numb)0.5 * H * ku2;
             
            numb kv3 = (P(p0) * vmp * vmp + P(p1) * vmp + P(p2) - ump + i2);
            numb ku3 = (P(a) * (P(b) * vmp - ump));
            Vnext(t) = V(t) + H;

            vmp = V(v) + (numb)0.5 * H * kv2;
            ump = V(u) + (numb)0.5 * H * ku2;

            numb i3 = P(Idc) + (fmod((Vnext(t) - P(Idel)) > (numb)0 ? (Vnext(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - Vnext(t)), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            numb kv4 = (P(p0) * vmp * vmp + P(p1) * vmp + P(p2) - ump + i3);
            numb ku4 = (P(a) * (P(b) * vmp - ump));

            Vnext(i) = i3;
            Vnext(v) = V(v) + H * (kv1 + (numb)2.0 * kv2 + (numb)2.0 * kv3 + kv4) / (numb)6.0;
            Vnext(u) = V(u) + H * (ku1 + (numb)2.0 * ku2 + (numb)2.0 * ku3 + ku4) / (numb)6.0;

            if (Vnext(v) >= P(p3))
            {
                Vnext(v) = P(c);
                Vnext(u) = Vnext(u) + P(d);
            }
        }

        ifMETHOD(P(method), ExplicitDormandPrince8)
        {
            static const double c[13] = {
                0.0,
                1.0 / 18.0,
                1.0 / 12.0,
                1.0 / 8.0,
                5.0 / 16.0,
                3.0 / 8.0,
                59.0 / 400.0,
                93.0 / 200.0,
                5490023248.0 / 9719169821.0,
                13.0 / 20.0,
                12.0 / 25.0,
                1.0,
                1.0
            };

            static const double a[13][12] = {
                {},
                {1.0 / 18.0},
                {1.0 / 48.0, 1.0 / 16.0},
                {1.0 / 32.0, 0.0, 3.0 / 32.0},
                {5.0 / 128.0, 0.0, -15.0 / 128.0, 5.0 / 32.0},
                {-3.0 / 80.0, 0.0, 0.0, 3.0 / 16.0, 3.0 / 20.0},
                {32933113.0 / 3014655488.0, 0.0, 0.0, 134014359.0 / 753663872.0, -1354378445.0 / 3014655488.0, 327980961.0 / 753663872.0},
                {-757460963.0 / 2653356192.0, 0.0, 0.0, -262546707.0 / 442226032.0, 1012840125.0 / 884452064.0, 1290531855.0 / 884452064.0, 202759525.0 / 221113016.0},
                {-25232958782419.0 / 87130555970439.0, 0.0, 0.0, -800514088877.0 / 3057787176511.0, 93953711538425.0 / 116174074627252.0, 92841053569925.0 / 58087037313626.0, 10100166976385.0 / 58087037313626.0, -257553667771.0 / 1144572155999.0},
                {-8571631609.0 / 18884848314.0, 0.0, 0.0, 6479160439.0 / 25179797752.0, -15102881525.0 / 25179797752.0, -24536262555.0 / 25179797752.0, -2687155255.0 / 6294949438.0, 897144750.0 / 786868679.0, 1044340225.0 / 7868686794.0},
                {-1839.0 / 4000.0, 0.0, 0.0, 1689.0 / 2500.0, -3009.0 / 2500.0, -2019.0 / 2500.0, -229.0 / 625.0, 756.0 / 625.0, 75.0 / 1250.0, 24.0 / 125.0},
                {31.0 / 30.0, 0.0, 0.0, -51.0 / 10.0, 63.0 / 10.0, 54.0 / 10.0, 14.0 / 5.0, -56.0 / 15.0, -49.0 / 300.0, -16.0 / 75.0, 1.0 / 15.0},
                {1.0 / 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0 / 10.0}
            };

            static const double b8[13] = {
                1.0 / 10.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                9.0 / 10.0
            };

            static const double b7[13] = {
                1168991055722003.0 / 17912098568973600.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1402743477427559.0 / 9194859585263040.0,
                379790491576451.0 / 1791209856897360.0,
                3259891213292487.0 / 49540097866927400.0,
                181069659226147.0 / 3302673191128493.0,
                144162308391168.0 / 13210692764513975.0,
                57427079392303.0 / 13210692764513975.0,
                17836812207633.0 / 13210692764513975.0
            };

            numb I_input = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            numb t_next = V(t) + H;

            numb k_v[13], k_u[13];

            k_v[0] = P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + I_input;
            k_u[0] = P(a) * (P(b) * V(v) - V(u));

            for (int i = 1; i < 13; ++i)
            {
                numb v_stage = V(v);
                numb u_stage = V(u);

                for (int j = 0; j < i; ++j)
                {
                    v_stage += H * (numb)(a[i][j]) * k_v[j];
                    u_stage += H * (numb)(a[i][j]) * k_u[j];
                }

                numb I_stage = I_input;
                k_v[i] = P(p0) * v_stage * v_stage + P(p1) * v_stage + P(p2) - u_stage + I_stage;
                k_u[i] = P(a) * (P(b) * v_stage - u_stage);
            }

  
            numb v_next = V(v);
            numb u_next = V(u);
            for (int i = 0; i < 13; ++i)
            {
                v_next += H * (numb)(b8[i]) * k_v[i];
                u_next += H * (numb)(b8[i]) * k_u[i];
            }

            if (v_next >= P(p3))
            {
                v_next = P(c);
                u_next += P(d);
            }

            Vnext(v) = v_next;
            Vnext(u) = u_next;
            Vnext(t) = t_next;
        }

        ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = (numb)0.5 * H - P(symmetry);
            numb h2 = (numb)0.5 * H + P(symmetry);

            Vnext(i) = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            Vnext(v) = V(v) + h1 * (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + Vnext(i));
            Vnext(u) = V(u) + h1 * (P(a) * (P(b) * Vnext(v) - V(u)));

            numb vmp = Vnext(v);
            numb ump = Vnext(u);

            Vnext(u) = (ump + h2 * P(b) * P(a) * vmp) / ((numb)1.0 + h2 * P(a));

            Vnext(v) = vmp + h2 * (P(p0) * Vnext(v) * Vnext(v) + P(p1) * Vnext(v) + P(p2) - Vnext(u) + Vnext(i));
            Vnext(v) = vmp + h2 * (P(p0) * Vnext(v) * Vnext(v) + P(p1) * Vnext(v) + P(p2) - Vnext(u) + Vnext(i));
            Vnext(v) = vmp + h2 * (P(p0) * Vnext(v) * Vnext(v) + P(p1) * Vnext(v) + P(p2) - Vnext(u) + Vnext(i));
            Vnext(v) = vmp + h2 * (P(p0) * Vnext(v) * Vnext(v) + P(p1) * Vnext(v) + P(p2) - Vnext(u) + Vnext(i));
            Vnext(v) = vmp + h2 * (P(p0) * Vnext(v) * Vnext(v) + P(p1) * Vnext(v) + P(p2) - Vnext(u) + Vnext(i));
            Vnext(v) = vmp + h2 * (P(p0) * Vnext(v) * Vnext(v) + P(p1) * Vnext(v) + P(p2) - Vnext(u) + Vnext(i));
            Vnext(v) = vmp + h2 * (P(p0) * Vnext(v) * Vnext(v) + P(p1) * Vnext(v) + P(p2) - Vnext(u) + Vnext(i));
            Vnext(t) = V(t) + H;


            if (Vnext(v) >= P(p3))
            {
                Vnext(v) = P(c);
                Vnext(u) = Vnext(u) + P(d);
            }
        }
	}
}

#undef name