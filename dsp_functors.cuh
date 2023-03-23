#include "dsp.cuh"

#ifndef DSP_FUNCTORS_CUH
#define DSP_FUNCTORS_CUH

struct calibration_functor : thrust::unary_function<tcf, tcf>
{
    float a_qi, a_qq;
    float c_i, c_q;

    calibration_functor(float _a_qi, float _a_qq,
        float _c_i, float _c_q) : a_qi{ _a_qi }, a_qq{ _a_qq },
        c_i{ _c_i }, c_q{ _c_q }
    {
    }

    __host__ __device__
        tcf operator()(tcf x)
    {
        return tcf(x.real() + c_i,
            a_qi * x.real() + a_qq * x.imag() + c_q);
    }
};

struct millivolts_functor : thrust::binary_function<char, char, tcf>
{
    float scale;

    millivolts_functor(float s) : scale(s) {}

    __host__ __device__ tcf operator()(char i, char q)
    {
        return tcf(static_cast<float>(i) * scale, static_cast<float>(q) * scale);
    }
};

struct field_functor
{
    __host__ __device__
        void operator()(const tcf& x, const tcf& y, tcf& z)
    {
        z += x - y;
    }
};

struct power_functor
{
    __host__ __device__
        void operator()(const tcf& x, const tcf& y, float& z)
    {
        z += thrust::norm(x) - thrust::norm(y);
    }
};


#endif // !DSP_FUNCTORS_CUH