//
// Created by andrei on 3/27/21.
//

#include "dsp.cuh"
#include <cstdio>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <npp.h>
#include <nppcore.h>
#include <nppdefs.h>
#include <npps.h>
#include <complex>
#include <cublas_v2.h>
#include <cmath>
#include <thrust/complex.h>
#include <thrust/transform.h>
#include <thrust/tabulate.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>


inline void check_cufft_error(cufftResult cufft_err, std::string &&msg)
{
#ifdef NDEBUG

    if (cufft_err != CUFFT_SUCCESS)
        throw std::runtime_error(msg);

#endif // NDEBUG
}

inline void check_cublas_error(cublasStatus_t err, std::string &&msg)
{
#ifdef NDEBUG

    if (err != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(msg);

#endif // NDEBUG
}

inline void check_npp_error(NppStatus err, std::string &&msg)
{
#ifdef NDEBUG
    if (err != NPP_SUCCESS)
        throw std::runtime_error(msg);
#endif // NDEBUG
}

// DSP constructor
dsp::dsp(int len, int n, float part) : trace_length{(int)std::round((float)len * part)}, // Length of a signal or noise trace
                                       batch_size{n},                                    // Number of segments in a buffer (same: number of traces in data)
                                       total_length{batch_size * trace_length},
                                       out_size{trace_length * trace_length},
                                       trace1_start{0},       // Start of the signal data
                                       trace2_start{len / 2}, // Start of the noise data
                                       pitch{len},            // Segment length in a buffer
                                       A_gpu(2 * total_length), C_gpu(total_length),
                                       firwin(total_length), // GPU memory for the filtering window
                                       subtraction_trace(total_length, std::complex(0.f)),
                                       downconversion_coeffs(total_length)
{
    // Streams
    for (int i = 0; i < num_streams; i++)
    {
        // Create streams for parallel data processing
        handleError(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        check_npp_error(nppGetStreamContext(&streamContexts[i]), "Npp Error GetStreamContext");
        streamContexts[i].hStream = streams[i];

        // Allocate arrays on GPU for every stream
        gpu_buf[i].resize(2 * total_length);
        gpu_buf2[i].resize(2 * total_length);
        data[i].resize(2 * total_length);
        noise[i].resize(2 * total_length);
        data_calibrated[i].resize(2 * total_length);
        noise_calibrated[i].resize(2 * total_length);
        power[i].resize(2 * total_length);
        field[i].resize(2 * total_length);
        out[i].resize(out_size);

        // Initialize cuFFT plans
        check_cufft_error(cufftPlan1d(&plans[i], trace_length, CUFFT_C2C, batch_size),
                          "Error initializing cuFFT plan\n");

        // Assign streams to cuFFT plans
        check_cufft_error(cufftSetStream(plans[i], streams[i]),
                          "Error assigning a stream to a cuFFT plan\n");

        // Initialize cuBLAS
        check_cublas_error(cublasCreate(&cublas_handles[i]),
                           "Error initializing a cuBLAS handle\n");
        check_cublas_error(cublasCreate(&cublas_handles2[i]),
                           "Error initializing a cuBLAS handle\n");

        // Assign streams to cuBLAS handles
        check_cublas_error(cublasSetStream(cublas_handles[i], streams[i]),
                           "Error assigning a stream to a cuBLAS handle\n");
        check_cublas_error(cublasSetStream(cublas_handles2[i], streams[i]),
                           "Error assigning a stream to a cuBLAS handle\n");
    }
    resetOutput();

    // Allocate GPU memory for the minmax function
    cudaMalloc((void **)(&minfield), sizeof(Npp32f) * 1);
    cudaMalloc((void **)(&maxfield), sizeof(Npp32f) * 1);
    int nBufferSize;
    nppsMinMaxGetBufferSize_32f(2 * total_length, &nBufferSize);
    cudaMalloc((void **)(&minmaxbuffer), nBufferSize);
}

// DSP destructor
dsp::~dsp()
{
    for (int i = 0; i < num_streams; i++)
    {
        // Destroy cuBLAS
        cublasDestroy(cublas_handles[i]);
        cublasDestroy(cublas_handles2[i]);

        // Destroy cuFFT plans
        cufftDestroy(plans[i]);

        // Destroy GPU streams
        handleError(cudaStreamDestroy(streams[i]));
    }

    cudaFree(minfield);
    cudaFree(maxfield);
    cudaFree(minmaxbuffer);

    cudaDeviceReset();
}

// Creates a rectangular window with specified cutoff frequencies for the further usage in a filter
void dsp::setFirwin(float cutoff_l, float cutoff_r, int oversampling)
{
    using namespace std::complex_literals;
    hostvec_c hFirwin(total_length);
    float fs = 1250.f / (float)oversampling;
    int l_idx = (int)std::roundf((float)trace_length / fs * cutoff_l);
    int r_idx = (int)std::roundf((float)trace_length / fs * cutoff_r);
    for (int i = 0; i < total_length; i++)
    {
        int j = i % trace_length;
        hFirwin[i] = ((j < l_idx) || (j > r_idx)) ? 0if : 1.0f + 0if;
    }
    firwin = hFirwin;
}

// Error handler
void dsp::handleError(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::string name = cudaGetErrorName(err);
        std::string text = cudaGetErrorString(err);
        throw std::runtime_error(name + ": " + text);
    }
}

void dsp::createBuffer(uint32_t size)
{
    buffer.resize(size);
}

void dsp::setIntermediateFrequency(float frequency, int oversampling)
{
    const float pi = std::acos(-1);
    float ovs = static_cast<float>(oversampling);
    thrust::tabulate(downconversion_coeffs.begin(), downconversion_coeffs.end(),
        [=] __device__ (int i) {
            float t = 0.8 * ovs * static_cast<float>(i % trace_length);
            return thrust::exp(tcf(0, -2 * pi * frequency * t));
        });
}

void dsp::downconvert(gpuvec_c data, cudaStream_t& stream)
{
    thrust::transform(thrust::cuda::par.on(stream), data.begin(), data.end(), downconversion_coeffs.begin(), data.begin(), thrust::multiplies());
}

struct downconv_functor : thrust::unary_function<tcf, tcf>
{
    float a_qi, a_qq;
    float c_i, c_q;

    downconv_functor(float _a_qi, float _a_qq,
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

void dsp::setDownConversionCalibrationParameters(float r, float phi,
    float offset_i, float offset_q)
{
    a_ii = 1;
    a_qi = std::tan(phi);
    a_qq = 1 / (r * std::cos(phi));
    c_i = offset_i;
    c_q = offset_q;
}

// Applies down-conversion calibration to traces
void dsp::applyDownConversionCalibration(gpuvec_c& data, gpuvec_c& data_calibrated, cudaStream_t &stream)
{
    auto sync_exec_policy = thrust::cuda::par.on(stream);
    thrust::transform(sync_exec_policy, data.begin(), data.end(), data_calibrated.begin(), downconv_functor(a_qi, a_qq, c_i, c_q));
}

char *dsp::getBufferPointer()
{
    return &buffer[0];
}

// Fills with zeros the arrays for cumulative field and power in the GPU memory
void dsp::resetOutput()
{
    using namespace std::complex_literals;
    for (int i = 0; i < num_streams; i++)
    {
        thrust::fill(out[i].begin(), out[i].end(), 0if);
        thrust::fill(field[i].begin(), field[i].end(), 0.f);
        thrust::fill(power[i].begin(), power[i].end(), 0.f);
    }
}

void dsp::compute(const hostbuf &buffer)
{
    const int stream_num = semaphore;
    switchStream();
    loadDataToGPUwithPitchAndOffset(buffer, gpu_buf[stream_num], pitch, trace1_start, stream_num);
    loadDataToGPUwithPitchAndOffset(buffer, gpu_buf2[stream_num], pitch, trace2_start, stream_num);
    convertDataToMillivolts(data[stream_num], gpu_buf[stream_num], streams[stream_num]);
    convertDataToMillivolts(noise[stream_num], gpu_buf2[stream_num], streams[stream_num]);
    applyDownConversionCalibration(data[stream_num], data_calibrated[stream_num], streams[stream_num]);
    applyDownConversionCalibration(noise[stream_num], noise_calibrated[stream_num], streams[stream_num]);
    applyFilter(data_calibrated[stream_num], firwin, stream_num);
    applyFilter(noise_calibrated[stream_num], firwin, stream_num);
    downconvert(data_calibrated[stream_num], streams[stream_num]);
    downconvert(noise_calibrated[stream_num], streams[stream_num]);
    subtractDataFromOutput(subtraction_trace, data_calibrated[stream_num], streams[stream_num]);

    calculateField(data_calibrated[stream_num], noise_calibrated[stream_num],
        field[stream_num], streams[stream_num]);
    calculatePower(data_calibrated[stream_num], noise_calibrated[stream_num],
        power[stream_num], streams[stream_num]);
    calculateG1(data_calibrated[stream_num], noise_calibrated[stream_num],
        out[stream_num], cublas_handles[stream_num]);
}

// This function uploads data from the specified section of a buffer array to the GPU memory
void dsp::loadDataToGPUwithPitchAndOffset(
    const hostbuf &buffer, gpubuf &gpu_buf,
    size_t pitch, size_t offset, int stream_num)
{
    size_t width = 2 * size_t(trace_length) * sizeof(Npp8s);
    size_t src_pitch = 2 * pitch * sizeof(Npp8s);
    size_t dst_pitch = width;
    size_t shift = 2 * offset;
    handleError(cudaMemcpy2DAsync(get(gpu_buf), dst_pitch,
                                  get(buffer) + shift, src_pitch, width, batch_size,
                                  cudaMemcpyHostToDevice, streams[stream_num]));
}

struct millivolts_functor : thrust::unary_function<char, float>
{
    float scale;

    millivolts_functor(float s) : scale(s) {}

    __host__ __device__ float operator()(char v)
    {
        return static_cast<float>(v) * scale;
    }
};

// Converts bytes into 32-bit floats with mV dimensionality
void dsp::convertDataToMillivolts(gpuvec data, gpubuf gpu_buf, cudaStream_t &stream)
{
    thrust::transform(thrust::cuda::par.on(stream),
        gpu_buf.begin(), gpu_buf.end(), data.begin(), millivolts_functor(scale));
}

// Applies the filter with the specified window to the data using FFT convolution
void dsp::applyFilter(gpuvec_c &data, const gpuvec_c &window, int stream_num)
{
    // Step 1. Take FFT of each segment
    cufftComplex *cufft_data = reinterpret_cast<cufftComplex *>(get(data));
    auto cufftstat = cufftExecC2C(plans[stream_num], cufft_data, cufft_data, CUFFT_FORWARD);
    check_cufft_error(cufftstat, "Error executing cufft");
    // Step 2. Multiply each segment by a window
    auto npp_status = nppsMul_32fc_I_Ctx(to_Npp32fc_p(get(window)), to_Npp32fc_p(get(data)),
        total_length, streamContexts[stream_num]);
    check_npp_error(npp_status, "Error multiplying by window #" + std::to_string(npp_status));
    // Step 3. Take inverse FFT of each segment
    cufftExecC2C(plans[stream_num], cufft_data, cufft_data, CUFFT_INVERSE);
    check_cufft_error(cufftstat, "Error executing cufft");
    // Step 4. Normalize the FFT for the output to equal the input
    Npp32fc denominator = { trace_length, 0.f };
    npp_status = nppsDivC_32fc_I_Ctx(denominator, to_Npp32fc_p(get(data)), total_length, streamContexts[stream_num]);
    check_npp_error(npp_status, "Error dividing by denomintator #" + std::to_string(npp_status));
}

// Sums newly processed data with previous data for averaging
void dsp::addDataToOutput(gpuvec_c &data, gpuvec_c &output, cudaStream_t& stream)
{
    thrust::transform(thrust::cuda::par.on(stream), output.begin(), output.end(), data.begin(),
        output.begin(), thrust::plus<tcf>());
}

// Subtracts newly processed data from previous data
void dsp::subtractDataFromOutput(gpuvec_c& data, gpuvec_c& output, cudaStream_t &stream)
{
    thrust::transform(thrust::cuda::par.on(stream), output.begin(), output.end(), data.begin(),
        output.begin(), thrust::minus<tcf>());
}


struct field_functor
{
    __host__ __device__
    void operator()(const tcf& x, const tcf& y, tcf& z)
    {
        z += x - y;
    }
};

// Calculates the field from the data in the GPU memory
void dsp::calculateField(gpuvec_c data, gpuvec_c noise, gpuvec_c output, cudaStream_t &stream)
{
    thrust::for_each(thrust::cuda::par.on(stream),
        thrust::make_zip_iterator(data.begin(), noise.begin(), output.begin()),
        thrust::make_zip_iterator(data.end(), noise.end(), output.end()),
        thrust::make_zip_function(field_functor()));
}

struct power_functor
{
    __host__ __device__
    void operator()(const tcf& x, const tcf& y, float& z)
    {
        z += thrust::norm(x) - thrust::norm(y);
    }
};

// Calculates the power from the data in the GPU memory
void dsp::calculatePower(gpuvec_c data, gpuvec_c noise, gpuvec output, cudaStream_t& stream)
{
    thrust::for_each(thrust::cuda::par.on(stream),
        thrust::make_zip_iterator(data.begin(), noise.begin(), output.begin()),
        thrust::make_zip_iterator(data.end(), noise.end(), output.end()),
        thrust::make_zip_function(power_functor()));
}

void dsp::calculateG1(gpuvec_c data, gpuvec_c noise, gpuvec_c output, cublasHandle_t &handle)
{
    using namespace std::string_literals;

    const float alpha_data = 1;   // this alpha multiplies the result to be added to the output
    const float alpha_noise = -1; // this alpha multiplies the result to be added to the output
    const float beta = 1;
    // Compute correlation for the signal and add it to the output
    auto cublas_status = cublasCherk(handle,
                                     CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, trace_length, batch_size,
                                     &alpha_data, reinterpret_cast<cuComplex *>(get(data)), trace_length,
                                     &beta, reinterpret_cast<cuComplex *>(get(output)), trace_length);
    // Check for errors
    check_cublas_error(cublas_status,
        "Error of rank-1 update (data) with code #"s + std::to_string(cublas_status));
    // Compute correlation for the noise and subtract it from the output
    cublas_status = cublasCherk(handle,
                                CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, trace_length, batch_size,
                                &alpha_noise, reinterpret_cast<cuComplex *>(get(noise)), trace_length,
                                &beta, reinterpret_cast<cuComplex *>(get(output)), trace_length);
    // Check for errors
    check_cublas_error(cublas_status,
        "Error of rank-1 update (noise) with code #"s + std::to_string(cublas_status));
}

// Returns the average value
void dsp::getCorrelator(hostvec_c &result)
{
    this->handleError(cudaDeviceSynchronize());
    for (int i = 1; i < num_streams; i++)
        thrust::transform(out[i].begin(), out[i].end(), out[0].begin(), thrust::plus<tcf>());
    result = out[0];
}

// Returns the cumulative power
void dsp::getCumulativePower(hostvec &result)
{
    this->handleError(cudaDeviceSynchronize());
    for (int i = 1; i < num_streams; i++)
        thrust::transform(power[i].begin(), power[i].end(), power[0].begin(), thrust::plus<float>());
    result = power[0];
}

// Returns the cumulative field
void dsp::getCumulativeField(hostvec_c &result)
{
    this->handleError(cudaDeviceSynchronize());
    for (int i = 1; i < num_streams; i++)
        thrust::transform(field[i].begin(), field[i].end(), field[0].begin(), thrust::plus<tcf>());
    result = field[0];
}

// Returns the useful length of the data in a segment
// (trace is assumed complex valued)
int dsp::getTraceLength()
{
    return trace_length;
}

// Returns the total length of the data comprised of several segments
// (trace is assumed complex valued)
int dsp::getTotalLength()
{
    return total_length;
}

int dsp::getOutSize()
{
    return out_size;
}

void dsp::setAmplitude(int ampl)
{
    scale = static_cast<float>(ampl) / 128.f;
}

void dsp::setSubtractionTrace(hostvec_c &trace)
{
    subtraction_trace = trace;
}

void dsp::getSubtractionTrace(hostvec_c &trace)
{
    trace = subtraction_trace;
}

void dsp::resetSubtractionTrace()
{
    thrust::fill(subtraction_trace.begin(), subtraction_trace.end(), 0.f);
}
