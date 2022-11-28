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

inline void check_cufft_error(cufftResult cufft_err, std::string &&msg)
{
    if (cufft_err != CUFFT_SUCCESS)
        throw std::runtime_error(msg);
}

inline void check_cublas_error(cublasStatus_t err, std::string &&msg)
{
    if (err != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(msg);
}

inline void check_npp_error(NppStatus err, std::string &&msg)
{
    if (err != NPP_SUCCESS)
        throw std::runtime_error(msg);
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

// Initializes matrices and arrays required for down-conversion calibration with given parameters
void dsp::setDownConversionCalibrationParameters(float r, float phi, float offset_i, float offset_q)
{
    // Filling A-matrix (4x4) in Fortran-style row order
    float a_ii = 1;
    float a_qi = std::tan(phi);
    float a_qq = 1 / (r * std::cos(phi));
    hostvec A_mat(cal_mat_size);
    if (cal_mat_size == 4)
    {
        A_mat[0] = a_ii;
        A_mat[1] = a_qi;
        A_mat[2] = 0.f;
        A_mat[3] = a_qq;
    }
    else
    {
        for (int i = 0; i < cal_mat_size; i++)
            A_mat[i] = 0.f;
        A_mat[0] = a_ii;
        A_mat[1] = a_qi;
        A_mat[5] = a_qq;
        A_mat[10] = a_ii;
        A_mat[11] = a_qi;
        A_mat[15] = a_qq;
    }

    // Creating an array with repeated matrices
    hostvec A_host(2 * total_length);
    for (int i = 0; i < 2 * total_length; i += cal_mat_size)
        for (int j = 0; j < cal_mat_size; j++)
            A_host[i + j] = A_mat[j];

    // Transferring it onto GPU memory
    A_gpu = A_host;

    // Estimating the number of matrix multiplications
    batch_count = 2 * total_length / cal_mat_size;

    // Filling the offsets array C_gpu
    thrust::fill(C_gpu.begin(), C_gpu.end(), Npp32fc{offset_i, offset_q});

    // cleaning
    cudaDeviceSynchronize();
}

typedef thrust::complex<float> tcf;
struct downconv_functor : thrust::unary_function<tcf, tcf>
{
    float a_ii, a_qi, a_qq;
    float c_i, c_q;

    downconv_functor(float _a_qi, float _a_qq,
                     float _c_i, float _c_q) : a_ii{_a_ii}, a_qi{_a_qi},
                                               a_qq{_a_qq},
                                               c_i{_c_i}, c_q{_c_q}
    {
    }

    __host__ __device__ float operator()(tcf x)
    {
        return tcf(a_ii * x.real() + c_i,
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
    using namespace std::complex_literals;
    hostvec_c dcov_host(total_length);

    const float pi = std::acos(-1);

    float ovs = static_cast<float>(oversampling);
    float t = 0;

    for (int j = 0; j < batch_size; j++)
    {
        for (int k = 0; k < trace_length; k++)
        {
            t = 0.8 * k * ovs;
            dcov_host[j * trace_length + k] = std::exp(-2if * pi * frequency * t);
        }
    }
    downconversion_coeffs = dcov_host;
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

void dsp::downconvert(gpuvec data, int stream_num)
{
    nppsMul_32fc_I_Ctx(get(downconversion_coeffs), get(data), total_length, streamContexts[stream_num]);
}

void dsp::compute(const hostbuf &buffer)
{
    const int stream_num = semaphore;
    switchStream();
    loadDataToGPUwithPitchAndOffset(buffer, gpu_buf[stream_num], pitch, trace1_start, stream_num);
    loadDataToGPUwithPitchAndOffset(buffer, gpu_buf2[stream_num], pitch, trace2_start, stream_num);
    convertDataToMilivolts(data[stream_num], gpu_buf[stream_num], stream_num);
    convertDataToMilivolts(noise[stream_num], gpu_buf2[stream_num], stream_num);
    applyDownConversionCalibration(data[stream_num], data_calibrated[stream_num], stream_num);
    applyDownConversionCalibration(noise[stream_num], noise_calibrated[stream_num], stream_num);
    applyFilter(data_calibrated[stream_num], firwin, stream_num);
    applyFilter(noise_calibrated[stream_num], firwin, stream_num);
    downconvert(data_calibrated[stream_num], stream_num);
    downconvert(noise_calibrated[stream_num], stream_num);
    subtractDataFromOutput(subtraction_trace, data_calibrated[stream_num], stream_num);

    calculateField(stream_num);
    calculateG1(stream_num);
    calculatePower(stream_num);
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

// Converts bytes into 32-bit floats with mV dimensionality
void dsp::convertDataToMilivolts(gpuvec data, gpubuf gpu_buf, int stream_num)
{
    // convert from int8 to float32
    check_npp_error(nppsConvert_8s32f_Ctx(get(gpu_buf), get(data),
                                          2 * total_length, streamContexts[stream_num]),
                    "error converting int8 to float32");
    // multiply by a constant in order to convert into mV
    check_npp_error(nppsMulC_32fc_I_Ctx(scale, reinterpret_cast<Npp32fc *>(get(data)), total_length, streamContexts[stream_num]),
                    "error when scaling to mV");
}

// Applies down-conversion calibration to traces
void dsp::applyDownConversionCalibration(gpuvec_c &data, gpuvec_c &data_calibrated, int stream_num)
{
    auto sync_exec_policy = thrust::cuda::par.on(streams[stream_num]);
    thrust::transform(sync_exec_policy, data.begin(), data.end(), data_calibrated.begin(), downconv_functor(a_ii, a_qi, a_qq, c_i, c_q));
}

// Applies the filter with the specified window to the data using FFT convolution
void dsp::applyFilter(gpuvec data, const gpuvec window, int stream_num)
{
    // Step 1. Take FFT of each segment
    cufftComplex *cufft_data = reinterpret_cast<cufftComplex *>(get(data));
    cufftExecC2C(plans[stream_num], cufft_data, cufft_data, CUFFT_FORWARD);
    // Step 2. Multiply each segment by a window
    auto npp_status = nppsMul_32fc_I_Ctx(get(window), get(data), total_length, streamContexts[stream_num]);
    if (npp_status != NPP_SUCCESS)
    {
        throw std::runtime_error("Error multiplying by window #" + std::to_string(npp_status));
    }
    // Step 3. Take inverse FFT of each segment
    cufftExecC2C(plans[stream_num], cufft_data, cufft_data, CUFFT_INVERSE);
    // Step 4. Normalize the FFT for the output to equal the input
    Npp32fc denominator;
    denominator.re = (Npp32f)trace_length;
    denominator.im = 0.f;
    nppsDivC_32fc_I_Ctx(denominator, get(data), total_length, streamContexts[stream_num]);
}

// Sums newly processed data with previous data for averaging
void dsp::addDataToOutput(Npp32f *data, Npp32f *output, int stream_num)
{
    auto status = nppsAdd_32f_I_Ctx(data, output,
                                    2 * total_length, streamContexts[stream_num]);
    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error("Error adding new data to previous #" + std::to_string(status));
    }
}

// Subtracts newly processed data from previous data
void dsp::subtractDataFromOutput(Npp32f *data, Npp32f *output, int stream_num)
{
    auto status = nppsSub_32f_I_Ctx(data, output,
                                    2 * total_length, streamContexts[stream_num]);
    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error("Error adding new data to previous #" + std::to_string(status));
    }
}

// Calculates the field from the data in the GPU memory
void dsp::calculateField(int stream_num)
{
    // Add signal field to the cumulative field
    this->addDataToOutput(data_calibrated[stream_num], field[stream_num], stream_num);
    // Subtract noise field from the cumulative field
    this->subtractDataFromOutput(noise_calibrated[stream_num], field[stream_num], stream_num);
    this->getMinMax(field[stream_num], stream_num);
}

// Calculates the power from the data in the GPU memory
void dsp::calculatePower(int stream_num)
{
    // Calculate squared signal
    auto status = nppsSqr_32f_I_Ctx(data_calibrated[stream_num], 2 * total_length, streamContexts[stream_num]);
    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error("Error squaring data #" + std::to_string(status));
    }
    // Calculate squared noise
    status = nppsSqr_32f_I_Ctx(noise_calibrated[stream_num], 2 * total_length, streamContexts[stream_num]);
    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error("Error squaring data #" + std::to_string(status));
    }
    // Add signal power to the cumulative power
    this->addDataToOutput(data_calibrated[stream_num], power[stream_num], stream_num);
    // Subtract noise power from the cumulative power
    this->subtractDataFromOutput(noise_calibrated[stream_num], power[stream_num], stream_num);
}

void dsp::calculateG1(int stream_num)
{
    using namespace std::string_literals;

    const float alpha_data = 1;   // this alpha multiplies the result to be added to the output
    const float alpha_noise = -1; // this alpha multiplies the result to be added to the output
    const float beta = 1;
    // Compute correlation for the signal and add it to the output
    auto cublas_status = cublasCherk(cublas_handles2[stream_num],
                                     CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, trace_length, batch_size,
                                     &alpha_data, reinterpret_cast<cuComplex *>(data_calibrated[stream_num]), trace_length,
                                     &beta, reinterpret_cast<cuComplex *>(out[stream_num]), trace_length);
    // Check for errors
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("Error of rank-1 update (data) with code #"s + std::to_string(cublas_status));
    }
    // Compute correlation for the noise and subtract it from the output
    cublas_status = cublasCherk(cublas_handles2[stream_num],
                                CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, trace_length, batch_size,
                                &alpha_noise, reinterpret_cast<cuComplex *>(noise_calibrated[stream_num]), trace_length,
                                &beta, reinterpret_cast<cuComplex *>(out[stream_num]), trace_length);
    // Check for errors
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("Error of rank-1 update (noise) with code #"s + std::to_string(cublas_status));
    }
}

// Returns the average value
void dsp::getCorrelator(std::vector<std::complex<float>> &result)
{
    this->handleError(cudaDeviceSynchronize());
    for (int i = 1; i < num_streams; i++)
        nppsAdd_32fc_I(out[i], out[0], out_size);
    this->handleError(cudaMemcpy(result.data(), out[0],
                                 out_size * sizeof(Npp32fc), cudaMemcpyDeviceToHost));
}

// Returns the cumulative power
void dsp::getCumulativePower(std::vector<std::complex<float>> &result)
{
    this->handleError(cudaDeviceSynchronize());
    for (int i = 1; i < num_streams; i++)
        nppsAdd_32f_I(power[i], power[0], 2 * total_length);
    this->handleError(cudaMemcpy(result.data(), power[0],
                                 2 * total_length * sizeof(Npp32f), cudaMemcpyDeviceToHost));
}

// Returns the cumulative field
void dsp::getCumulativeField(std::vector<std::complex<float>> &result)
{
    this->handleError(cudaDeviceSynchronize());
    for (int i = 1; i < num_streams; i++)
        nppsAdd_32f_I(field[i], field[0], 2 * total_length);
    this->handleError(cudaMemcpy(result.data(), field[0],
                                 2 * total_length * sizeof(Npp32f), cudaMemcpyDeviceToHost));
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

// Get mininimal and maximal values from an array for the debug purposes
void dsp::getMinMax(Npp32f *data, int stream_num)
{
    nppsMinMax_32f_Ctx(data, 2 * total_length, minfield, maxfield, minmaxbuffer, streamContexts[stream_num]);
}

void dsp::setAmplitude(int ampl)
{
    scale = Npp32fc{static_cast<float>(ampl) / 128.f, 0.f};
}

void dsp::setSubtractionTrace(std::vector<std::complex<float>> &trace)
{
    this->handleError(cudaMemcpy((void *)subtraction_trace, (void *)trace.data(),
                                 total_length * sizeof(Npp32fc), cudaMemcpyHostToDevice));
}

void dsp::getSubtractionTrace(std::vector<std::complex<float>> &trace)
{
    this->handleError(cudaMemcpy((void *)trace.data(), (void *)subtraction_trace,
                                 total_length * sizeof(Npp32fc), cudaMemcpyDeviceToHost));
}

void dsp::resetSubtractionTrace()
{
    nppsSet_32fc(Npp32fc{0.f, 0.f}, subtraction_trace, total_length);
}
