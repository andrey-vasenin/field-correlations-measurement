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


// DSP constructor
dsp::dsp(int len, int n, float part)
{
    trace_length = (int)std::round((float)len * part); // Length of a signal or noise trace
    batch_size = n; // Number of segments in a buffer (same: number of traces in data)
    total_length = batch_size * trace_length;
    //out_size = trace_length * (trace_length + 1) / 2;
    out_size = trace_length * trace_length;

    trace1_start = 0; // Start of the signal data
    trace2_start = len / 2; // Start of the noise data
    pitch = len; // Segment length in a buffer

    // allocating down-conversion calibration vectors
    this->handleError(cudaMalloc(&A_gpu, 2 * total_length * sizeof(float)));
    this->handleError(cudaMalloc(&C_gpu, total_length * sizeof(Npp32fc)));

    // Streams
    for (int i = 0; i < num_streams; i++)
    {
        // Create streams for parallel data processing
        this->handleError(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        auto nppStatus = nppGetStreamContext(&streamContexts[i]);
        if (nppStatus != NPP_SUCCESS)
            throw std::runtime_error("Couldn't get stream context");
        streamContexts[i].hStream = streams[i];

        // Allocate arrays on GPU for every stream
        gpu_buf[i] = nppsMalloc_8s(2 * total_length);
        gpu_buf2[i] = nppsMalloc_8s(2 * total_length);
        data[i] = nppsMalloc_32f(2 * total_length);
        noise[i] = nppsMalloc_32f(2 * total_length);
        data_calibrated[i] = nppsMalloc_32f(2 * total_length);
        noise_calibrated[i] = nppsMalloc_32f(2 * total_length);
        power[i] = nppsMalloc_32f(2 * total_length);
        field[i] = nppsMalloc_32f(2 * total_length);
        out[i] = nppsMalloc_32fc(out_size);

        // Initialize the result arrays with zeros
        nppsSet_32f(0.f, field[i], 2 * total_length);
        nppsSet_32f(0.f, power[i], 2 * total_length);
        nppsSet_32fc(Npp32fc{0.f, 0.f}, out[i], out_size);

        // Initialize cuFFT plans
        auto cufft_err = cufftPlan1d(&plans[i], trace_length, CUFFT_C2C, batch_size);
        if (cufft_err != CUFFT_SUCCESS)
            throw std::runtime_error("Error initializing cuFFT plan\n");

        // Assign streams to cuFFT plans
        cufft_err = cufftSetStream(plans[i], streams[i]);
        if (cufft_err != CUFFT_SUCCESS)
            throw std::runtime_error("Error assigning a stream to a cuFFT plan\n");

        // Initialize cuBLAS
        auto cublas_err = cublasCreate(&cublas_handles[i]);
        if (cublas_err != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Error initializing a cuBLAS handle\n");
        
        cublas_err = cublasCreate(&cublas_handles2[i]);
        if (cublas_err != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Error initializing a cuBLAS handle\n");

        // Assign streams to cuBLAS handles
        cublas_err = cublasSetStream(cublas_handles[i], streams[i]);
        if (cublas_err != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Error assigning a stream to a cuBLAS handle\n");
        cublas_err = cublasSetStream(cublas_handles2[i], streams[i]);
        if (cublas_err != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Error assigning a stream to a cuBLAS handle\n");
    }
    // Allocate GPU memory for the filtering window
    firwin = nppsMalloc_32fc(total_length);

    // Allocate GPU memory for the minmax function
    cudaMalloc((void**)(&minfield), sizeof(Npp32f) * 1);
    cudaMalloc((void**)(&maxfield), sizeof(Npp32f) * 1);
    int nBufferSize;
    nppsMinMaxGetBufferSize_32f(2 * total_length, &nBufferSize);
    cudaMalloc((void**)(&minmaxbuffer), nBufferSize);

    // Allocate GPU memory for the subtraction trace
    subtraction_trace = nppsMalloc_32fc(total_length);
    Npp32fc initval{ 0, 0 };
    nppsSet_32fc(initval, subtraction_trace, total_length);

    // Allocate GPU memory for the downconversion coefficients
    downconversion_coeffs = nppsMalloc_32fc(total_length);
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
        this->handleError(cudaStreamDestroy(streams[i]));

        // Free GPU memory
        nppsFree(data[i]);
        nppsFree(data_complex[i]);
        nppsFree(data_calibrated[i]);
        nppsFree(gpu_buf[i]);
        nppsFree(power[i]);
        nppsFree(field[i]);
        nppsFree(out[i]);
        if (gpu_buf2[i] != nullptr)
            nppsFree(gpu_buf2[i]);
        if (noise[i] != nullptr)
            nppsFree(noise[i]);
        if (noise_calibrated[i] != nullptr)
            nppsFree(noise_calibrated[i]);
    }
    nppsFree(firwin);
    cudaFree(A_gpu);
    cudaFree(C_gpu);

    cudaFree(minfield);
    cudaFree(maxfield);
    cudaFree(minmaxbuffer);

    nppsFree(subtraction_trace);
    nppsFree(downconversion_coeffs);

    this->deleteBuffer();

    cudaDeviceReset();
}

// Creates a rectangular window with specified cutoff frequencies for the further usage in a filter
void dsp::setFirwin(float cutoff_l, float cutoff_r, int oversampling)
{
    using namespace std::complex_literals;
    auto hFirwin = new std::complex<float>[total_length];
    float fs = 1250.f / (float)oversampling;
    int l_idx = (int)std::roundf((float)trace_length / fs * cutoff_l);
    int r_idx = (int)std::roundf((float)trace_length / fs * cutoff_r);;
    for (int i = 0; i < total_length; i++)
    {
        int j = i % trace_length;
        hFirwin[i] = ((j < l_idx) || (j > r_idx)) ? 0if : 1.0f + 0if;
    }
    this->handleError(cudaMemcpy((void*)firwin, (void*)hFirwin,
        total_length * sizeof(hFirwin), cudaMemcpyHostToDevice));
    delete[] hFirwin;
}

// Initializes matrices and arrays required for down-conversion calibration with given parameters
void dsp::setDownConversionCalibrationParameters(float r, float phi, float offset_i, float offset_q)
{
    // Filling A-matrix (4x4) in Fortran-style row order
    float a_ii = 1;
    float a_qi = std::tan(phi);
    float a_qq = 1 / (r * std::cos(phi));
    float A_mat[cal_mat_size];
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
    std::vector<float> A_host(2 * total_length);
    for (int i = 0; i < 2 * total_length; i += cal_mat_size)
        for (int j = 0; j < cal_mat_size; j++)
            A_host[i + j] = A_mat[j];

    // Transferring it onto GPU memory
    this->handleError(cudaMemcpyAsync(A_gpu, A_host.data(), A_host.size() * sizeof(float),
        cudaMemcpyHostToDevice, streams[0]));

    // Estimating the number of matrix multiplications
    batch_count = 2 * total_length / cal_mat_size;

    // Filling the offsets array C_gpu
    Npp32fc init_val{ offset_i, offset_q };
    nppsSet_32fc(init_val, C_gpu, total_length);

    // cleaning
    cudaDeviceSynchronize();
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

void dsp::createBuffer(size_t size)
{
    this->handleError(cudaMallocHost((void**)&buffer, size));
}

void dsp::setIntermediateFrequency(float frequency, int oversampling)
{
    using namespace std::complex_literals;
    std::vector<std::complex<float>> dcov_host(total_length);
    
    const float pi = std::acos(-1);

    float ovs = static_cast<float>(oversampling);
    float t = 0;

    for (int j = 0; j < batch_size; j++)
    {
        for (int k = 0; k < trace_length; k++)
        {
            t = 0.8 * k * ovs;
            dcov_host[j * trace_length + k] = std::exp(-2if * pi* frequency* t);
        }
    }
    this->handleError(cudaMemcpy(reinterpret_cast<void*>(downconversion_coeffs),
        reinterpret_cast<void*>(dcov_host.data()),
        total_length * sizeof(Npp32fc), cudaMemcpyHostToDevice));
}

char* dsp::getBufferPointer()
{
    return buffer;
}

void dsp::deleteBuffer()
{
    this->handleError(cudaFreeHost(buffer));
}

// Fills with zeros the arrays for cumulative field and power in the GPU memory
void dsp::resetOutput()
{
    for (int i = 0; i < num_streams; i++)
    {
        nppsSet_32fc(Npp32fc{ 0.f , 0.f}, out[i], out_size);
        nppsSet_32f(0.f, field[i], 2 * total_length);
        nppsSet_32f(0.f, power[i], 2 * total_length);
    }
}

void dsp::downconvert(Npp32fc* data, int stream_num)
{
    nppsMul_32fc_I_Ctx(downconversion_coeffs, data, total_length, streamContexts[stream_num]);
}

void dsp::compute(const char* buffer)
{
    const int stream_num = semaphore;
    this->switchStream();
    this->loadDataToGPUwithPitchAndOffset(buffer, gpu_buf[stream_num], pitch, trace1_start, stream_num);
    this->loadDataToGPUwithPitchAndOffset(buffer, gpu_buf2[stream_num], pitch, trace2_start, stream_num);
    this->convertDataToMilivolts(data[stream_num], gpu_buf[stream_num], stream_num);
    this->convertDataToMilivolts(noise[stream_num], gpu_buf2[stream_num], stream_num);
    this->applyDownConversionCalibration(data[stream_num], data_calibrated[stream_num], stream_num);
    this->applyDownConversionCalibration(noise[stream_num], noise_calibrated[stream_num], stream_num);
    this->applyFilter(reinterpret_cast<Npp32fc*>(data_calibrated[stream_num]), firwin, stream_num);
    this->applyFilter(reinterpret_cast<Npp32fc*>(noise_calibrated[stream_num]), firwin, stream_num);
    this->downconvert(reinterpret_cast<Npp32fc*>(data_calibrated[stream_num]), stream_num);
    this->downconvert(reinterpret_cast<Npp32fc*>(noise_calibrated[stream_num]), stream_num);
    this->subtractDataFromOutput(reinterpret_cast<Npp32f*>(subtraction_trace), data_calibrated[stream_num], stream_num);
    
    this->calculateField(stream_num);
    this->calculateG1(stream_num);
    this->calculatePower(stream_num);
}

// This function uploads data from the specified section of a buffer array to the GPU memory
void dsp::loadDataToGPUwithPitchAndOffset(
    const char* buffer, Npp8s* gpu_buf,
    size_t pitch, size_t offset, int stream_num)
{
    size_t width = 2 * size_t(trace_length) * sizeof(Npp8s);
    size_t src_pitch = 2 * pitch * sizeof(Npp8s);
    size_t dst_pitch = width;
    size_t shift = 2 * offset;
    auto err = cudaMemcpy2DAsync(gpu_buf, dst_pitch,
        buffer + shift, src_pitch, width, batch_size,
        cudaMemcpyHostToDevice, streams[stream_num]);
    this->handleError(err);
}

// Converts bytes into 32-bit floats with mV dimensionality
void dsp::convertDataToMilivolts(Npp32f* data, Npp8s* gpu_buf, int stream_num)
{
    // convert from int8 to float32
    auto status = nppsConvert_8s32f_Ctx(gpu_buf, data,
        2 * total_length, streamContexts[stream_num]);
    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error("error converting int8 to float32 #" + std::to_string(status));
    }
    // reinterpret interleaved channel samples as interleaved real and imaginary floats (basically do nothing)
    data_complex[stream_num] = reinterpret_cast<Npp32fc*>(data);
    // multiply by a constant in order to convert into mV
    status = nppsMulC_32fc_I_Ctx(scale, data_complex[stream_num], total_length, streamContexts[stream_num]);
    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error("error when scaling to mV #" + std::to_string(status));
    }
}

// Applies down-conversion calibration to traces
void dsp::applyDownConversionCalibration(Npp32f* data, Npp32f* data_calibrated, int stream_num)
{
    // Subtract offsets
    nppsSub_32fc_I_Ctx(C_gpu, data_complex[stream_num], total_length, streamContexts[stream_num]);

    // Apply rotation
    cublasGemmStridedBatchedEx(cublas_handles[stream_num],
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        cal_mat_side, cal_mat_side, cal_mat_side,
        &alpha,
        A_gpu, CUDA_R_32F, cal_mat_side, cal_mat_size,
        reinterpret_cast<float*>(data), CUDA_R_32F, cal_mat_side, cal_mat_size,
        &beta,
        reinterpret_cast<float*>(data_calibrated), CUDA_R_32F, cal_mat_side,
        cal_mat_size,
        batch_count,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        //                              CUBLAS_GEMM_ALGO13_TENSOR_OP);
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Applies the filter with the specified window to the data using FFT convolution
void dsp::applyFilter(Npp32fc* data, const Npp32fc* window, int stream_num)
{
    // Step 1. Take FFT of each segment
    cufftComplex* cufft_data = reinterpret_cast<cufftComplex*>(data);
    cufftExecC2C(plans[stream_num], cufft_data, cufft_data, CUFFT_FORWARD);
    // Step 2. Multiply each segment by a window
    auto npp_status = nppsMul_32fc_I_Ctx(window, data, total_length, streamContexts[stream_num]);
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
    nppsDivC_32fc_I_Ctx(denominator, data, total_length, streamContexts[stream_num]);
}

// Sums newly processed data with previous data for averaging
void dsp::addDataToOutput(Npp32f* data, Npp32f* output, int stream_num)
{
    auto status = nppsAdd_32f_I_Ctx(data, output,
        2 * total_length, streamContexts[stream_num]);
    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error("Error adding new data to previous #" + std::to_string(status));
    }
}

// Subtracts newly processed data from previous data
void dsp::subtractDataFromOutput(Npp32f* data, Npp32f* output, int stream_num)
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

    const float alpha_data = 1;  // this alpha multiplies the result to be added to the output
    const float alpha_noise = -1;  // this alpha multiplies the result to be added to the output
    const float beta = 1;
    // Compute correlation for the signal and add it to the output
    auto cublas_status = cublasCherk(cublas_handles2[stream_num],
        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, trace_length, batch_size,
        &alpha_data, reinterpret_cast<cuComplex*>(data_calibrated[stream_num]), trace_length,
        &beta, reinterpret_cast<cuComplex*>(out[stream_num]), trace_length);
    // Check for errors
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("Error of rank-1 update (data) with code #"s
            + std::to_string(cublas_status));
    }
    // Compute correlation for the noise and subtract it from the output
    cublas_status = cublasCherk(cublas_handles2[stream_num],
        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, trace_length, batch_size,
        &alpha_noise, reinterpret_cast<cuComplex*>(noise_calibrated[stream_num]), trace_length,
        &beta, reinterpret_cast<cuComplex*>(out[stream_num]), trace_length);
    // Check for errors
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("Error of rank-1 update (noise) with code #"s
            + std::to_string(cublas_status));
    }
}

// Returns the average value
void dsp::getCorrelator(std::vector <std::complex<float>>& result)
{
    this->handleError(cudaDeviceSynchronize());
    for (int i = 1; i < num_streams; i++)
        nppsAdd_32fc_I(out[i], out[0], out_size);
    this->handleError(cudaMemcpy(result.data(), out[0],
        out_size * sizeof(Npp32fc), cudaMemcpyDeviceToHost));
}

// Returns the cumulative power
void dsp::getCumulativePower(std::vector <std::complex<float>>& result)
{
    this->handleError(cudaDeviceSynchronize());
    for (int i = 1; i < num_streams; i++)
        nppsAdd_32f_I(power[i], power[0], 2 * total_length);
    this->handleError(cudaMemcpy(result.data(), power[0],
        2 * total_length * sizeof(Npp32f), cudaMemcpyDeviceToHost));
}

// Returns the cumulative field
void dsp::getCumulativeField(std::vector <std::complex<float>>& result)
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
void dsp::getMinMax(Npp32f* data, int stream_num)
{
    nppsMinMax_32f_Ctx(data, 2 * total_length, minfield, maxfield, minmaxbuffer, streamContexts[stream_num]);
}

void dsp::setAmplitude(int ampl)
{
    scale = Npp32fc{ static_cast<float>(ampl) / 128.f, 0.f };
}

void dsp::setSubtractionTrace(std::vector<std::complex<float>>& trace)
{
    this->handleError(cudaMemcpy((void*)subtraction_trace, (void*)trace.data(),
        total_length * sizeof(Npp32fc), cudaMemcpyHostToDevice));
}

void dsp::getSubtractionTrace(std::vector<std::complex<float>>& trace)
{
    this->handleError(cudaMemcpy((void*)trace.data(), (void*)subtraction_trace,
        total_length * sizeof(Npp32fc), cudaMemcpyDeviceToHost));
}

void dsp::resetSubtractionTrace()
{
    nppsSet_32fc(Npp32fc{ 0.f, 0.f }, subtraction_trace, total_length);
}
