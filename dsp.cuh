//
// Created by andrei on 3/27/21.
//

#ifndef CPPMEASUREMENT_DSP_CUH
#define CPPMEASUREMENT_DSP_CUH

#include <nppdefs.h>
#include <vector>
#include <complex>
#include <cufft.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

const int num_streams = 4;
const int cal_mat_size = 16;
const int cal_mat_side = 4;

typedef thrust::device_vector<float> gpuvec;
typedef thrust::host_vector<float> hostvec;
typedef thrust::device_vector<std::complex<float> > gpuvec_c;
typedef thrust::host_vector<std::complex<float>> hostvec_c;
typedef thrust::device_vector<Npp8s> gpubuf;
typedef thrust::host_vector<Npp8s,thrust::cuda::experimental::pinned_allocator<Npp8s>> hostbuf;

template <typename T>
inline T *get(thrust::device_vector<T> vec)
{
    return thrust::raw_pointer_cast(&vec[0]);
}

class dsp
{
    /* Pointer */
    hostbuf buffer;

    /* Pointers to arrays with data */
    gpubuf gpu_buf[num_streams];  // buffers for loading data
    gpubuf gpu_buf2[num_streams]; // buffers for loading data
    gpuvec data[num_streams];
    gpuvec data_calibrated[num_streams];
    gpuvec noise[num_streams];
    gpuvec noise_calibrated[num_streams];
    gpuvec_c data_complex[num_streams];

    gpuvec power[num_streams]; // arrays for storage of average power
    gpuvec field[num_streams]; // arrays for storage of average field
    gpuvec_c out[num_streams];

    int cnt = 0;

    /* Min-max values */
    Npp8u *minmaxbuffer;
    Npp32f *minfield;
    Npp32f *maxfield;
    Npp32f min, max;

    /* Filtering window */
    gpuvec_c firwin;

    /* Subtraction trace */
    gpuvec_c subtraction_trace;

    /* Downconversion coefficients */
    gpuvec_c downconversion_coeffs;

private:
    /* Useful variables */
    int trace_length; // for keeping the length of a trace
    int trace1_start, trace2_start, pitch;
    int batch_size;   // for keeping the number of segments in data array
    int total_length; // batch_size * trace_length
    int out_size;
    int semaphore = 0;                           // for selecting the current stream
    Npp32fc scale = Npp32fc{500.f / 128.f, 0.f}; // for conversion into mV

    /* Streams' arrays */
    cudaStream_t streams[num_streams];
    NppStreamContext streamContexts[num_streams];

    /* cuFFT required variables */
    cufftHandle plans[num_streams];

    /* cuBLAS required variables */
    cublasHandle_t cublas_handles[num_streams];
    cublasHandle_t cublas_handles2[num_streams];

    /* Down-conversion calibration variables */
    gpuvec A_gpu;
    gpuvec_c C_gpu;
    int batch_count;
    float alpha = 1.f;
    float beta = 0.f;

public:
    dsp(int len, int n, float part);

    ~dsp();

    int getTraceLength();

    int getTotalLength();

    int getOutSize();

    void setFirwin(float cutoff_l, float cutoff_r, int oversampling = 1);

    void resetOutput();

    int getCounter() { return cnt; }

    float getMin()
    {
        cudaMemcpy(&min, minfield, sizeof(Npp32f) * 1, cudaMemcpyDeviceToHost);
        return min;
    }

    float getMax()
    {
        cudaMemcpy(&max, maxfield, sizeof(Npp32f) * 1, cudaMemcpyDeviceToHost);
        return max;
    }

    void compute(const char *buffer);

    void getCumulativePower(std::vector<std::complex<float>> &result);

    void getCumulativeField(std::vector<std::complex<float>> &result);

    void getCorrelator(std::vector<std::complex<float>> &result);

    void setDownConversionCalibrationParameters(float r, float phi, float offset_i, float offset_q);

    void setSubtractionTrace(std::vector<std::complex<float>> &trace);

    void getSubtractionTrace(std::vector<std::complex<float>> &trace);

    void resetSubtractionTrace();

    void createBuffer(uint32_t size);

    void setIntermediateFrequency(float frequency, int oversampling);

    char *getBufferPointer();

    void setAmplitude(int ampl);

protected:
    void handleError(cudaError_t error);

    void switchStream() { semaphore = (semaphore < (num_streams - 1)) ? semaphore + 1 : 0; };

    void loadDataToGPUwithPitchAndOffset(const char *buffer, Npp8s *gpu_buf, size_t pitch, size_t offset, int stream_num);

    void convertDataToMilivolts(gpuvec data, gpubuf gpu_buf, int stream_num);

    void downconvert(Npp32fc *data, int stream_num);

    void applyDownConversionCalibration(Npp32f *data, Npp32f *data_calibrated, int stream_num);

    void addDataToOutput(Npp32f *data, Npp32f *output, int stream_num);

    void subtractDataFromOutput(Npp32f *data, Npp32f *output, int stream_num);

    void applyFilter(Npp32fc *data, const Npp32fc *window, int stream_num);

    void getMinMax(Npp32f *data, int stream_num);

    void calculateField(int stream_num);

    void calculatePower(int stream_num);

    void calculateG1(int stream_num);
};

#endif // CPPMEASUREMENT_DSP_CUH
