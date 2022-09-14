//
// Created by andrei on 4/13/21.
//
#include <memory>
#include <iostream>
#include <functional>
#include <vector>
#include <numeric>
#include <complex>
#include <cstdint>
#include "digitizer.h"
#include "dsp.cuh"
#include "measurement.cuh"
#include <pybind11/pybind11.h>

namespace py = pybind11;

Measurement::Measurement(std::uintptr_t dig_handle, unsigned long long averages, int batch, float part)
{
    dig = new Digitizer(reinterpret_cast<void*>(dig_handle));
    segment_size = dig->getSegmentSize();
    batch_size = batch;
    this->setAveragesNumber(averages);
    notify_size = 2 * segment_size * batch_size;
    dig->handleError();
    dig->setTimeout(5000);  // ms
    processor = new dsp(segment_size, batch_size, part);
    this->initializeBuffer();

    func = [this](const char* data) mutable { this->processor->compute(data); };

    int trace_length = processor->getTraceLength();

    test_input = new char[notify_size * 2];
}

void Measurement::initializeBuffer()
{
    // Create the buffer in page-locked memory
    int buffersize = 4 * notify_size;
    processor->createBuffer(buffersize * sizeof(char));
    auto buffer = processor->getBufferPointer();
    dig->setBuffer(buffer, buffersize);
}

void Measurement::setAmplitude(int ampl)
{
    processor->setAmplitude(ampl);
}

/* Use frequency in GHz */
void Measurement::setIntermediateFrequency(float frequency)
{
    int oversampling = (int)std::round(1.25E+9f / dig->getSamplingRate());
    processor->setIntermediateFrequency(frequency, oversampling);
    cudaDeviceSynchronize();
}

void Measurement::setAveragesNumber(unsigned long long averages)
{
    segments_count = averages;
    iters_num = static_cast<int>(averages / static_cast<unsigned long long>(batch_size));
    iters_done = 0;
}

void Measurement::setCalibration(float r, float phi, float offset_i, float offset_q)
{
    processor->setDownConversionCalibrationParameters(r, phi, offset_i, offset_q);
}

void Measurement::setFirwin(float left_cutoff, float right_cutoff)
{
    int oversampling = (int) std::round(1.25E+9f / dig->getSamplingRate());
    processor->setFirwin(left_cutoff, right_cutoff, oversampling);
    cudaDeviceSynchronize();
}

int Measurement::getCounter()
{
    return processor->getCounter();
}

void Measurement::measure()
{
    dig->launchFifo(notify_size, iters_num, func);
    iters_done += iters_num;
}

void Measurement::measureTest()
{
    for (int i = 0; i < iters_num; i++)
        func(test_input);
    iters_done += iters_num;
}

void Measurement::setTestInput(py::buffer input)
{
    py::buffer_info info = input.request();
    if (info.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");
    if (info.size < 2 * segment_size)
        throw std::runtime_error("Number of element in the imput array "
            "must be larger or equal to the two segment sizes");

    char* input_ptr = (char*)info.ptr;

    for (int j = 0; j < batch_size; j++)
    {
        for (int i = 0; i < segment_size; i++)
        {
            test_input[2 * (j * segment_size + i)] = (char)input_ptr[2 * i];
            test_input[2 * (j * segment_size + i) + 1] = (char)input_ptr[2 * i + 1];
        }
    }
}

std::vector<std::complex<double>> Measurement::getMeanField()
{
    int len = processor->getTotalLength();
    int tl = processor->getTraceLength();

    std::vector <std::complex<float>> field_from_gpu(len);
    processor->getCumulativeField(field_from_gpu);

    // Compute mean 
    std::vector<std::complex<double>> mean_field(tl, 0.);
    double denominator{ 1 };
    if (iters_done > 0)
        denominator = static_cast<double>(iters_done) * static_cast<double>(batch_size);
    for (int j = 0; j < tl; j++)
    {
        for (int i = 0; i < batch_size; i++)
        {
            int idx = i * tl + j;
            std::complex<double> fval = field_from_gpu[idx];
            mean_field[j] += fval;
        }
        mean_field[j] /= denominator;
    }
    return mean_field;
}

std::vector<double> Measurement::getMeanPower()
{
    int len = processor->getTotalLength();
    int tl = processor->getTraceLength();

    std::vector <std::complex<float>> power_from_gpu(len);
    processor->getCumulativePower(power_from_gpu);

    // Compute mean 
    std::vector<double> mean_power(tl, 0.);
    double denominator{ 1 };
    if (iters_done > 0)
        denominator = iters_done * batch_size;
    for (int j = 0; j < tl; j++)
    {
        for (int i = 0; i < batch_size; i++)
        {
            int idx = i * tl + j;
            std::complex<double> pval = power_from_gpu[idx];
            mean_power[j] += pval.real() + pval.imag();
        }
        mean_power[j] /= denominator;
    }
    return mean_power;
}


std::vector <std::vector<std::complex<double>>> Measurement::getCorrelator()
{
    int len = processor->getOutSize();
    int side = processor->getTraceLength();

    std::vector <std::complex<float>> result(len);
    std::vector <std::vector<std::complex<double>>> average_result(
        side, std::vector<std::complex<double>>(side));

    // Receive data from GPU
    processor->getCorrelator(result);

    // Divide the data by a number of traces measured
    int k = 0;
    float denominator{ 1 };
    if (iters_done > 0)
        denominator = iters_done * batch_size;
    std::complex<float> X(denominator, 0);
    for (int t1 = 0; t1 < side; t1++)
    {
        for (int t2 = t1; t2 < side; t2++)
        {
            k = t1 * side + t2;
            average_result[t1][t2] = result[k] / X;
            average_result[t2][t1] = std::conj(result[k]) / X;
        }
    }
    return average_result;
}

std::vector<std::complex<float>> Measurement::getRawCorrelator()
{
    int len = processor->getOutSize();
    int side = processor->getTraceLength();

    std::vector <std::complex<float>> result(len);

    // Receive data from GPU
    processor->getCorrelator(result);

    return result;
}

void Measurement::setSubtractionTrace(std::vector<std::complex<float>> trace)
{
    //using namespace std::complex_literals;
    int N = processor->getTraceLength();
    int M = processor->getTotalLength();

    //py::buffer_info info = trace.request();
    //if (info.ndim != 1)
    //    throw std::runtime_error("Number of dimensions must be one");
    //if (info.size < 2 * N)
    //    throw std::runtime_error("Number of element in the input array "
    //        "must be larger or equal to the returned trace lenght");

    //float* trace_ptr = (float*)info.ptr;
    
    std::vector<std::complex<float>> average(M);

    for (int i = 0; i < batch_size; i++)
    {
        for (int j = 0; j < N; j++)
        {
            //average[i * N + j] = std::complex<float>(trace_ptr[2 * j], trace_ptr[2 * j + 1]);
            average[i * N + j] = trace[j];
        }
    }

    processor->setSubtractionTrace(average);
}

std::vector<std::complex<float>> Measurement::getSubtractionTrace()
{
    int len = processor->getTotalLength();
    std::vector<std::complex<float>> subtraction_trace(len);
    processor->getSubtractionTrace(subtraction_trace);
    return subtraction_trace;
}

void Measurement::reset()
{
    this->resetOutput();
    processor->resetSubtractionTrace();
}

void Measurement::resetOutput()
{
    iters_done = 0;
    processor->resetOutput();
}

void Measurement::free()
{
    delete processor;
    delete dig;
    processor = NULL;
    dig = NULL;
    delete[] test_input;
}

Measurement::~Measurement()
{
    if ((processor != NULL) || (dig != NULL))
        this->free();
}