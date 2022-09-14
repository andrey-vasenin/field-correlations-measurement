#include <iostream>
#include <cstdio>
#include <stdexcept>
#include "digitizer.h"
#include <functional>
#include <vector>
#include <numeric>
#include <complex>
#include "dsp.cuh"
#include <fstream>
#include <chrono>
#include <cmath>

extern const int CHANNELS_AMPLITUDE = 500;
const char DATA_FILENAME[] = "results.dt";

struct digitizer_parameters {
    int segment_size,
        segments,
        pretrigger,
        batch_size,
        notify_size,
        channels[2],
        amplitudes[2],
        delay,
        ext0_trigger_threshold,
        sampling_rate;
};

void setup_digitizer(Digitizer& dig, digitizer_parameters& params)
{
#ifdef _DEBUG
    std::cout << "Found " << dig << std::endl;
#endif // _DEBUG
    dig.handleError();
    dig.setupChannels(params.channels, params.amplitudes, 2);
    dig.setDelay(params.delay);
    dig.setSamplingRate(params.sampling_rate);
    dig.antialiasing(true);
    dig.setExt0TriggerOnPositiveEdge(params.ext0_trigger_threshold);
    dig.setupMultRecFifoMode(params.segment_size, params.pretrigger, params.segments);
    dig.createBuffer(params.notify_size);
#ifdef _DEBUG
    std::cout << "Buffer size: " << dig.getBufferSize() << std::endl;
#endif // _DEBUG
}

void init_digitizer_parameters(digitizer_parameters& dig_params)
{
    dig_params.segment_size = 6240;
    dig_params.segments = 1 << 24;
    dig_params.pretrigger = 32;
    dig_params.batch_size = 8;
    dig_params.notify_size = 2 * dig_params.segment_size * dig_params.batch_size;
    dig_params.channels[0] = 0;
    dig_params.channels[1] = 1;
    dig_params.delay = 32;
    dig_params.amplitudes[0] = CHANNELS_AMPLITUDE;
    dig_params.amplitudes[1] = CHANNELS_AMPLITUDE;
    dig_params.ext0_trigger_threshold = 500;
    dig_params.sampling_rate = 1'250'000'000 / 4;
}

template<typename T>
void save_data(std::vector<T>& data)
{
    std::ofstream fout(DATA_FILENAME);
    for (int i = 0; i < data.size() - 1; i++)
        fout << data[i] << ";";
    fout << data[data.size() - 1] << std::endl;
}

int main()
{
    using namespace std::chrono;
    using namespace std::complex_literals;

    auto start = high_resolution_clock::now();
    try
    {
        // Initialize a digitizer with parameters
        auto dig = Digitizer{ "/dev/spcm1" };
        digitizer_parameters dig_params;
        init_digitizer_parameters(dig_params);
        setup_digitizer(dig, dig_params);

        // Initialize the GPU memory by creating an object of dsp class
        float part = 0.4;
        auto processor = dsp{ dig_params.segment_size, 1, part };

#ifdef _DEBUG
        std::cout << "dsp object created\n";
#endif // _DEBUG

        // Set up paramaters for the compensation of IQ-mixer imbalanse in traces
        processor.setDownConversionCalibrationParameters(1.0768582980666606, 0.029679474536344622,
            1.4299266585593358, -1.8753134382883068);
        //cudaDeviceSynchronize();

#ifdef _DEBUG
        std::cout << "Initialization finished successfully\n";
#endif // _DEBUG

        // Choosing the measurement processor and wrapping it with a lambda expression
        auto funct = [&processor](const char* data) mutable { processor.computeNonstationaryG1(data); };

        // Start the measurement and processing procedure
        dig.launchFifo(dig_params.notify_size, dig_params.segments / dig_params.batch_size, dig_params.batch_size, funct);

        // Prepare to receive data from GPU
        int len = processor.getOutSize();  // divizion by 2 since the function return size in floats
        int side = processor.getTraceLength();
        std::vector <std::complex<float>> result(len);
        std::vector <std::complex<double>> average_result(side * side, 0);

        // Receive data from GPU
        processor.getCorrelator(result);
#if _DEBUG
        std::cout << "Received data from the GPU\n";
#endif // _DEBUG


        // Divide the data by a number of traces measured
        int k = 0;
        for (int t1= 0; t1 < side; t1++)
        {
            for (int t2 = t1; t2 < side; t2++)
            {
                int idx = (t2 - t1) * side + t1;
                average_result[idx] = std::conj(result[k]);
                average_result[idx] /= (double)dig_params.segments;
                k++;
            }
        }

        // Save the data to a file
        save_data(average_result);
#ifdef _DEBUG
        std::cout << average_result[0] << std::endl;
#endif // _DEBUG
    }
    catch (const std::exception exc)
    {
        // In case of an exception, print it out
        std::cerr << exc.what() << std::endl;
        return EXIT_FAILURE;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Execution " << (double)duration.count()/1000. << std::endl;
    return EXIT_SUCCESS;
}