//
// Created by andrei on 4/13/21.
//
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/iostream.h>
#include "measurement.cuh"

using namespace pybind11::literals;

namespace py = pybind11;

PYBIND11_MODULE(CorrelatorsMeasurement, m) {
    py::class_<Measurement>(m, "CorrelatorsMeasurer")
        .def(py::init<std::uintptr_t, unsigned long long, int, float>())
        .def("set_calibration", &Measurement::setCalibration)
        .def("set_firwin", &Measurement::setFirwin)
        .def("measure", &Measurement::measure,
            py::call_guard<py::scoped_ostream_redirect,
            py::scoped_estream_redirect,
            py::gil_scoped_release>())
        .def("get_mean_field", &Measurement::getMeanField)
        .def("get_mean_power", &Measurement::getMeanPower)
        .def("get_correlator", &Measurement::getCorrelator)
        .def("get_raw_correlator", &Measurement::getRawCorrelator)
        .def("reset", &Measurement::reset)
        .def("reset_output", &Measurement::resetOutput)
        .def("get_counter", &Measurement::getCounter)
        .def("get_max_field", &Measurement::getMaxField)
        .def("get_min_field", &Measurement::getMinField)
        .def("free", &Measurement::free)
        .def("measure_test", &Measurement::measureTest,
            py::call_guard<py::scoped_ostream_redirect,
            py::scoped_estream_redirect>())
        .def("set_test_input", &Measurement::setTestInput)
        .def("set_subtraction_trace", &Measurement::setSubtractionTrace)
        .def("get_subtraction_trace", &Measurement::getSubtractionTrace)
        .def("set_amplitude", &Measurement::setAmplitude)
        .def("set_intermediate_frequency", &Measurement::setIntermediateFrequency)
        .def("set_averages_number", &Measurement::setAveragesNumber)
        .def("get_total_length", &Measurement::getTotalLength)
        .def("get_trace_length", &Measurement::getTraceLength)
        .def("get_out_size", &Measurement::getOutSize)
        .def("get_notify_size", &Measurement::getNotifySize);
}
