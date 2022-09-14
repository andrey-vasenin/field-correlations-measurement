//
// Created by andrei on 3/26/21.
//

#ifndef CPPMEASUREMENT_DIGITIZER_H
#define CPPMEASUREMENT_DIGITIZER_H

#include <vector>
#include <functional>
#include <ostream>
#include "dlltyp.h"
#include "regs.h"

typedef std::function<void(const char*)> proc_t;

class Digitizer {
    drv_handle handle;
    int32 slot;
    int8* buffer;
    int32 buffersize;
    char errortext[ERRORTEXTLEN];
    bool created_here = true;

    void loadProperties();

public:
    /* Constructors */
    Digitizer(void* h);

    Digitizer(const char* addr);

    ~Digitizer();

    /* Getters */
    int32 getSlotNumber();

    int32 getBufferSize();

    int64 getMemsize();

    int8* getBuffer();

    int getSegmentSize();

    int getSegmentsNumber();

    void* getHandle();

    /* Setters */
    void createBuffer(int32 totalsize);

    void setBuffer(char* buffer_ptr, int size);

    /* Setup */
    void setupChannels(const int* channels, const int* amplitudes, int size);

    void antialiasing(bool flag);

    void setDelay(int delay);

    void setSamplingRate(int samplerate);

    int getSamplingRate();

    void setTimeout(int milliseconds);

    void handleError();

    void setExt0TriggerOnPositiveEdge(int32 voltageThreshold);

    /* Mode setters */
    void setupMultRecFifoMode(int32 segmentsize, int32 pretrigger, int segments);

    /* Measurers */
    void launchFifo(int32 notifysize, int n, proc_t processor);

    /* Control */
    void stopCard();

    /* Operators */
    friend std::ostream& operator<<(std::ostream& out, const Digitizer& dig) {
        return out << "digitizer in PXIe slot #" << dig.slot;
    };
};

#endif //CPPMEASUREMENT_DIGITIZER_H
