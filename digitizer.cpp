//
// Created by andrei on 3/26/21.
//

#include <stdexcept>
#include <iostream>
#include <functional>
#include <vector>
#include "digitizer.h"
#include "dlltyp.h"
#include "regs.h"
#include "spcerr.h"
#include "spcm_drv.h"
#include <numeric>

// Open the digitizer by passing the address, i.e. "/dev/spcm0/"
Digitizer::Digitizer(const char* addr)
{
    handle = spcm_hOpen(addr);
    if (!handle)
        throw std::runtime_error("Digitizer can not be imported");
    this->loadProperties();
}

// Open the digitizer by passing its handle
Digitizer::Digitizer(void* h)
{
    created_here = false;
    handle = h;
    this->loadProperties();
}

// Digitizer destructor
Digitizer::~Digitizer() {
    if (created_here)
        spcm_vClose(handle);
}

// private method to use with initializer
void Digitizer::loadProperties() {
    spcm_dwGetParam_i32(handle, SPC_PXIHWSLOTNO, &slot);
    this->handleError();
}

// returns the digitizer's slot number in the PXIe chassis
int32 Digitizer::getSlotNumber() {
    return slot;
}

// returns the number of samples per segment
int Digitizer::getSegmentSize() {
    long segmentSize;
    spcm_dwGetParam_i32(handle, SPC_SEGMENTSIZE, &segmentSize);
    return static_cast<int>(segmentSize);
}

// returns the total number of segments to measure
int Digitizer::getSegmentsNumber() {
    long segmentsCount;
    spcm_dwGetParam_i32(handle, SPC_LOOPS, &segmentsCount);
    return static_cast<int>(segmentsCount);
}

void* Digitizer::getHandle() {
    return handle;
}

void Digitizer::setDelay(int delay)
{
    spcm_dwSetParam_i32(handle, SPC_TRIG_DELAY, delay);
}

void Digitizer::setSamplingRate(int samplerate)
{
    spcm_dwSetParam_i32(handle, SPC_SAMPLERATE, samplerate);
}

int Digitizer::getSamplingRate()
{
    long samplerate = 0;
    spcm_dwGetParam_i32(handle, SPC_SAMPLERATE, &samplerate);
    return static_cast<int>(samplerate);
}

void Digitizer::setTimeout(int milliseconds)
{
    spcm_dwSetParam_i32(handle, SPC_TIMEOUT, milliseconds);
}

// sets ADC channels and their amplitudes
void Digitizer::setupChannels(const int* channels, const int* amplitudes, int size) {
    int32 mask = 0;
    for (int i = 0; i < size; i++)
    {
        mask += 1 << channels[i];
        int32 amp_register = SPC_AMP0 + 100 * channels[i];
        spcm_dwSetParam_i32(handle, amp_register, amplitudes[i]);
    }
    spcm_dwSetParam_i32(handle, SPC_CHENABLE, mask);
    this->handleError();
}

// Switches the input filter with 350 MHz that prevents aliasing
void Digitizer::antialiasing(bool flag) {
    spcm_dwSetParam_i32(handle, SPC_FILTER0, (int32)flag);
    this->handleError();
}

// Sets the Multiple Recording Mode
void Digitizer::setupMultRecFifoMode(int32 segmentsize, int32 pretrigger, int segments) {
    spcm_dwSetParam_i32(handle, SPC_CARDMODE, SPC_REC_FIFO_MULTI);
    spcm_dwSetParam_i32(handle, SPC_SEGMENTSIZE, segmentsize);
    spcm_dwSetParam_i32(handle, SPC_POSTTRIGGER, segmentsize - pretrigger);
    spcm_dwSetParam_i32(handle, SPC_LOOPS, segments);
    this->handleError();
}

// Sets the digitizer to trigger on a positive edge at EXT0 port
void Digitizer::setExt0TriggerOnPositiveEdge(int32 voltageThreshold) {
    spcm_dwSetParam_i32(handle, SPC_TRIG_ORMASK, SPC_TMASK_EXT0);
    spcm_dwSetParam_i32(handle, SPC_TRIG_EXT0_MODE, SPC_TM_POS);
    spcm_dwSetParam_i32(handle, SPC_TRIG_EXT0_LEVEL0, voltageThreshold);
    this->handleError();
}

void Digitizer::setBuffer(char* buffer_ptr, int size)
{
    buffer = (int8*)buffer_ptr;
    buffersize = static_cast<int32>(size);
}

// returns a pointer to a host buffer
int8* Digitizer::getBuffer() {
    return buffer;
}

// returns the size of a host buffer
int32 Digitizer::getBufferSize() {
    return buffersize;
}

int64 Digitizer::getMemsize() {
    int64 memsize;
    spcm_dwGetParam_i64(handle, SPC_PCIMEMSIZE, &memsize);
    return memsize;
}

// starts the fifo measurement
void Digitizer::launchFifo(int32 notifysize, int n, proc_t processor) {
    /*
    * int32 notifysize: size in bytes of one batch
    * int n: number of iterations, what basically means the total number of segments to measure
    *        divided by the number of segments per batch
    * proc_t processor: a function to be called after processing each batch
    */
    // Define the host buffer for the digitizer
    spcm_dwDefTransfer_i64(handle, SPCM_BUF_DATA, SPCM_DIR_CARDTOPC, (uint32)notifysize,
        buffer, 0, (uint64)buffersize);
    this->handleError();

    spcm_dwSetParam_i32(handle, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_DATA_STARTDMA);
    this->handleError();
    int32 shift = 0;
    int32 availBytes = 0;
    int i = 0;
    while (i < n) {
        auto err = spcm_dwSetParam_i32(handle, SPC_M2CMD, M2CMD_DATA_WAITDMA);
        /* Since the transfer speed is slower than the acquisition speed, an error ERR_FIFOHWOVERRUN (hardware buffer overrun)
        may occur. The following exception handling restarts the measurement. */
        try
        {
            this->handleError();
        }
        catch (const std::exception exc)
        {
            if (err == ERR_FIFOHWOVERRUN)
            {
#ifdef _DEBUG
                std::cerr << exc.what() << std::endl;
#endif // _DEBUG
                spcm_dwSetParam_i32(handle, SPC_M2CMD, M2CMD_CARD_STOP);
                spcm_dwDefTransfer_i64(handle, SPCM_BUF_DATA, SPCM_DIR_CARDTOPC, (uint32)notifysize,
                    buffer, 0, (uint64)buffersize);
                this->handleError();
                spcm_dwSetParam_i32(handle, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_DATA_STARTDMA);
                continue;
            }
            else
            {
                throw exc;
            }
        }
        spcm_dwGetParam_i32(handle, SPC_DATA_AVAIL_USER_POS, &shift);
        this->handleError();
        spcm_dwGetParam_i32(handle, SPC_DATA_AVAIL_USER_LEN, &availBytes);
        this->handleError();
        if (availBytes < notifysize)
        {
#ifdef _DEBUG
            std::cerr << "not enough bytes available\n";
#endif // _DEBUG
            continue;
        }
        processor(buffer + shift);
        spcm_dwSetParam_i32(handle, SPC_DATA_AVAIL_CARD_LEN, notifysize);
        this->handleError();
        i++;
    }
    spcm_dwInvalidateBuf(handle, SPCM_BUF_DATA);
    this->handleError();
    this->stopCard();
}

void Digitizer::handleError() {
    auto err = spcm_dwGetErrorInfo_i32(handle, NULL, NULL, errortext);
    if (err != ERR_OK)
        throw std::runtime_error(errortext);
}

void Digitizer::stopCard() {
    spcm_dwSetParam_i32(handle, SPC_M2CMD, M2CMD_CARD_STOP);
    this->handleError();
}