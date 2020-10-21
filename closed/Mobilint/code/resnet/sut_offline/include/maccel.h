#ifndef MACCEL_H
#define MACCEL_H

#include <memory>
#include <vector>
#include "type.h"

using namespace std;

class MobilintAccelerator {
private:
    class Impl;
    unique_ptr<Impl> mImpl;

public:
    MobilintAccelerator(int numCore, bool verbose);
    
    /**
     * Instantiate Mobilint Accelerator with verbose option set.
     * 
     * @param verbose Print the verbose log
     */
    MobilintAccelerator(bool verbose);

    /**
     * Instantiate Mobilint Accelerator with verbose option off.
     */
    MobilintAccelerator();

    /**
     * Destruct the class.
     */
    ~MobilintAccelerator();

    /**
     * Perform the device initialization.
     * 
     * @return true if succesful, false if unsuccessful
     */
    bool initDevice();

    /**
     * Return the initialization status of the device.
     * 
     * @return true if it is initialized, false if not
     */
    bool isInitialized();

    /**
     * Destroy the device.
     */
    void destroyDevice();

    /**
     * Get the device status.
     * Currently, it only returns the initialization status.
     * 
     * @return the same result as isInitalized()
     */
    int getDeviceStatus();

    /**
     * Post the inference request on the buffer.
     * 
     * @param buf The data buffer
     * @param size The size of the buffer
     * @return Request id if successful, -1 if not.
     */
    int enqueueInferRequest(int8_t* buf, uint64_t size);
    int enqueueInferRequest(uint8_t* buf, uint64_t size);
    int64_t enqueueInferRequest(vector<RequestBlock>& arr);

    /**
     * Receive the inference result in the buffer.
     * 
     * @param buf The buffer
     * @param size The size of the buffer
     * @return 0 if successful, 1 if not
     */
    int receiveInferResult(int8_t* buf, uint64_t size);
    int receiveInferResult(uint8_t* buf, uint64_t size);
    int receiveInferResult(vector<RequestBlock>& arr);
    int receiveInferResult(int64_t id, vector<RequestBlock>& arr);

    /**
     * Setup the model into the Accelerator.
     * 
     * @param imem_path Path to imem.bin
     * @param lmem_path Path to lmem.bin
     * @param dmem_path Path to dmem.bin
     * @param weight_path Path to ddr.bin
     * @return 0 if succesful, 1 otherwise.
     */
    int setModel(const string& imem_path, const string& lmem_path, const string& dmem_path, const string& weight_path);
    int setModel(const string& bmem_path, const string& dmem_path);

    /**
     * Upload the additional data to the desired position.
     */
    int storeData(uint64_t offset, unsigned char* buffer, uint64_t size);
};

#endif