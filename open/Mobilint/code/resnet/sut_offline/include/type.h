#ifndef MACCEL_TYPE_H
#define MACCEL_TYPE_H

#include <cstdint>
#include <vector>

using namespace std;

typedef struct ResultRepo {
    uint64_t offset = 0;
    uint64_t size = 0;
    int8_t* buf;
} ResultRepo;

/**
 * @struct RequestBlock
 * Data Structure of all requests (upload/download data from FPGA).
 * 
 * @var RequestBlock::offset
 * Offset of requested medium.
 * @var RequestBlock::payload
 * The return buffer. Supports two type- uint8_t buffer, uint32_t int.
 * @var RequestBlock::szRequest
 * The buffer size.
 */
typedef struct RequestBlock {
    uint64_t offset = 0;

    union { 
        uint8_t* buf;
        uint32_t* value;
    } payload = {nullptr};

    uint64_t szRequest = 0;
} RequestBlock;

/**
 * 
 */
typedef struct QueueBlock {
    int64_t id = -1;
    vector<RequestBlock> rb;
} QueueBlock;

typedef struct ReceiveBlock {
    int64_t id = -1;
    vector<RequestBlock> rb;
    bool received = false;
} ReceiveBlock;

#endif