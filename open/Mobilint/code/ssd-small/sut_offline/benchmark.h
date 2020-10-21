#include "include/bindings/c_api.h"
#include "include/loadgen.h"
#include "include/system_under_test.h"
#include "include/query_sample_library.h"
#include "include/test_settings.h"

#include "include/maccel.h"

#define SUT_NAME "MOBILINT"

namespace resnet_50_v1_5 {
    const int TOTAL_SAMPLE_COUNT = 50000;
    const int QSL_SIZE = 1024;
    const int IMAGE_SIZE = 224 * 224 * 3;
}

namespace ssd_resnet_34 {
    const int TOTAL_SAMPLE_COUNT = 5000;
    const int QSL_SIZE = 64;
    const int IMAGE_SIZE = 1200 * 1216 * 3;
}

namespace ssd_mobilenet_v1 {
    const int TOTAL_SAMPLE_COUNT = 5000;
    const int QSL_SIZE = 256;
    const int IMAGE_SIZE = 300 * 320 * 3;
}

