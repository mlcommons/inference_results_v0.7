//
// Created by mmoroz on 8/18/20.
//

#include <experimental/filesystem>
#include "catch2/catch.hpp"
#include <util.h>
#include <IvaTPUQuerySampleLibrary.h>
#include <IvaTPUSystemUnderTest.h>

#include <iostream>

using namespace ivatpu;
static int detected_class = 0;
static std::mutex m;
static std::condition_variable cv;

namespace mlperf {
    void QuerySamplesComplete(QuerySampleResponse* responses, unsigned long size){
        std::cout << "got response " << size << " elements" << std::endl;
        std::unique_lock<std::mutex> lk(m);
        auto response = responses[0];
        int *pclass = reinterpret_cast<int *>(response.data);
        detected_class = *pclass;
        for(size_t i = 0; i < size; ++i) {
            response = responses[i];
            int *pclass = reinterpret_cast<int *>(response.data);
            std::cout << "Detected class: " << *pclass << std::endl;
        }

        lk.unlock();
        cv.notify_one();

    }
}

TEST_CASE("test_inference") {
    std::string model_path = "/auto/tests/mlperf/resnet50.tpu";
    std::string valmap_path = "/auto/tests/mlperf/val_map.txt";

    const char *env_p = std::getenv("HOME");
    auto dataset_path = std::string(env_p) + "/datasets/imagenet";

    // test directory exists
    if (!std::experimental::filesystem::exists(dataset_path)) {
        throw std::runtime_error(std::string("Dataset path not found ") + dataset_path);
    }

    auto [device, program] = device_program_load(model_path);

    IvaTPUQuerySampleLibrary qsl(program, dataset_path);
    qsl.LoadValMap(valmap_path);
    IvaTPUSystemUnderTest sut(device, program, qsl);

    qsl.LoadSamplesToRam({43790});
    std::vector<mlperf::QuerySample> queries = {{0, 43790}, {1, 43790}};
    sut.IssueQuery(queries);
    // wait for the inference
    {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, []{return detected_class;});
    }
    sut.FlushQueries();
    REQUIRE(detected_class == 221);
}