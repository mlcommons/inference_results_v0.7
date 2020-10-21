#include <iostream>
#include <map>
#include <numeric>
#include <experimental/filesystem>

#include <test_settings.h>
#include <loadgen.h>
#include <clipp.h>
#include <spdlog/spdlog.h>

#include "IvaTPUSystemUnderTest.h"
#include "IvaTPUQuerySampleLibrary.h"
#include "util.h"

#include <tpu.h>

using namespace ivatpu;
using namespace clipp;


const std::map<mlperf::TestScenario, std::string> mlperf_scenario = {{mlperf::TestScenario::SingleStream, "SingleStream"},
                                                                      {mlperf::TestScenario::Offline, "Offline"},
                                                                      {mlperf::TestScenario::Server, "Server"},
                                                                      {mlperf::TestScenario::MultiStream, "MultiStream"},
                                                                      {mlperf::TestScenario::MultiStreamFree, "MultiStreamFree"}};
std::string get_scenario_name(mlperf::TestScenario scenario)
{
    return mlperf_scenario.at(scenario);
}

const std::map<mlperf::TestMode, std::string> mlperf_mode = {{mlperf::TestMode::PerformanceOnly, "Performance"},
                                                {mlperf::TestMode::AccuracyOnly, "Accuracy"},
                                                {mlperf::TestMode::SubmissionRun, "Submission"}};

std::string get_mode_name(mlperf::TestMode mode)
{
    return mlperf_mode.at(mode);
}

int main(int argc, char *argv[]) {
    mlperf::TestSettings testSettings;
    mlperf::LogSettings logSettings;

    testSettings.offline_expected_qps = 80;

    auto singleStreamScenario = (
            command("SingleStream").set(testSettings.scenario, mlperf::TestScenario::SingleStream));
    auto offlineScenario = (
            command("Offline").set(testSettings.scenario, mlperf::TestScenario::Offline));
    auto serverScenario = (
            command("Server").set(testSettings.scenario, mlperf::TestScenario::Server));
    auto multiStreamScenario = (
            command("MultiStream").set(testSettings.scenario, mlperf::TestScenario::MultiStream));

    auto performanceMode = (command("performance").set(testSettings.mode, mlperf::TestMode::PerformanceOnly));
    auto accuracyMode = (command("accuracy").set(testSettings.mode, mlperf::TestMode::AccuracyOnly));
    auto submissionMode = (command("submission").set(testSettings.mode, mlperf::TestMode::SubmissionRun));

    std::string test_network = "resnet50";
    std::string dataset_path;
    std::string valmap_path;
    std::string config_path;
    std::string user_config_path;
    std::string model_path = "resnet50.tpu";

    auto cli = (
            (singleStreamScenario | offlineScenario | serverScenario | multiStreamScenario),
            (accuracyMode | performanceMode | submissionMode),
            (option("-d", "--dataset") & value("dataset", dataset_path)).doc("ImageNet dataset"),
            (option("-c", "--config") & value("config", config_path)).doc("mlperf config"),
            (option("-u", "--user-config") & value("user config", user_config_path)).doc("mlperf config user overrides"),
            (option("-v", "--val-map") & value("val_map", valmap_path)).doc("val_map.txt dataset file"),
            (option("-m", "--model") & value("model", model_path)).doc("model")
    );

    if(!parse(argc, argv, cli)) {
        std::cout << make_man_page(cli, argv[0]);
        exit(EXIT_FAILURE);
    }

    // Init logging

    if(dataset_path.empty()) {
        const char *env_p = std::getenv("HOME");
        dataset_path = std::string(env_p) + "/datasets/imagenet";
        // test directory exists
        if(!std::experimental::filesystem::exists(dataset_path)) {
            throw std::runtime_error(std::string("Dataset path not found ") + dataset_path);
        }
    }

    auto [device, program] = device_program_load(model_path);
    IvaTPUQuerySampleLibrary qsl(program, dataset_path);
    qsl.LoadValMap(valmap_path);
    IvaTPUSystemUnderTest sut(device, program, qsl);

    if(!config_path.empty()) {
        testSettings.FromConfig(config_path, test_network, get_scenario_name(testSettings.scenario));
    }

    if(!user_config_path.empty()) {
        testSettings.FromConfig(user_config_path, test_network, get_scenario_name(testSettings.scenario));
    }

    spdlog::set_level(spdlog::level::info);

    spdlog::info("Run {} scenario in {} mode", get_scenario_name(testSettings.scenario), get_mode_name(testSettings.mode));

    mlperf::StartTest(&sut, &qsl, testSettings, logSettings);

    return 0;
}
