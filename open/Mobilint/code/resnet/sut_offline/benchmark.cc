#include <iostream>
#include <chrono>
#include <omp.h>
#include <string.h>
#include <string>
#include <getopt.h>
#include <thread>
#include <cmath>
#include <fstream>
#include <queue>
#include <future>

#include "benchmark.h"
#include "ioutils.h"
#include "postprocessor.h"

using namespace std;
using namespace mlperf;
using namespace mlperf::c;
using std::thread;

static struct option const long_opts[] = {
    {"imem",     required_argument, NULL, 'a'},
    {"lmem",     required_argument, NULL, 'b'},
    {"dmem",     required_argument, NULL, 'c'},
    {"weight",   required_argument, NULL, 'd'},
    {"bmem",     required_argument, NULL, 'e'},
    {"dram",     required_argument, NULL, 'f'},
    {"config",   required_argument, NULL, 'g'},
    {"scenario", required_argument, NULL, 'h'},
    {"mode",     required_argument, NULL, 'i'},
    {"model",    required_argument, NULL, 'j'},
    {"dataset",  required_argument, NULL, 'k'},
    {0, 0, 0, 0}
};

vector<unique_ptr<MobilintAccelerator>> mAccelerator;
unique_ptr<PostprocessorManager> mPostprocessor = nullptr;

string model = "*";
string scenario = "SingleStream";
string mode = "AccuracyOnly";
int dataset_count = -1;
int qsl_size = -1;
string config_path = "mlperf.conf";
string dataset_path = "";

string bmem_path = "./bmem.bin";
string dram_path = "./ddr.bin";

string imem_path = "./imem.bin";
string lmem_path = "./lmem.bin";
string dmem_path = "./dmem.bin";
string weight_path = "./ddr.bin";

uint8_t** loadedSample = nullptr;
uint64_t* lookup = nullptr;

int check[50000] = {0, };

int dummy = 0;
atomic<int> infer_count;

#include <unistd.h>

/**
 * ResNet Worker Thread
 * 
 * Make Inference Request and Receive the result.
 */
void worker_resnet(const QuerySample* qs, int offset, size_t size, promise<vector<QuerySampleResponse>>& ret, int accNo) {
    vector<QuerySampleResponse> v;

    for (size_t i = offset; i < offset+size; i++) {
        infer_count.fetch_add(1);
        
        int8_t *result;
        int64_t requestId = -1;
        int szAlign = 4096;
        int ret;

        ret = posix_memalign((void **) &result, szAlign, 1024 + szAlign);

        vector<RequestBlock> vRB = {
            {0, loadedSample[lookup[qs[i].index]], 224*224*3}
        };

        vector<RequestBlock> vRBR = {
            { 752640, (uint8_t *) result, 1024 }
        };

        requestId = mAccelerator[accNo]->enqueueInferRequest(vRB);
        mAccelerator[accNo]->receiveInferResult(requestId, vRBR);
        

        int max = -9999999;
        int *res_bucket = (int *) malloc(sizeof(int));
        for (int i = 0; i < 1001; i++) {

            if (((int8_t) result[i]) > max) {
                max = (int8_t) result[i];
                *res_bucket = i - 1;
            }
        }

        free(result);

        v.push_back({
            qs[i].id, (uintptr_t) res_bucket, 4
        });
    }

    ret.set_value(v);
}

/**
 * SSD-MobileNet Worker Thread
 * 
 * Make Inference Request and Receive the result.
 */
void worker(const QuerySample* qs, int offset, size_t size, promise<vector<QuerySampleResponse>>& ret, int accNo) {
    vector<QuerySampleResponse> v;

    for (size_t i = offset; i < offset+size; i++) {
        //infer_count.fetch_add(1);
        //cout << "SUT: Inferencing " + to_string(infer_count) + "th item." << endl;
        
        int64_t requestId = -1;
        int8_t *cls0, *cls1, *cls2, *cls3, *cls4, *cls5;
        int8_t *box0, *box1, *box2, *box3, *box4, *box5;
        vector<float> boxes, classes, scores;

        int ret;
        int szAlign = 4096;
        
        cls0 = cls1 = cls2 = cls3 = cls4 = cls5 = nullptr;
        box0 = box1 = box2 = box3 = box4 = box5 = nullptr;
        ret = posix_memalign((void **) &cls0, szAlign, 103968 + szAlign);
        ret = posix_memalign((void **) &cls1, szAlign, 57600 + szAlign);
        ret = posix_memalign((void **) &cls2, szAlign, 14400 + szAlign);
        ret = posix_memalign((void **) &cls3, szAlign, 5184 + szAlign);
        ret = posix_memalign((void **) &cls4, szAlign, 2304 + szAlign);
        ret = posix_memalign((void **) &cls5, szAlign, 576 + szAlign);
        ret = posix_memalign((void **) &box0, szAlign, 11552 + szAlign);
        ret = posix_memalign((void **) &box1, szAlign, 3200 + szAlign);
        ret = posix_memalign((void **) &box2, szAlign, 800 + szAlign);
        ret = posix_memalign((void **) &box3, szAlign, 288 + szAlign);
        ret = posix_memalign((void **) &box4, szAlign, 128 + szAlign);
        ret = posix_memalign((void **) &box5, szAlign, 32 + szAlign);

        vector<RequestBlock> vRB = {
            {268435456-268435456, loadedSample[lookup[qs[i].index]], 300*320*3}
        };

        vector<RequestBlock> vRBR = {
            {791616, (uint8_t *) cls0, 103'968},
            {907136, (uint8_t *) cls1,  57'600},
            {967936, (uint8_t *) cls2,  14'400},
            {983136, (uint8_t *) cls3,   5'184},
            {988608, (uint8_t *) cls4,   2'304},
            {991040, (uint8_t *) cls5,     576},
            {895584, (uint8_t *) box0,  11'552},
            {964736, (uint8_t *) box1,   3'200},
            {982336, (uint8_t *) box2,     800},
            {988320, (uint8_t *) box3,     288},
            {990912, (uint8_t *) box4,     128},
            {991616, (uint8_t *) box5,      32}
        };
        
        requestId = mAccelerator[accNo]->enqueueInferRequest(vRB);
        mAccelerator[accNo]->receiveInferResult(requestId, vRBR);

        requestId = mPostprocessor->enqueue(
            cls0, cls1, cls2, cls3, cls4, cls5, 
            box0, box1, box2, box3, box4, box5, 
            boxes, classes, scores);
        mPostprocessor->receive(requestId);
        
        unsigned char* alloc_dyn = (unsigned char *) malloc(28 * scores.size());
        unsigned char* offset = alloc_dyn;
        memset(alloc_dyn, 0, 28 * scores.size());

        for (int j = 0; j < scores.size(); j++) {
            float *t = (float *) malloc (sizeof(float));
            *t = qs[i].index;
            memcpy(offset, (unsigned char*) t, 4);
            offset += 4;
            free(t);

            memcpy(offset, (unsigned char*) &(boxes[4*j + 1]), 4);
            offset += 4;
            memcpy(offset, (unsigned char*) &(boxes[4*j + 0]), 4);
            offset += 4;
            memcpy(offset, (unsigned char*) &(boxes[4*j + 3]), 4);
            offset += 4;
            memcpy(offset, (unsigned char*) &(boxes[4*j + 2]), 4);
            offset += 4;

            memcpy(offset, (unsigned char*) &(scores[j]), 4);
            offset += 4;
            memcpy(offset, (unsigned char*) &(classes[j]), 4);
            offset += 4;
        }

        v.push_back({
            qs[i].id, (uintptr_t) alloc_dyn, 28 * scores.size() 
        });

        free(cls0); free(cls1); free(cls2); free(cls3); free(cls4); free(cls5);
        free(box0); free(box1); free(box2); free(box3); free(box4); free(box5);
    }

    ret.set_value(v);
}

uint64_t getopt_integer(char *optarg)
{
	int rc;
	uint64_t value;

	rc = sscanf(optarg, "0x%lx", &value);
	if (rc <= 0)
		rc = sscanf(optarg, "%lu", &value);

	return value;
}

#include <algorithm>

namespace sut {
    const ClientData CLIENT_DATA = 0;
    const int MAX_WORKER_THREAD = 20;
    int mQueryCount = 1;

    void issueQuery(ClientData cd, const QuerySample* qs, size_t size) {
        cout << "SUT: Issue Query - Count " << mQueryCount++ << endl;
        promise<vector<QuerySampleResponse>> p[20];
        future<vector<QuerySampleResponse>> f[20] = {
            p[0].get_future(), p[1].get_future(), p[2].get_future(), p[3].get_future(),
            p[4].get_future(), p[5].get_future(), p[6].get_future(), p[7].get_future(),
            p[8].get_future(), p[9].get_future(), p[10].get_future(), p[11].get_future(),
            p[12].get_future(), p[13].get_future(), p[14].get_future(), p[15].get_future(),
            p[16].get_future(), p[17].get_future(), p[18].get_future(), p[19].get_future()
            };

        thread tWorker[MAX_WORKER_THREAD];
        int szThread = min((int) size, (int) MAX_WORKER_THREAD);
        size_t szSamplesPerThread = (size_t) ceil((float) size / MAX_WORKER_THREAD);
        int szRemain = size % (MAX_WORKER_THREAD);
        
        int offset = 0;
        bool equaled = false;
        if (szRemain == 0) {
            equaled = true;
        }
        
        for (int i = 0; i < szThread; i++) {
            tWorker[i] = thread(worker_resnet, qs, offset, szSamplesPerThread, ref(p[i]), i % mAccelerator.size());

            offset += szSamplesPerThread;

            szRemain--;
            if (!equaled && szRemain <= 0) {
                szSamplesPerThread--;
                equaled = true;
            }
        }
        
        for (int i = 0; i < szThread; i++) {
            tWorker[i].join();
            cout << "SUT: " << i << "th thread is successfully joined." << endl;
            
            auto vec = f[i].get();
        
            mlperf::c::QuerySamplesComplete(vec.data(), vec.size());
        }        
    }

    void flushQueries(void) {
        cout << "SUT: Flush Queries" << endl;
    }

    void processLatencies(ClientData cd, const int64_t* data, size_t size) {
        cout << "Process Latencies" << endl;
    }
}

namespace qsl {
    const ClientData CLIENT_DATA = 0;

    void loadImageNet(const QuerySampleIndex* qsi, size_t size) {
        cout << "Loading ImageNet Dataset from " << dataset_path << ", dataset count is " << size << endl;

        string buf[50000];
        ifstream readFile;
        readFile.open(dataset_path);
        int k = 0;

        while (!readFile.eof()) {
            getline(readFile, buf[k++]);
        }

        readFile.close();

        int resolution = 224*224;

        for (unsigned int i = 0; i < size; i++) {
            uint8_t buf_img[resolution*3] = {0, };
            int dummy = 0;
            lookup[qsi[i]] = i;

            ReadFile(buf[(int) qsi[i]], buf_img, &dummy);
            
            for (int j=0; j <resolution; j++){
                loadedSample[i][3*j + 0] = buf_img[3*j + 0];
                loadedSample[i][3*j + 1] = buf_img[3*j + 1];
                loadedSample[i][3*j + 2] = buf_img[3*j + 2];
            }
        }

        cout << "ImageNet Loading End!" << endl;
    }

    void loadCOCO(const QuerySampleIndex* qsi, size_t size, bool big) {
        cout << "Loading COCO Dataset from " << dataset_path << ", dataset count is " << size << endl;
        
        string buf[5000];
        ifstream readFile;
        readFile.open(dataset_path);
        int k = 0;

        while (!readFile.eof()) {
            getline(readFile, buf[k++]);
        }

        readFile.close();
        int resolution = big ? 1200*1216 : 300*320;

        for (unsigned int i = 0; i < size; i++) {
            uint8_t buf_img[resolution*3] = {0, };
            int dummy = 0;
            lookup[qsi[i]] = i;

            ReadFile(buf[(int) qsi[i]], buf_img, &dummy);
            
            for (int j=0; j <resolution; j++){
                loadedSample[i][3*j + 0] = buf_img[3*j + 0];
                loadedSample[i][3*j + 1] = buf_img[3*j + 1];
                loadedSample[i][3*j + 2] = buf_img[3*j + 2];
            }
        }

        cout << "COCO Loading End!" << endl;
    }

    void loadSamplesToRAM(ClientData cd, const QuerySampleIndex* qsi, size_t size) {
        cout << "Load " << size << " samples to RAM." << endl;

        loadedSample = (uint8_t **) malloc(sizeof(uint8_t *) * (int) size);

        if (model == "Resnet50-v1.5") {
            for (int i = 0; i < (int) size; i++) {
                posix_memalign((void **)&loadedSample[i], 4096, resnet_50_v1_5::IMAGE_SIZE + 4096);
            }

            loadImageNet(qsi, size);
        } else if (model == "SSD-ResNet34") {
            for (int i = 0; i < (int) size; i++) {
                posix_memalign((void **)&loadedSample[i], 4096, ssd_resnet_34::IMAGE_SIZE + 4096);
            }

            loadCOCO(qsi, size, true);
        } else if (model == "SSD-MobileNets-v1") {
            for (int i = 0; i < (int) size; i++) {
                posix_memalign((void **)&loadedSample[i], 4096, ssd_mobilenet_v1::IMAGE_SIZE + 4096);
            }

            loadCOCO(qsi, size, false);
        }
    }

    void unloadSamplesFromRAM(ClientData cd, const QuerySampleIndex* qsi, size_t size) {
        cout << "SUT: Unload " << size << " samples from RAM." << endl;

        for (int i = 0; i < (int) size; i++) {
            free(loadedSample[i]);
        }

        free(loadedSample);
    }
}

int main(int argc, char **argv) {
    infer_count = 0;

    /* Break down the arguments */
    int cmd_opt;
    while ((cmd_opt = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:j:k", long_opts,
			    NULL)) != -1) {
		switch (cmd_opt) {
		case 0:
			break;
		case 'a':
			imem_path = strdup(optarg);
			break;
		case 'b':
			lmem_path = strdup(optarg);
			break;
		case 'c':
			dmem_path = strdup(optarg);
			break;
		case 'd':
			weight_path = strdup(optarg);
			break;
		case 'e':
			bmem_path = strdup(optarg);
			break;
		case 'f':
			dram_path = strdup(optarg);
			break;
		case 'g':
			config_path = strdup(optarg);
			break;
        case 'h':
            scenario = strdup(optarg);
            break;
        case 'i':
            mode = strdup(optarg);
            break;
        case 'j':
            model = strdup(optarg);
            break;
        case 'k':
            dataset_path = strdup(optarg);
            break;
		default:
            cerr << "No such argument" << endl;
			break;
		}
	}

    /* Load the benchmark settings */
    TestSettings settings;
    settings.FromConfig(config_path, model, scenario);

    if (scenario == "SingleStream") {
        settings.scenario = TestScenario::SingleStream;
    } else if (scenario == "MultiStream") {
        settings.scenario = TestScenario::MultiStream;
    } else if (scenario == "Offline") {
        settings.scenario = TestScenario::Offline;
    } else if (scenario == "Server") {
        settings.scenario = TestScenario::Server;
    } else {
        cerr << "Wrong scenario " << scenario << endl;
        exit(1);
    }

    if (model == "Resnet50-v1.5") {
        dataset_count = resnet_50_v1_5::TOTAL_SAMPLE_COUNT;
        qsl_size = resnet_50_v1_5::QSL_SIZE;
    } else if (model == "SSD-ResNet34") {
        dataset_count = ssd_resnet_34::TOTAL_SAMPLE_COUNT;
        qsl_size = ssd_resnet_34::QSL_SIZE;
    } else if (model == "SSD-MobileNets-v1") {
        dataset_count = ssd_mobilenet_v1::TOTAL_SAMPLE_COUNT;
        qsl_size = ssd_mobilenet_v1::QSL_SIZE;
    } else {
        cerr << "Wrong model " << model << endl;
        exit(1);
    }

    if (mode == "AccuracyOnly") {
        settings.mode = TestMode::AccuracyOnly;
    } else if (mode == "PerformanceOnly") {
        settings.mode = TestMode::PerformanceOnly;
    } else {
        cerr << "Wrong mode " << mode << endl;
        exit(1);
    }

    lookup = (uint64_t *) malloc(sizeof(uint64_t) * dataset_count);

    /* Prepare SUT, QSL */
    SystemUnderTest *sut = (SystemUnderTest*) ConstructSUT(
        sut::CLIENT_DATA, SUT_NAME, sizeof(SUT_NAME) - 1, sut::issueQuery, sut::flushQueries, sut::processLatencies);
    QuerySampleLibrary *qsl = (QuerySampleLibrary*) ConstructQSL(
        qsl::CLIENT_DATA, SUT_NAME, sizeof(SUT_NAME) - 1, dataset_count, qsl_size, 
        qsl::loadSamplesToRAM, qsl::unloadSamplesFromRAM);

    /* Instantiate the Accelerators (Multiple Accelerator if needed) */
    mAccelerator.push_back(make_unique<MobilintAccelerator>(0, true));
    mAccelerator.push_back(make_unique<MobilintAccelerator>(1, true));
    mAccelerator.push_back(make_unique<MobilintAccelerator>(2, true));
    mAccelerator.push_back(make_unique<MobilintAccelerator>(3, true));
    mAccelerator.push_back(make_unique<MobilintAccelerator>(4, true));
    
    /* Instantiate the Postprocessing Module */
    mPostprocessor = make_unique<PostprocessorManager>();
    
    /* Perform Initialization */
    for (int i = 0; i < mAccelerator.size(); i++) {
        mAccelerator[i]->initDevice();
        mAccelerator[i]->setModel(imem_path, lmem_path, dmem_path, weight_path);
    }

    StartTest(sut, qsl, settings);

    DestroyQSL(qsl);
    DestroySUT(sut);

    free(lookup);

    return 0;
}
