/*
 * Copyright (C) 2020, Xilinx Inc - All rights reserved
 * Xilinx Runtime (XRT) Experimental APIs
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <dirent.h>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include "json.hpp"
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <google/protobuf/text_format.h>
#include <glog/logging.h>
/* mlperf headers */
#include <loadgen.h>
#include <query_sample_library.h>
#include <system_under_test.h>
#include <test_settings.h>

/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

/* header file for Runner APIs */
#include <dpu/dpu_runner.hpp>
#include <vitis/ai/nnpp/tfssd.hpp>
#include <vitis/ai/nnpp/ssd.hpp>
#include <vitis/ai/profiling.hpp>

#define _HW_SORT_EN_ 1

#if _HW_SORT_EN_
/* HW post proc-sort Init */
#include "tfssd/sort_xrt/sort_wrapper.h"
PPHandle* pphandle;
#endif

using namespace mlperf;
using namespace std;
using namespace cv;
using namespace std::chrono;
#ifdef ENABLE_INT
struct InputData {
  std::string image;
  std::shared_ptr<int8_t[]> data;
};
#else
struct InputData {
  std::string image;
  std::shared_ptr<float[]> data;
};
#endif
int threadnum;
int batchSize;
int modeScenario;
int num_channels_ = 3;
int is_server = 0;
int image_height_ = 300;
int image_width_ = 300;
std::thread   postThread;
string model;// = "ssd_mobilenet";
std::string dpuDir, imgDir;

std::string slurp(const char* filename);
vitis::ai::proto::DpuModelParam get_config() {
  
  string config_file;//
  if (model == "ssd_mobilenet")
    config_file = dpuDir+"/ssd_mobilenet_v1_coco_tf.prototxt";
  vitis::ai::proto::DpuModelParamList mlist;
  auto text = slurp(config_file.c_str());
  auto ok = google::protobuf::TextFormat::ParseFromString(text, &mlist);
  CHECK(ok) << "cannot parse config file. config_file=" << config_file;
  CHECK_EQ(mlist.model_size(), 1)
      << "only support one model per config file."
      << "config_file " << config_file << " "       //
      << "content: " << mlist.DebugString() << " "  //
      ;
  return mlist.model(0);
}
std::string slurp(const char* filename) {
  std::ifstream in;
  in.open(filename, std::ifstream::in);
  CHECK(in.good()) << "failed to read config file. filename=" << filename;
  std::stringstream sstr;
  sstr << in.rdbuf();
  in.close();
  return sstr.str();
}


class ImageCache {
  public:  
    static ImageCache& getInst() {
      static ImageCache imageCache;  
      return imageCache;
    }
    void central_crop (const Mat& image, int height, int width, Mat& img) {
      int offset_h = (image.rows - height)/2;
      int offset_w = (image.cols - width)/2;
      Rect box(offset_w, offset_h, width, height);
      img = image(box);
    }
    void init(std::string imagePath, int numImages) {
    
    imageList_.clear();
    imageList_.reserve(numImages);
    std::string imageListFile;//
      imageListFile = imagePath + "/annotations/instances_val2017.json";
    
      std::ifstream f(imageListFile);
      if(f) {
        std::stringstream buffer;
        buffer << f.rdbuf();
        boost::property_tree::ptree pt;
        boost::property_tree::read_json(buffer,pt);
        for (auto& img : pt.get_child("images")) {

          unsigned int label;
          for (auto& prop : img.second) {
            if (prop.first == "file_name") {
              imageList_.push_back(std::make_pair(0,
                      imagePath+ "/val2017/" + prop.second.get_value<std::string>()));
            }

            // limit dataset
            if (numImages
                    && (imageList_.size() >= (uint) numImages)) {
              break;
            }
          }
        }
      }
    }
    void preProcessSsdmobilenet(Mat &orig, float scale, int width, int height, QuerySampleIndex index) {

        cv::Mat img, float_image;
        if (num_channels_ < 3) {
            cv::cvtColor(orig, float_image,  cv::COLOR_GRAY2RGB);
        } else {
            cv::cvtColor(orig, float_image,  cv::COLOR_BGR2RGB);
        }
 
        cv::resize((float_image), (img),
                cv::Size(image_width_, image_height_), cv::INTER_LINEAR);
        int i = 0;
        for (int c=0; c < 3; c++) {
          for (int h=0; h < height; h++) {
            for (int w=0; w < width; w++) {
#ifdef ENABLE_INT
              inputData_[index].data.get()[0
                + (3*h*width) + (w*3) + c] 
                = (int8_t)((img.at<Vec3b>(h,w)[c]*2/255.0-1.0)*scale);
#else
              inputData_[index].data.get()[0
                + (3*h*width) + (w*3) + c] 
                = img.at<Vec3b>(h,w)[c]*2/255.0-1.0 ;
#endif
            }
          }
        }
        i++;


    }
    void load(float scale, int width, int height, const std::vector<QuerySampleIndex> &samples)
    {
      const int inSize = width * height * 3;
      if (model == "ssd_mobilenet") {
        image_height_ =image_width_ = 300;
      }
      // populate input data
      inputData_.clear();
      inputData_.resize(samples.size());
      for (auto &in: inputData_) 
#ifdef ENABLE_INT
        in.data.reset(new int8_t[inSize]);
#else
        in.data.reset(new float[inSize]);
#endif
      for (auto index : samples) 
      {
        const std::string &filename = imageList_[index].second;
        inputData_[index].image = filename;
        /* pre-process */
        //Mat orig = imread("/home/gaoyue/image_temp/001.jpg");
        Mat orig = imread(filename);
        if (model == "ssd_mobilenet")
          preProcessSsdmobilenet(orig, scale, width, height, index);
      }

      std::cout << "Loaded samples: " << inputData_.size() << std::endl;
    }

    void unload() {
      inputData_.clear();
    }
    InputData& get(int idx) {
    //T& get(int idx) {
      assert(idx < inputData_.size());
      return inputData_[idx];
    }
    //}
    int size() const {
      return inputData_.size(); 
    }

     std::vector<std::pair<int, std::string>> imageList_;
  private:
     std::vector<InputData> inputData_;
     std::mutex lock_;
};

class XlnxSystemUnderTest : public SystemUnderTest {
private:
  const std::string name = "XLNX_AI";
  std::unique_ptr<vitis::ai::DpuRunner> runner_;
  vector<std::unique_ptr<vitis::ai::TFSSDPostProcess>> processor_;
  vector<std::unique_ptr<vitis::ai::SSDPostProcess>> processor2_;
  vitis::ai::proto::DpuModelParam config_;
  std::vector<vitis::ai::Tensor*> outputTensors_; 
  std::vector<vitis::ai::Tensor*> inputTensors_; 
//  int outSize1_, outSize2_;

  int inSize_;// = inputTensors[0]->get_element_num() / inputTensors[0]->get_dim_size(0);
  std::vector<shared_ptr<vitis::ai::CpuFlatTensorBuffer>>inputs_, outputs_;
  std::vector<vitis::ai::TensorBuffer*> inputsPtr_, outputsPtr_;
  std::vector<std::shared_ptr<vitis::ai::Tensor>> batchTensors_;
  int8_t* output_;
  float* out_;
  std::vector<std::int32_t> in_dims_;
  vector<float> resout;
  vector<int> outSize_;
  int sumSize;
  QuerySampleResponse res_;
public:
  void init() {
    if (is_server)
      postThread = std::thread(&XlnxSystemUnderTest::postProcess, this);
    outputTensors_ = runner_->get_output_tensors();
    inputTensors_ = runner_->get_input_tensors();
    in_dims_ = inputTensors_[0]->get_dims();
    in_dims_[0] = batchSize;//out_dims1_[0] = out_dims2_[0]= batchSize;
    sumSize = 0;
    for (int i=0;i<outputTensors_.size();i++) {
      outSize_.push_back(outputTensors_[i]->get_element_num() / outputTensors_[i]->get_dim_size(0));
      sumSize +=outSize_[i];
    }
    inSize_ = inputTensors_[0]->get_element_num() / inputTensors_[0]->get_dim_size(0);
    int8_t* output = new int8_t[sumSize];
    output_ = output;
    int idx = 0;
    for (int i=0;i<outputTensors_.size();i++) {
      auto out_dims1 = outputTensors_[i]->get_dims();
      batchTensors_.push_back(std::shared_ptr<vitis::ai::Tensor>(
            new vitis::ai::Tensor(outputTensors_[i]->get_name(), out_dims1, 
                           outputTensors_[i]->get_data_type())));
      outputs_.push_back(make_shared<vitis::ai::CpuFlatTensorBuffer>(output_+idx, batchTensors_.back().get()));
      outputsPtr_.push_back(outputs_[i].get());
      idx += outSize_[i];
    }

    config_ = get_config();
    for (int i=0; i< threadnum; i++) {

      if (model == "ssd_mobilenet")
        processor_.push_back( vitis::ai::TFSSDPostProcess::create(
          image_width_, image_height_, (runner_->get_output_tensors())[0]->get_scale(), (runner_->get_output_tensors())[1]->get_scale(), config_));
    }
  }
  void IssueQuerySingle(const std::vector<QuerySample> &samples) {
    const int batchSize = 1;
    ImageCache &imageCache = ImageCache::getInst();
    const QuerySample &sample = samples[0];

    auto &input = imageCache.get(sample.index);
    auto data = input.data.get();
    batchTensors_.push_back(std::shared_ptr<vitis::ai::Tensor>(
          new vitis::ai::Tensor(inputTensors_[0]->get_name(), in_dims_, 
                         inputTensors_[0]->get_data_type())));
    inputs_.push_back(make_shared<vitis::ai::CpuFlatTensorBuffer>(data, batchTensors_.back().get()));

    inputsPtr_.push_back(inputs_[0].get());
    __TIC_MLPERF__(MLPERF_RUNTIME)
    auto job_id = runner_->execute_async(inputsPtr_,outputsPtr_);
    runner_->wait(job_id.first, -1);
    __TOC_MLPERF__(MLPERF_RUNTIME)
    unsigned int count = 0;
    if (model == "ssd_mobilenet" ) {
      __TIC_MLPERF__(MLPERF_POST)
      auto results = processor_[0]->ssd_post_process(output_,output_+outSize_[0]);
      for (auto &box :  results[0].bboxes) {

        resout.push_back(float(sample.index));
        resout.push_back(float(box.y));
        resout.push_back(float(box.x));
        resout.push_back(float(box.y+box.height));
        resout.push_back(float(box.x+box.width));
        resout.push_back(float(box.score));
        resout.push_back(int(box.label));
        ++count;
      }
      __TOC_MLPERF__(MLPERF_POST)
    }
    res_.id = sample.id;
    res_.data = reinterpret_cast<uintptr_t>(&resout[0]);
    res_.size = sizeof(float)*count*7;
    QuerySamplesComplete(&res_, 1);
    inputsPtr_.clear();
    resout.clear();
    inputs_.clear();
  }
  XlnxSystemUnderTest(std::string runnerDir) {
    auto runners = vitis::ai::DpuRunner::create_dpu_runner(runnerDir); 
    runner_ = std::move(runners[0]);
  }
  ~XlnxSystemUnderTest() {
    if (is_server) {
      is_server = 0;
      cout << "Finish server testing" << endl;
      if(postThread.joinable())
      {

        postThread.join();
      }
    }

  }

  const std::string &Name() const { return name; }
  mutex mtx_post_queue1;
  template <typename T>
  vector<unsigned int> runBatch( int index,int runSize,vector<T*> dataInput, std::vector<float> &results, std::vector<int> samples_idx) {
    
    auto outputTensors = runner_->get_output_tensors();
    auto inputTensors = runner_->get_input_tensors();
    auto in_dims = inputTensors[0]->get_dims();
    in_dims[0] = runSize;

    T* data = new T[runSize*inSize_];
    for (int i=0; i < runSize; i++) {
      for (int j=0;j<inSize_;j++) {   
        data[i*inSize_+j] = dataInput[i][j];
      }
    }
    std::vector<std::shared_ptr<vitis::ai::Tensor>> batchTensors;
    std::vector<shared_ptr<vitis::ai::CpuFlatTensorBuffer>>inputs, outputs;
    std::vector<vitis::ai::TensorBuffer*> inputsPtr, outputsPtr;
    std::vector<vitis::ai::TensorBuffer*>  outputsPtrPost;
    batchTensors.push_back(std::shared_ptr<vitis::ai::Tensor>(
          new vitis::ai::Tensor(inputTensors[0]->get_name(), in_dims,
                         inputTensors[0]->get_data_type())));
    inputs.push_back(make_shared<vitis::ai::CpuFlatTensorBuffer>(data, batchTensors.back().get()));
    inputsPtr.push_back(inputs[0].get());
    shared_ptr<int8_t[]> out1;
    out1.reset(new int8_t[sumSize*runSize]);
    int idx = 0;
    for (int tsize=0; tsize<outputTensors.size(); tsize++) {
      auto out_dims = outputTensors[tsize]->get_dims();
      out_dims[0] = runSize;
      batchTensors.push_back(std::shared_ptr<vitis::ai::Tensor>(
            new vitis::ai::Tensor(outputTensors[tsize]->get_name(), out_dims,
                           outputTensors[tsize]->get_data_type())));
      outputs.push_back(make_shared<vitis::ai::CpuFlatTensorBuffer>(out1.get()+idx*runSize, batchTensors.back().get()));
      outputsPtr.push_back(outputs[tsize].get());
      idx+=outSize_[tsize];
    }
    auto job_id = runner_->execute_async(inputsPtr, outputsPtr);
    runner_->wait(job_id.first, -1);
    vector<unsigned int> rtn;
    for (int i=0; i < runSize; i++) {
      unsigned int count = 0;
      if (model == "ssd_mobilenet" ) {
        auto result = processor_[index]->ssd_post_process(out1.get()+i*outSize_[0],out1.get()+outSize_[0]*runSize+i*outSize_[1]);
        for (auto &box :  result[0].bboxes) {
  
          results.push_back(float(samples_idx[i]));
          results.push_back(float(box.y));
          results.push_back(float(box.x));
          results.push_back(float(box.y+box.height));
          results.push_back(float(box.x+box.width));
          results.push_back(float(box.score));
          results.push_back(int(box.label));
          ++count;
        }
      }
  
      rtn.push_back(count*7);
    }
    inputsPtr.clear();
    outputsPtr.clear();
    inputs.clear();
    outputs.clear();
    delete[] data;
    return rtn;
   
  }
  template<typename T>
  void IssueQueryMulti(const std::vector<QuerySample> &samples) {
    ImageCache &imageCache = ImageCache::getInst();
    thread workers[threadnum];
    for (int i =0; i< threadnum;i++) {
      workers[i] = thread([&,i]() {
        for (int k=i*batchSize;k<samples.size();k+=threadnum*batchSize) {
        
          unsigned int runSize = (samples.size() < (k+batchSize))? (samples.size()-k) : batchSize;
          vector<T*> dataInput;
          vector<int> samples_idx;
          for ( int b=0;b<runSize;b++){
            const QuerySample &sample = samples[k+b];
            auto &input = imageCache.get(sample.index);
            auto data = input.data.get();
            dataInput.push_back(data);
            samples_idx.push_back(sample.index);
          }
          std::vector<float> results; 
          auto counts =  runBatch<T>(i, runSize,dataInput,results,samples_idx);
          unsigned int idx = 0;
          std::vector<mlperf::QuerySampleResponse> responses;
          for (int j=0;j<runSize;j++) {
            const QuerySample &sample = samples[k+j];
            QuerySampleResponse res {sample.id, reinterpret_cast<uintptr_t>(&results[idx]),(sizeof(float) * counts[j])};
            mtx_post_queue1.lock();
            responses.push_back(res);
            mtx_post_queue1.unlock();
            idx += counts[j];
          }
          mtx_post_queue1.lock();
          QuerySamplesComplete(responses.data(), responses.size());
          responses.clear();
          mtx_post_queue1.unlock();
          dataInput.clear();
          samples_idx.clear();
          results.clear();
          counts.clear();
        }
      });
    }
    for (auto &w : workers) {
      if (w.joinable()) w.join();
       
    }
  }
  vector<long unsigned int> samples_idx;
  queue<pair<int8_t*, int>> postqueue;
  queue<uint32_t> jobids;
  queue<int> postqueue2;
  mutex mtx_post_queue;
  void postProcess() {
    while (is_server) {
      uint32_t job_id;
      int sample_idx;
      long unsigned int sample_id;
      int8_t* out1;
      mtx_post_queue.lock();
      if (postqueue.empty()) {
        mtx_post_queue.unlock();
      } else {
        job_id = jobids.front();
        sample_idx = postqueue.front().second;
        if (model == "ssd_mobilenet")
          out1 = postqueue.front().first;
        postqueue.pop();
        jobids.pop();
        sample_id = postqueue2.front();
        postqueue2.pop();
        mtx_post_queue.unlock();
        runner_->wait(job_id, -1);
        std::vector<float> results;
        std::vector < mlperf::QuerySampleResponse > responses;
        vector<unsigned int> rtn;
        unsigned int count = 0;
        if (model == "ssd_mobilenet" ) {
          auto result = processor_[0]->ssd_post_process(out1,out1+outSize_[0]*1);

          for (auto &box :  result[0].bboxes) {
            results.push_back(float(sample_idx));
            results.push_back(float(box.y));
            results.push_back(float(box.x));
            results.push_back(float(box.y+box.height));
            results.push_back(float(box.x+box.width));
            results.push_back(float(box.score));
            results.push_back(int(box.label));
            ++count;
          }
        }
        rtn.push_back(count*7);
        delete[] out1;
        QuerySampleResponse res {sample_id, reinterpret_cast<uintptr_t>(&results[0]),(sizeof(float) * rtn[0])};
        responses.push_back(res);
        QuerySamplesComplete(responses.data(), responses.size());
        responses.clear();
        results.clear();
        rtn.clear();
      }
    }
  }

  template<typename T>
  void IssueQueryServer(const std::vector<QuerySample> &samples) {
    ImageCache &imageCache = ImageCache::getInst();
    unsigned int runSize = (samples.size() < batchSize)? samples.size() : batchSize;
    vector<T*> dataInput;
    for ( int b=0;b<runSize;b++){
      const QuerySample &sample = samples[b];
      auto &input = imageCache.get(sample.index);
      auto data = input.data.get();
      dataInput.push_back(data);
      samples_idx.push_back(sample.index);
    }
    auto outputTensors = runner_->get_output_tensors();
    auto inputTensors = runner_->get_input_tensors();
    auto in_dims = inputTensors[0]->get_dims();
    in_dims[0] = runSize;

    T* data = new T[runSize*inSize_];
    for (int i=0; i < runSize; i++) {
      for (int j=0;j<inSize_;j++) {
        data[i*inSize_+j] = dataInput[i][j];
      }
    }
    std::vector<shared_ptr<vitis::ai::CpuFlatTensorBuffer>>inputs, outputs;
    std::vector<vitis::ai::TensorBuffer*> inputsPtr, outputsPtr;
    std::vector<std::shared_ptr<vitis::ai::Tensor>> batchTensors;
    batchTensors.push_back(std::shared_ptr<vitis::ai::Tensor>(
          new vitis::ai::Tensor(inputTensors_[0]->get_name(), in_dims,
                         inputTensors_[0]->get_data_type())));
    inputs.push_back(make_shared<vitis::ai::CpuFlatTensorBuffer>(data, batchTensors.back().get()));
    inputsPtr.push_back(inputs[0].get());
    std::vector<int> jobIds;
    int8_t* out1;
    int8_t* out = new int8_t[sumSize*runSize];
    out1 = out;

    int idx = 0;
    for (int tsize=0; tsize<outputTensors.size(); tsize++) {
      auto out_dims = outputTensors[tsize]->get_dims();
      out_dims[0] = runSize;
      batchTensors.push_back(std::shared_ptr<vitis::ai::Tensor>(
            new vitis::ai::Tensor(outputTensors[tsize]->get_name(), out_dims,
                           outputTensors[tsize]->get_data_type())));
      outputs.push_back(make_shared<vitis::ai::CpuFlatTensorBuffer>(out1+idx*runSize, batchTensors.back().get()));
      outputsPtr.push_back(outputs[tsize].get());
      idx+=outSize_[tsize];
    }

    auto job_id = runner_->execute_async(inputsPtr, outputsPtr);
    const QuerySample &sample = samples[0];

    mtx_post_queue.lock();
    if (model == "ssd_mobilenet")
      postqueue.push(make_pair(out1,samples_idx[0]));
    postqueue2.push(sample.id);
    jobids.push(job_id.first);
    mtx_post_queue.unlock();
    dataInput.clear();
    samples_idx.clear();
  }
  void IssueQuery(const std::vector<QuerySample> &samples) {
    if (modeScenario == 1 ) {
#ifdef ENABLE_INT
      IssueQueryMulti<int8_t>(samples);
#else
      IssueQueryMulti<float>(samples);
#endif
    }
    else if (modeScenario == 2) {
#ifdef ENABLE_INT
      IssueQueryServer<int8_t>(samples);
#else
      IssueQueryServer<float>(samples);
#endif
    }
    else {
      IssueQuerySingle(samples);
    }
  }
  int getWidth() {
    auto inputTensors = runner_->get_input_tensors();
    if (runner_->get_tensor_format() == vitis::ai::DpuRunner::TensorFormat::NCHW)
      return inputTensors[0]->get_dim_size(3);
    else
      return inputTensors[0]->get_dim_size(2);
  }
  int getHeight() {
    auto inputTensors = runner_->get_input_tensors();
    if (runner_->get_tensor_format() == vitis::ai::DpuRunner::TensorFormat::NCHW)
      return inputTensors[0]->get_dim_size(2);
    else
      return inputTensors[0]->get_dim_size(1);
  }
  int getScale() {
    auto inputTensors = runner_->get_input_tensors();
    return inputTensors[0]->get_scale();
    //return 1;
  }
  void FlushQueries() {}
  void ReportLatencyResults(const std::vector<QuerySampleLatency> &/*latencies_ns*/) {}
};

class XlnxQuerySampleLibrary : public QuerySampleLibrary {
private:
  const std::string name = "XLNX_AI";
  const int width_;
  const int height_;
  const int numSamples_;
  const float scale_;
public:
  XlnxQuerySampleLibrary(const std::string &path, int width, int height, float scale, int nSamples):
    width_(width), height_(height),scale_(scale), numSamples_(nSamples) {

    ImageCache &imageCache = ImageCache::getInst();
    imageCache.init(path, nSamples);
  }

  ~XlnxQuerySampleLibrary() {}

  const std::string &Name() const { return name; }

  size_t TotalSampleCount() { 
    return numSamples_;
  }

  size_t PerformanceSampleCount() { 
    return std::min(5000, numSamples_);
  }

  void LoadSamplesToRam(const std::vector<QuerySampleIndex> &samples) override {
    ImageCache::getInst().load(scale_, width_, height_, samples);
  }

  void UnloadSamplesFromRam(const std::vector<QuerySampleIndex> &samples) override {
    ImageCache::getInst().unload();
  }
};

/* 
 * Usage: 
 * app.exe <options>
 */
int main(int argc, char **argv) {
  TestSettings testSettings = TestSettings();
  testSettings.scenario = TestScenario::SingleStream;
  testSettings.mode = TestMode::PerformanceOnly;
  testSettings.min_query_count = 1024;
//  testSettings.max_query_count = 1000;
  testSettings.min_duration_ms = 60000;
  testSettings.multi_stream_max_async_queries 
    = testSettings.server_max_async_queries 
    = 1;
  testSettings.multi_stream_target_qps = 20;
  testSettings.multi_stream_samples_per_query = 64;
  testSettings.server_target_latency_ns = 10000000;
  testSettings.server_target_qps = 100;
  testSettings.offline_expected_qps = 100;
  testSettings.qsl_rng_seed = 12786827339337101903ULL;
  testSettings.schedule_rng_seed = 3135815929913719677ULL;
  testSettings.sample_index_rng_seed = 12640797754436136668ULL;
  
  LogSettings logSettings = LogSettings();
  logSettings.enable_trace = false;
  batchSize = 1;
  threadnum = 1; 
  modeScenario = 0;
  int numSamples = -1;
  for(;;) {
    struct option long_options[] = {
      //{"verbose", no_argument,       &verbose_flag, 1},
      {"imgdir", required_argument, 0, 'i'},
      {"dpudir", required_argument, 0, 'd'},
      {"logtrace", no_argument, 0, 'l'},
      {"rngseed", no_argument, 0, 'g'},
      {"num_queries", required_argument, 0, 'q'},
      {"num_samples", required_argument, 0, 's'},
      {"max_async_queries", required_argument, 0, 'a'},
      {"min_time", required_argument, 0, 't'},
      {"scenario", required_argument, 0, 'c'},
      {"mode", required_argument, 0, 'm'},
      {"samples_per_q", required_argument,0,'p'},
      {"batch", required_argument,0,'b'},
      {"thread_num",required_argument,0,'n'},
      {"qps", required_argument,0,'r'},
      {"unique", required_argument,0,'u'},
      {0, 0, 0, 0}
    };
    /* getopt_long stores the option index here. */
    int option_index = 0;
    int c = getopt_long (argc, argv, "r:b:n:p:a:c:d:i:q:s:t:m:", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
      case 'r':
        testSettings.server_target_qps = stoi(optarg);
        testSettings.offline_expected_qps = stoi(optarg);
      case 'b':
        batchSize = stoi(optarg);
        break;
      case 'n':
        threadnum = stoi(optarg);
        break; 
      case 'p':
        testSettings.multi_stream_samples_per_query = stoi(optarg);
        break;
      case 'a':
        testSettings.multi_stream_max_async_queries 
          = testSettings.server_max_async_queries 
          = atoi(optarg);
      case 'c':
        if (std::string(optarg) == "SingleStream") {
          testSettings.scenario = TestScenario::SingleStream;
          modeScenario = 0;
        }
        else if (std::string(optarg) == "MultiStream") {
          testSettings.scenario = TestScenario::MultiStream;
          modeScenario = 1;
        }
        else if (std::string(optarg) == "Server") {
          testSettings.scenario = TestScenario::Server;
          modeScenario = 2;
          is_server = 1;
        }
        else if (std::string(optarg) == "Offline") {
          testSettings.scenario = TestScenario::Offline;
          modeScenario = 1;
        }
        break;

      case 'd':
        dpuDir = std::string(optarg);
        if (dpuDir == "model_ssd_mobilenet")
          model = "ssd_mobilenet";
        break;

      case 'i':
        imgDir = std::string(optarg);
        break;

      case 'l':
        logSettings.enable_trace = true;
        break;

      case 'q':
        //testSettings.min_query_count = testSettings.max_query_count = atoi(optarg);
        testSettings.min_query_count = atoi(optarg);
        break;

      case 's':
        numSamples = atoi(optarg);
        break;

      case 't':
        testSettings.min_duration_ms = atoi(optarg);
        break;

      case 'm':
        if (std::string(optarg) == "SubmissionRun")
          testSettings.mode = TestMode::SubmissionRun;
        else if (std::string(optarg) == "AccuracyOnly")
          testSettings.mode = TestMode::AccuracyOnly;
        else if (std::string(optarg) == "PerformanceOnly")
          testSettings.mode = TestMode::PerformanceOnly;
    }
  }
  
  if (model == "ssd_mobilenet") {
    image_height_ =image_width_ = 300;
  }
  if (numSamples < 0)
    numSamples = testSettings.multi_stream_samples_per_query * testSettings.multi_stream_max_async_queries * 2;
  XlnxSystemUnderTest *sut = new XlnxSystemUnderTest(dpuDir);
  sut->init();
  QuerySampleLibrary *qsl = new XlnxQuerySampleLibrary(imgDir,
    sut->getWidth(), sut->getHeight(), sut->getScale(), numSamples);

#if _HW_SORT_EN_
  hw_sort_init(pphandle, "/usr/lib/dpu.xclbin");
#endif

  cout << "Query count: " << testSettings.min_query_count << "\n";
  cout << "Start loadgen\n";
  StartTest(sut, qsl, testSettings, logSettings);

  delete qsl;
  delete sut;

  return 0;
}
