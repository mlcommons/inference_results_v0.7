//
// Created by Мороз Максим on 21.05.2020.
//

#include <iostream>

#include "ImageNetUtils.h"

using namespace std;
using namespace cv;

namespace imagenet {

    list<ImageClass> get_classes(const TPUTensor &tensor, size_t top_k, double threshold)
    {
        char *p = (char *)tensor.data;
        vector<char> output(p, p + tpu_tensor_get_size(&tensor));

        auto indices = argsort(output);
        list<ImageClass> classes_list;
        for(auto r = indices.rbegin(); r != indices.rend() && top_k != 0; ++r, --top_k) {
            classes_list.push_back({*r + 1, double(output[*r])});
        }
        return classes_list;
    }

    Mat central_crop_scale(const Mat &image, double ratio) {
        auto width = image.size().width;
        auto height = image.size().height;
        auto min_dim = static_cast<int>(std::min(width, height) * ratio);
        int crop_top = (height - min_dim) / 2;
        int crop_left = (width - min_dim) / 2;
        Rect myROI(crop_left, crop_top, min_dim, min_dim);
        return image(myROI);
    }

    Mat crop_and_resize(const Mat &image, size_t square_size) {
        Mat output;
        auto img = central_crop_scale(image);
        resize(img, output, Size(square_size, square_size), INTER_CUBIC);
        return output;
    }

    void readTpuTensorFromMat(const Mat &mat, TPUTensor &outTensor) {
        Mat fakeMat(mat.rows, mat.cols, CV_8SC3, outTensor.data);
        mat.convertTo(fakeMat, CV_8SC3);
    }

    void image_to_tensor(Mat image, TPUTensor *tensor, double scale) {
        // resize and make rgb format
        image = crop_and_resize(image, 224);
        cvtColor(image, image, COLOR_BGR2RGB);

        // subtract mean
        image.convertTo(image, CV_32FC3);
        divide(image, Scalar(255.0, 255.0, 255.0), image);
        subtract(image, Scalar(0.5, 0.5, 0.5), image);
        multiply(image, Scalar(2.0 * scale, 2.0 * scale, 2.0 * scale), image);
        // copy data to tensor
        readTpuTensorFromMat(image, *tensor);
    }

    ImageLoader::iterator& ImageLoader::iterator::operator++() {
        try {
            image = loader_->next();
        }
        catch (const ImageLoader::stop_iteration& e) {
            eof = true;
        }
        return *this;
    }

    ImageLoader::iterator::iterator(ImageLoader *loader) {
        try {
            loader_ = loader;
            image = loader->next();
            eof = false;
        }
        catch (const ImageLoader::stop_iteration& e) {
            cerr << "hello" << endl;
            loader_ = nullptr;
            image = cv::Mat();
            eof = true;
        }
    }

    bool operator==(const ImageLoader::iterator &lhs, const ImageLoader::iterator &rhs) {
        return(lhs.eof && rhs.eof);
    }

    bool operator!=(const ImageLoader::iterator &lhs, const ImageLoader::iterator &rhs) {
        return !operator==(lhs, rhs);
    }

    Mat ImageLoader::next() {
        if(iter++ == 0) {
            return imread(path_);
        }
        throw stop_iteration("Stop Iteration");
    }

    ImageLoader::iterator ImageLoader::begin() {
        return ImageLoader::iterator(this);
    }

    ImageLoader::iterator ImageLoader::end() {
        return ImageLoader::iterator();
    }

    ImageLoader::iterator::value_type ImageLoader::iterator::operator*() const {
        return image;
    }

    ImageLoader::iterator::pointer ImageLoader::iterator::operator->() {
        return &image;
    }

    ImageLoader::stop_iteration::stop_iteration(const string &string) : runtime_error(string) {}

    cv::Mat ImageRepeatLoader::next() {
        if(iter++ < count_) {
            cv::Mat image = imread(path_);
            if ( !image.data ) {
                throw std::runtime_error(std::string("Can't open image ") + path_) ;
            }
            return image;
        }
        throw stop_iteration("Stop Iteration");
    }
}