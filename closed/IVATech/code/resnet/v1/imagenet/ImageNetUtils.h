//
// Created by Мороз Максим on 21.05.2020.
//

#ifndef IMAGENET_CPP_IMAGENETUTILS_H
#define IMAGENET_CPP_IMAGENETUTILS_H

#include <algorithm>
#include <iterator>
#include <list>
#include <vector>
#include <numeric>

extern "C" {
#include <tpu_tensor.h>
};

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace imagenet {
    struct ImageClass {
        size_t index;
        double weight;
    };

    template <typename T>
    std::vector<size_t> argsort(const std::vector<T> &v) {

        // initialize original index locations
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // sort indexes based on comparing values in v
        // using std::stable_sort instead of std::sort
        // to avoid unnecessary index re-orderings
        // when v contains elements of equal values
        stable_sort(idx.begin(), idx.end(),
                    [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

        return idx;
    }

    std::list<ImageClass> get_classes(const TPUTensor &tensor, size_t top_k, double threshold);
    cv::Mat central_crop_scale(const cv::Mat &image, double ratio = 0.85);

    cv::Mat crop_and_resize(const cv::Mat &image, size_t square_size);

    void image_to_tensor(cv::Mat image, TPUTensor *tensor, double scale);

    class ImageLoader {
    public:
        ImageLoader(const std::string& path) : path_(path), iter(0) {};
        virtual cv::Mat next();
        virtual ~ImageLoader() = default;
    protected:
        std::string path_;
        size_t iter;
    public:
        class stop_iteration : public std::runtime_error {
        public:
            stop_iteration(const std::string &string);
        };

        class iterator {
        public:
            using iterator_category = std::input_iterator_tag;
            using value_type = cv::Mat; // crap
            using difference_type = ptrdiff_t;
            using pointer = cv::Mat *;
            using reference = cv::Mat&;

            // rest of the iterator class...
            iterator(): loader_(nullptr), eof(true) {};
            iterator(ImageLoader* loader);
            iterator& operator++();
            value_type operator*() const;
            pointer operator->();
            friend bool operator==(const iterator&, const iterator&);
            friend bool operator!=(const iterator&, const iterator&);
        private:
            ImageLoader *loader_;
            bool eof;
            value_type image;
        };
    public:
        iterator begin();
        iterator end();
    };

    class ImageRepeatLoader: public ImageLoader {
    public:
        ImageRepeatLoader(const std::string& path, size_t count): ImageLoader(path), count_(count) {};
        virtual cv::Mat next() override;
        virtual ~ImageRepeatLoader() = default;
    protected:
        size_t count_;
    };

    class ImageDirectoryLoader: public ImageLoader {
    public:
        ImageDirectoryLoader(const std::string& path, size_t number);
        virtual cv::Mat next() override;
        virtual ~ImageDirectoryLoader() {};
    };
}
#endif //IMAGENET_CPP_IMAGENETUTILS_H
