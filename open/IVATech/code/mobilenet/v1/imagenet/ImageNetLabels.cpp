//
// Created by Мороз Максим on 12.05.2020.
//

#include "ImageNetLabels.h"

#include <sstream>

using namespace imagenet;

std::string ImageNetLabels::operator[](int cls) const {
    return labels_.at(cls);
}

void ImageNetLabels::load(std::ifstream& ifs) {
    std::string tmp((std::istreambuf_iterator<char>(ifs)),
            (std::istreambuf_iterator<char>()));
    load(tmp);
}

void ImageNetLabels::load(std::string& str) {
    std::istringstream ss(str);
    std::string line;
    while(getline(ss, line)) {
        std::istringstream liness(line);
        int cls;
        liness >> cls;
        std::string label;
        getline(liness, label);
        label.erase(0, label.find_first_not_of(' '));
        labels_[cls] = label;
    }
}
