//
// Created by Мороз Максим on 12.05.2020.
//

#ifndef IMAGENET_CPP_IMAGENETLABELS_H
#define IMAGENET_CPP_IMAGENETLABELS_H

#include <map>
#include <string>
#include <fstream>
#include <list>

namespace imagenet {
    class ImageNetLabels {
    public:
        ImageNetLabels() = default;

        void load(std::string &str);

        void load(std::ifstream &file);

        std::string operator[](int cls) const;

    private:
        std::map<int, std::string> labels_;
    };
}

#endif //IMAGENET_CPP_IMAGENETLABELS_H
