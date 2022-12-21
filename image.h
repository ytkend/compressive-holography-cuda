#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

template <class T>
struct Image {
    int width;
    int height;
    int channels;
    std::vector<T> pixels;

    Image(int width, int height, int channels=1)
        : width(width), height(height), channels(channels), pixels(size(), static_cast<T>(0)) {}

    Image(T* data, int width, int height, int channels)
        : width(width), height(height), channels(channels), pixels(data, data + size()) {}

    template <class U>
    Image(U* data, int width, int height, int channels)
        : width(width), height(height), channels(channels), pixels(data, data + size()) {}

    int size() const { return width * height * channels; }
    T *data() { return pixels.data(); }
    const T* data() const { return pixels.data(); }
    T& at(int x, int y, int c=0) { return pixels[c + x * channels + y * channels * width]; }
    const T& at(int x, int y, int c=0) const { return pixels[c + x * channels + y * channels * width]; }
};

template <class T>
std::vector<T> normalize(const std::vector<T>& in, T min=0, T max=1)
{
    auto minmax_val = std::minmax_element(in.data(), in.data() + in.size());
    auto min_val = *minmax_val.first;
    auto max_val = *minmax_val.second;
    auto a = (max - min) / (max_val - min_val);
    std::vector<T> out(in.size());
    std::transform(in.begin(), in.end(), out.begin(), [=](auto&& i){return (i - min_val) * a + min;});
    return out;
}

template <class T>
Image<T> imread(const std::string& path)
{
    int w, h, n;
    uint8_t *data = stbi_load(path.c_str(), &w, &h, &n, 0);
    if (data == nullptr) {
        throw std::runtime_error("Failed to open: " + path);
    }
    Image<T> img(data, w, h, n);
    stbi_image_free(data);
    return img;
}

template <class T>
void imsave(const std::string& path, T* data, int w, int h, int c=1, bool norm=true)
{
    std::vector<T> pixel_data(data, data + w * h * c);
    if (norm) {
        pixel_data = normalize<T>(pixel_data, 0, 255);
    }
    std::vector<uint8_t> pixel_data_8bit(pixel_data.begin(), pixel_data.end());

    std::string ext = std::filesystem::path(path).extension();
    if (ext == ".png") {
        stbi_write_png(path.c_str(), w, h, c, pixel_data_8bit.data(), w * c);
    } else if (ext == ".bmp") {
        stbi_write_bmp(path.c_str(), w, h, c, pixel_data_8bit.data());
    } else {
        throw std::runtime_error("Unsupported format: " + path);
    }
}
