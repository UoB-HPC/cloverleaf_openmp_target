/*
 Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

 CloverLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 CloverLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 CloverLeaf. If not, see http://www.gnu.org/licenses/.
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <utility>
#include <cassert>
#include <vector>

namespace clover {

#ifdef USE_VECTOR

    template<typename T>
    struct Buffer1D {

        const size_t size;
        std::vector<T> data;

        explicit Buffer1D(size_t size) : size(size), data(size) {}
        T operator[](size_t i) const { return data[i]; }
        T &operator[](size_t i) { return data[i]; }
        T *actual() { return data.data(); }

        friend std::ostream &operator<<(std::ostream &os, const Buffer1D<T> &buffer) {
            os << "Buffer1D(size: " << buffer.size << ")";
            return os;
        }

    };

    template<typename T>
    struct Buffer2D {

        const size_t sizeX, sizeY;
        std::vector<T> data;

        Buffer2D(size_t sizeX, size_t sizeY) : sizeX(sizeX), sizeY(sizeY), data(sizeX * sizeY) {}
        T &operator()(size_t i, size_t j) { return data[i + j * sizeX]; }
        T const &operator()(size_t i, size_t j) const { return data[i + j * sizeX]; }
        T *actual() { return data.data(); }

        friend std::ostream &operator<<(std::ostream &os, const Buffer2D<T> &buffer) {
            os << "Buffer2D(sizeX: " << buffer.sizeX << " sizeY: " << buffer.sizeY << ")";
            return os;
        }

    };

#else

    template<typename T>
    struct Buffer1D {

        const size_t size;
        T *data;

        explicit Buffer1D(size_t size) : size(size), data(static_cast<T *>(std::malloc(sizeof(T) * size))) {}
        Buffer1D(const Buffer1D &that) : Buffer1D(that.size) { std::copy(that.data, that.data + size, data);  }
        Buffer1D &operator=(const Buffer1D &other) = delete;
        T operator[](size_t i) const { return data[i]; }
        T &operator[](size_t i) { return data[i]; }
        T *actual() { return data; }
        virtual ~Buffer1D() { std::free(data); }

        friend std::ostream &operator<<(std::ostream &os, const Buffer1D<T> &buffer) {
            os << "Buffer1D(size: " << buffer.size << ")";
            return os;
        }

    };

    template<typename T>
    struct Buffer2D {

        const size_t sizeX, sizeY;
        T *data;

        Buffer2D(size_t sizeX, size_t sizeY) : sizeX(sizeX), sizeY(sizeY),  data(static_cast<T *>(std::malloc(sizeof(T) * sizeX * sizeY))) {}
        Buffer2D(const Buffer2D &that) : Buffer2D(that.sizeX, that.sizeY) { std::copy(that.data, that.data + sizeX * sizeY, data); }
        Buffer2D &operator=(const Buffer2D &other) = delete;
        T &operator()(size_t i, size_t j) { return data[i + j * sizeX]; }
        T const &operator()(size_t i, size_t j) const { return data[i + j * sizeX]; }
        T *actual() { return data; }
        virtual ~Buffer2D() { std::free(data); }

        friend std::ostream &operator<<(std::ostream &os, const Buffer2D<T> &buffer) {
            os << "Buffer2D(sizeX: " << buffer.sizeX << " sizeY: " << buffer.sizeY << ")";
            return os;
        }
    };

#endif



}


using namespace clover;

#endif //UTILS_HPP
