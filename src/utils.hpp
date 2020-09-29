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

	template<typename T>
	struct Buffer1D {

		std::vector<T> data;

		explicit Buffer1D(size_t size) : data(size) {}

		inline T operator[](size_t i) const { return data[i]; }
		inline T &operator[](size_t i) { return data[i]; }

		inline T *actual() { return data.data(); }

		[[nodiscard]] constexpr inline size_t size() const { return data.size(); }

//		friend std::ostream &operator<<(std::ostream &os, const Buffer1D<T> &buffer) {
//			os << "Buffer1D(size: " << buffer.size << ")";
//			return os;
//		}

	};

	template<typename T>
	struct Buffer2D {

		const size_t sizeX, sizeY;
		std::vector<T> data;

		Buffer2D(size_t sizeX, size_t sizeY) : sizeX(sizeX), sizeY(sizeY), data(sizeX * sizeY) {}

		inline T &operator()(size_t i, size_t j) { return data[i + j * sizeX]; }
		inline T const &operator()(size_t i, size_t j) const { return data[i + j * sizeX]; }

		T *actual() { return data.data(); }

//		friend std::ostream &operator<<(std::ostream &os, const Buffer2D<T> &buffer) {
//			os << "Buffer2D(sizeX: " << buffer.sizeX << " sizeY: " << buffer.sizeY << ")";
//			return os;
//		}


	};


}


using namespace clover;

#endif //UTILS_HPP
