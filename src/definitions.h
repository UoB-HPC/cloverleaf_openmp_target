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

#ifndef GRID_H
#define GRID_H

#define DEBUG false
#define FLUSH_BUFFER 0
#define USE_CXX_OPERATORS 0


#include <iostream>
#include <utility>
#include <fstream>
#include <iostream>
#include <chrono>
#include <functional>
#include <cassert>


#define g_ibig 640000
#define g_small (1.0e-16)
#define g_big   (1.0e+21)
#define NUM_FIELDS 15

namespace clover {


	template<typename T>
	struct Buffer1D {

		const size_t size;
		T *data;

		explicit Buffer1D(size_t size) : size(size), data(new T[size]) {
			assert(size > 0);
		}

		Buffer1D(const Buffer1D<T> &that) : size(that.size), data(new T[size]) {
			std::copy(that.data, that.data + size, data);
		}

		Buffer1D(Buffer1D &&other) noexcept: size(other.size), data(std::exchange(other.data, nullptr)) {}

		Buffer1D &operator=(Buffer1D &&other) noexcept {
			size = other.size;
			std::swap(data, other.data);
			return *this;
		}
		#if USE_CXX_OPERATORS
		[[nodiscard]] constexpr T operator[](size_t i) const { return data[i]; }
		[[nodiscard]] constexpr T &operator[](size_t i) { return data[i]; }
		#endif


		[[nodiscard]] constexpr size_t N() const { return size; }


		Buffer1D<T> &operator=(const Buffer1D<T> &other) {
			if (this != &other) {
				delete[] data;
				std::copy(other.data, other.data + size, data);
				size = other.size;
			}
			return *this;
		}

		~Buffer1D() { delete[] data; }
	};


	template<typename T>
	struct Buffer2D {

		const size_t sizeX, sizeY;
		T *data;

		Buffer2D(size_t sizeX, size_t sizeY) : sizeX(sizeX), sizeY(sizeY), data(new T[sizeX * sizeY]) {
			assert(sizeX > 0);
			assert(sizeY > 0);
		}
		Buffer2D(const Buffer2D<T> &that) : sizeX(that.sizeX), sizeY(that.sizeY), data(new T[sizeX * sizeY]) {
			std::copy(that.data, that.data + (sizeX * sizeY), data);
		}

		Buffer2D(Buffer2D &&other) noexcept: sizeX(other.sizeX), sizeY(other.sizeY), data(std::exchange(other.data, nullptr)) {}


		#if USE_CXX_OPERATORS
		[[nodiscard]] constexpr T &operator()(size_t i, size_t j) { return data[i + j * sizeX]; }
		[[nodiscard]] constexpr T const &operator()(size_t i, size_t j) const { return data[i + j * sizeX]; }
		#endif


		[[nodiscard]] constexpr size_t N() const { return sizeX * sizeY; }


		Buffer2D<T> &operator=(const Buffer2D<T> &other) {
			if (this != &other) {
				return *this = Buffer2D(other);
			}
		}

		Buffer2D &operator=(Buffer2D &&other) noexcept {
			sizeX = other.sizeX;
			sizeY = other.sizeY;
			std::swap(data, other.data);
			return *this;
		}


		~Buffer2D() { delete[] data; }

	};

	#if USE_CXX_OPERATORS
	#define idx1(xs, i)    xs[i]
	#define idx2(xs, i, j) xs(i,j)
	#else
	#define idx1(xs, i) xs.data[i]
	#define idx2(xs, i, j) xs.data[(i) + (j) * xs.sizeX]
	#endif


	#define _xstr(s) _str(s)
	#define _str(s) #s

	#define parallel(n) omp target teams distribute parallel for simd collapse(n) device(0)
	#define enable_target(enable) if(target: (enable))


//	#define mapToFrom1D(xs) map(tofrom: xs.data[:0])
//	#define mapToFrom2D(xs) map(tofrom: xs.data[:0]) map(from: xs.sizeX)
//
////	#define mapTo(xs) map(to: xs.data[:xs.N()])
//	#define mapTo1D(xs) map(to: xs.data[:xs.N()])




	#define mapToFrom2Df(f, xs) double * xs = f.xs.data; const int xs##_sizex = f.xs.sizeX;
	#define mapToFrom1Df(f, xs) double * xs = f.xs.data;



	#define mapToFrom2Dfn(f, xs, name) double * name = f.xs.data; const int name##_sizex = f.xs.sizeX;

	#define mapToFrom2Dfe( xs, name) double * name = xs.data; const int name##_sizex = xs.sizeX;
	#define mapToFrom1Dfe(xs, name) double * name = xs.data;
//	#define idx2fn(f, xs, i, j) xs[(i) + (j) * f.xs.sizeX]



	#define idx1f(f, xs, i) xs[i]
	#define idx2f(f, xs, i, j) xs[(i) + (j) * xs##_sizex]

	#define mapTo1D(xs)


	#define omp(xs) _Pragma(_xstr(xs))


}


typedef std::chrono::time_point<std::chrono::system_clock> timepoint;

// current time
static inline timepoint mark() {
	return std::chrono::system_clock::now();
}

// elapsed time since start in milliseconds
static inline double elapsedMs(timepoint start) {
	timepoint end = mark();
	return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0;
}

// writes content of the provided stream to file with name
static inline void record(const std::string &name, const std::function<void(std::ofstream &)> &f) {
	std::ofstream out;
	out.open(name, std::ofstream::out | std::ofstream::trunc);
	f(out);
	out.close();
}

// formats and then dumps content of 1d double buffer to stream
static inline void
show(std::ostream &out, const std::string &name, const clover::Buffer1D<double> &buffer) {
	out << name << "(" << 1 << ") [" << buffer.size << "]" << std::endl;
	out << "\t";
	for (size_t i = 0; i < buffer.size; ++i) {
		out << idx1(buffer, i) << ", ";
	}
	out << std::endl;
}
// formats and then dumps content of 2d double buffer to stream
static inline void
show(std::ostream &out, const std::string &name, clover::Buffer2D<double> &buffer) {
	out << name << "(" << 2 << ") [" << buffer.sizeX << "x" << buffer.sizeY << "]"
	    << std::endl;
	for (size_t i = 0; i < buffer.sizeX; ++i) {
		out << "\t";
		for (size_t j = 0; j < buffer.sizeY; ++j) {
			out << idx2(buffer, i, j) << ", ";
		}
		out << std::endl;
	}
}

enum geometry_type {
	g_rect = 1, g_circ = 2, g_point = 3
};

// In the Fortran version these are 1,2,3,4,-1, but they are used firectly to index an array in this version
enum chunk_neighbour_type {
	chunk_left = 0, chunk_right = 1, chunk_bottom = 2, chunk_top = 3, external_face = -1
};
enum tile_neighbour_type {
	tile_left = 0, tile_right = 1, tile_bottom = 3, tile_top = 3, external_tile = -1
};

// Again, start at 0 as used for indexing an array of length NUM_FIELDS
enum field_parameter {

	field_density0 = 0,
	field_density1 = 1,
	field_energy0 = 2,
	field_energy1 = 3,
	field_pressure = 4,
	field_viscosity = 5,
	field_soundspeed = 6,
	field_xvel0 = 7,
	field_xvel1 = 8,
	field_yvel0 = 9,
	field_yvel1 = 10,
	field_vol_flux_x = 11,
	field_vol_flux_y = 12,
	field_mass_flux_x = 13,
	field_mass_flux_y = 14
};

enum data_parameter {
	cell_data = 1,
	vertex_data = 2,
	x_face_data = 3,
	y_face_data = 4
};

enum dir_parameter {
	g_xdir = 1, g_ydir = 2
};

struct state_type {

	bool defined;

	double density;
	double energy;
	double xvel;
	double yvel;

	geometry_type geometry;

	double xmin;
	double ymin;
	double xmax;
	double ymax;
	double radius;
};

struct grid_type {

	double xmin;
	double ymin;
	double xmax;
	double ymax;

	int x_cells;
	int y_cells;

};

struct profiler_type {

	double timestep = 0.0;
	double acceleration = 0.0;
	double PdV = 0.0;
	double cell_advection = 0.0;
	double mom_advection = 0.0;
	double viscosity = 0.0;
	double ideal_gas = 0.0;
	double visit = 0.0;
	double summary = 0.0;
	double reset = 0.0;
	double revert = 0.0;
	double flux = 0.0;
	double tile_halo_exchange = 0.0;
	double self_halo_exchange = 0.0;
	double mpi_halo_exchange = 0.0;

};

struct field_type {

	clover::Buffer2D<double> density0;
	clover::Buffer2D<double> density1;
	clover::Buffer2D<double> energy0;
	clover::Buffer2D<double> energy1;
	clover::Buffer2D<double> pressure;
	clover::Buffer2D<double> viscosity;
	clover::Buffer2D<double> soundspeed;
	clover::Buffer2D<double> xvel0, xvel1;
	clover::Buffer2D<double> yvel0, yvel1;
	clover::Buffer2D<double> vol_flux_x, mass_flux_x;
	clover::Buffer2D<double> vol_flux_y, mass_flux_y;

	clover::Buffer2D<double> work_array1; // node_flux, stepbymass, volume_change, pre_vol
	clover::Buffer2D<double> work_array2; // node_mass_post, post_vol
	clover::Buffer2D<double> work_array3; // node_mass_pre,pre_mass
	clover::Buffer2D<double> work_array4; // advec_vel, post_mass
	clover::Buffer2D<double> work_array5; // mom_flux, advec_vol
	clover::Buffer2D<double> work_array6; // pre_vol, post_ener
	clover::Buffer2D<double> work_array7; // post_vol, ener_flux

	clover::Buffer1D<double> cellx;
	clover::Buffer1D<double> celldx;
	clover::Buffer1D<double> celly;
	clover::Buffer1D<double> celldy;
	clover::Buffer1D<double> vertexx;
	clover::Buffer1D<double> vertexdx;
	clover::Buffer1D<double> vertexy;
	clover::Buffer1D<double> vertexdy;

	clover::Buffer2D<double> volume;
	clover::Buffer2D<double> xarea;
	clover::Buffer2D<double> yarea;


	explicit field_type(const size_t xrange, const size_t yrange) :
			density0(xrange, yrange),
			density1(xrange, yrange),
			energy0(xrange, yrange),
			energy1(xrange, yrange),
			pressure(xrange, yrange),
			viscosity(xrange, yrange),
			soundspeed(xrange, yrange),

			xvel0(xrange + 1, yrange + 1),
			xvel1(xrange + 1, yrange + 1),
			yvel0(xrange + 1, yrange + 1),
			yvel1(xrange + 1, yrange + 1),
			vol_flux_x(xrange + 1, yrange),
			mass_flux_x(xrange + 1, yrange),
			vol_flux_y(xrange, yrange + 1),
			mass_flux_y(xrange, yrange + 1),

			work_array1(xrange + 1, yrange + 1),
			work_array2(xrange + 1, yrange + 1),
			work_array3(xrange + 1, yrange + 1),
			work_array4(xrange + 1, yrange + 1),
			work_array5(xrange + 1, yrange + 1),
			work_array6(xrange + 1, yrange + 1),
			work_array7(xrange + 1, yrange + 1),
			cellx(xrange),
			celldx(xrange),
			celly(yrange),
			celldy(yrange),

			vertexx(xrange + 1),
			vertexdx(xrange + 1),
			vertexy(yrange + 1),
			vertexdy(yrange + 1),
			volume(xrange, yrange),
			xarea(xrange + 1, yrange),
			yarea(xrange, yrange + 1) {}


};


struct tile_info {

	std::array<int, 4> tile_neighbours;
	std::array<int, 4> external_tile_mask;
	int t_xmin, t_xmax, t_ymin, t_ymax;
	int t_left, t_right, t_bottom, t_top;

};

struct tile_type {

	tile_info info;
	field_type field;

	explicit tile_type(const tile_info &info) :
			info(info),
			// (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+2)
			// XXX see build_field()
			field((info.t_xmax + 2) - (info.t_xmin - 2) + 1,
			      (info.t_ymax + 2) - (info.t_ymin - 2) + 1) {}
};

struct chunk_type {

	// MPI Buffers in device memory

	// MPI Buffers in host memory - to be created with Kokkos::create_mirror_view() and Kokkos::deep_copy()
//	std::vector<double > hm_left_rcv, hm_right_rcv, hm_bottom_rcv, hm_top_rcv;
//	std::vector<double > hm_left_snd, hm_right_snd, hm_bottom_snd, hm_top_snd;
	const std::array<int, 4> chunk_neighbours; // Chunks, not tasks, so we can overload in the future

	const int task; // MPI task
	const int x_min;
	const int y_min;
	const int x_max;
	const int y_max;

	const int left, right, bottom, top;
	const int left_boundary, right_boundary, bottom_boundary, top_boundary;

	clover::Buffer1D<double> left_rcv, right_rcv, bottom_rcv, top_rcv;
	clover::Buffer1D<double> left_snd, right_snd, bottom_snd, top_snd;

	std::vector<tile_type> tiles;

	chunk_type(const std::array<int, 4> &chunkNeighbours,
	           const int task,
	           const int xMin, const int yMin, const int xMax, const int yMax,
	           const int left, const int right, const int bottom, const int top,
	           const int leftBoundary, const int rightBoundary, const int bottomBoundary,
	           const int topBoundary,
	           const int tiles_per_chunk)
			: chunk_neighbours(chunkNeighbours),
			  task(task),
			  x_min(xMin), y_min(yMin), x_max(xMax), y_max(yMax),
			  left(left), right(right), bottom(bottom), top(top),
			  left_boundary(leftBoundary), right_boundary(rightBoundary),
			  bottom_boundary(bottomBoundary), top_boundary(topBoundary),
			  left_rcv(10 * 2 * (yMax + 5)),
			  right_rcv(10 * 2 * (yMax + 5)),
			  bottom_rcv(10 * 2 * (xMax + 5)),
			  top_rcv(10 * 2 * (xMax + 5)),
			  left_snd(10 * 2 * (yMax + 5)),
			  right_snd(10 * 2 * (yMax + 5)),
			  bottom_snd(10 * 2 * (xMax + 5)),
			  top_snd(10 * 2 * (xMax + 5)) {}


};


// Collection of globally defined variables
struct global_config {

	std::vector<state_type> states;

	int number_of_states;

	int tiles_per_chunk;

	int test_problem;

	bool profiler_on;

	double end_time;

	int end_step;

	double dtinit;
	double dtmin;
	double dtmax;
	double dtrise;
	double dtu_safe;
	double dtv_safe;
	double dtc_safe;
	double dtdiv_safe;
	double dtc;
	double dtu;
	double dtv;
	double dtdiv;

	int visit_frequency;
	int summary_frequency;

	int number_of_chunks;

	grid_type grid;

};

struct global_variables {

	const global_config config;

	const size_t omp_device;
	bool use_target;

	chunk_type chunk;

	int error_condition;

	int step = 0;
	bool advect_x = true;
	double time = 0.0;

	double dt;
	double dtold;

	bool complete = false;
	int jdt, kdt;


	bool profiler_on; // Internal code profiler to make comparisons accross systems easier


	profiler_type profiler;


	explicit global_variables(
			const global_config &config,
			size_t omp_device,
			bool use_target,
			chunk_type chunk) :
			config(config), omp_device(omp_device), use_target(use_target), chunk(std::move(chunk)),
			dt(config.dtinit),
			dtold(config.dtinit),
			profiler_on(config.profiler_on) {}

	void hostToDevice() {
		#pragma omp flush

		for (int tile = 0; tile < config.tiles_per_chunk; ++tile) {
			tile_type &t = chunk.tiles[tile];
			field_type &field = t.field;
			#pragma omp target update \
                to(field.density0.data[:field.density0.N()]) \
                to(field.density1.data[:field.density1.N()]) \
                to(field.energy0.data[:field.energy0.N()]) \
                to(field.energy1.data[:field.energy1.N()]) \
                to(field.pressure.data[:field.pressure.N()]) \
                to(field.viscosity.data[:field.viscosity.N()]) \
                to(field.soundspeed.data[:field.soundspeed.N()]) \
                to(field.yvel0.data[:field.yvel0.N()]) \
                to(field.yvel1.data[:field.yvel1.N()]) \
                to(field.xvel0.data[:field.xvel0.N()]) \
                to(field.xvel1.data[:field.xvel1.N()]) \
                to(field.vol_flux_x.data[:field.vol_flux_x.N()]) \
                to(field.vol_flux_y.data[:field.vol_flux_y.N()]) \
                to(field.mass_flux_x.data[:field.mass_flux_x.N()]) \
                to(field.mass_flux_y.data[:field.mass_flux_y.N()]) \
                to(field.work_array1.data[:field.work_array1.N()]) \
                to(field.work_array2.data[:field.work_array2.N()]) \
                to(field.work_array3.data[:field.work_array3.N()]) \
                to(field.work_array4.data[:field.work_array4.N()]) \
                to(field.work_array5.data[:field.work_array5.N()]) \
                to(field.work_array6.data[:field.work_array6.N()]) \
                to(field.work_array7.data[:field.work_array7.N()]) \
                to(field.cellx.data[:field.cellx.N()]) \
                to(field.celldx.data[:field.celldx.N()]) \
                to(field.celly.data[:field.celly.N()]) \
                to(field.celldy.data[:field.celldy.N()]) \
                to(field.vertexx.data[:field.vertexx.N()]) \
                to(field.vertexdx.data[:field.vertexdx.N()]) \
                to(field.vertexy.data[:field.vertexy.N()]) \
                to(field.vertexdy.data[:field.vertexdy.N()]) \
                to(field.volume.data[:field.volume.N()]) \
                to(field.xarea.data[:field.xarea.N()]) \
                to(field.yarea.data[:field.yarea.N()])
		}

	}

	void deviceToHost() {
		#pragma omp flush

		for (int tile = 0; tile < config.tiles_per_chunk; ++tile) {
			tile_type &t = chunk.tiles[tile];
			field_type &field = t.field;
			#pragma omp target update \
                from(field.density0.data[:field.density0.N()]) \
                from(field.density1.data[:field.density1.N()]) \
                from(field.energy0.data[:field.energy0.N()]) \
                from(field.energy1.data[:field.energy1.N()]) \
                from(field.pressure.data[:field.pressure.N()]) \
                from(field.viscosity.data[:field.viscosity.N()]) \
                from(field.soundspeed.data[:field.soundspeed.N()]) \
                from(field.yvel0.data[:field.yvel0.N()]) \
                from(field.yvel1.data[:field.yvel1.N()]) \
                from(field.xvel0.data[:field.xvel0.N()]) \
                from(field.xvel1.data[:field.xvel1.N()]) \
                from(field.vol_flux_x.data[:field.vol_flux_x.N()]) \
                from(field.vol_flux_y.data[:field.vol_flux_y.N()]) \
                from(field.mass_flux_x.data[:field.mass_flux_x.N()]) \
                from(field.mass_flux_y.data[:field.mass_flux_y.N()]) \
                from(field.work_array1.data[:field.work_array1.N()]) \
                from(field.work_array2.data[:field.work_array2.N()]) \
                from(field.work_array3.data[:field.work_array3.N()]) \
                from(field.work_array4.data[:field.work_array4.N()]) \
                from(field.work_array5.data[:field.work_array5.N()]) \
                from(field.work_array6.data[:field.work_array6.N()]) \
                from(field.work_array7.data[:field.work_array7.N()]) \
                from(field.cellx.data[:field.cellx.N()]) \
                from(field.celldx.data[:field.celldx.N()]) \
                from(field.celly.data[:field.celly.N()]) \
                from(field.celldy.data[:field.celldy.N()]) \
                from(field.vertexx.data[:field.vertexx.N()]) \
                from(field.vertexdx.data[:field.vertexdx.N()]) \
                from(field.vertexy.data[:field.vertexy.N()]) \
                from(field.vertexdy.data[:field.vertexdy.N()]) \
                from(field.volume.data[:field.volume.N()]) \
                from(field.xarea.data[:field.xarea.N()]) \
                from(field.yarea.data[:field.yarea.N()])
		}
	}

	// dumps all content to file; for debugging only
	void dump(const std::string &filename) {

		deviceToHost();

		std::cout << "Dumping globals to " << filename << std::endl;

		record(filename, [&](std::ostream &out) {
			out << "Dump(tileCount = " << chunk.tiles.size() << ")" << std::endl;


			out << "error_condition" << '=' << error_condition << std::endl;

			out << "step" << '=' << step << std::endl;
			out << "advect_x" << '=' << advect_x << std::endl;
			out << "time" << '=' << time << std::endl;

			out << "dt" << '=' << dt << std::endl;
			out << "dtold" << '=' << dtold << std::endl;

			out << "complete" << '=' << complete << std::endl;
			out << "jdt" << '=' << jdt << std::endl;
			out << "kdt" << '=' << kdt << std::endl;

			for (size_t i = 0; i < config.states.size(); ++i) {
				out << "\tStates[" << i << "]" << std::endl;
				auto &t = config.states[i];
				out << "\t\tdefined=" << t.defined << std::endl;
				out << "\t\tdensity=" << t.density << std::endl;
				out << "\t\tenergy=" << t.energy << std::endl;
				out << "\t\txvel=" << t.xvel << std::endl;
				out << "\t\tyvel=" << t.yvel << std::endl;
				out << "\t\tgeometry=" << t.geometry << std::endl;
				out << "\t\txmin=" << t.xmin << std::endl;
				out << "\t\tymin=" << t.ymin << std::endl;
				out << "\t\txmax=" << t.xmax << std::endl;
				out << "\t\tymax=" << t.ymax << std::endl;
				out << "\t\tradius=" << t.radius << std::endl;
			}


			for (size_t i = 0; i < chunk.tiles.size(); ++i) {
				auto fs = chunk.tiles[i].field;
				out << "\tTile[ " << i << "]:" << std::endl;

				tile_info &info = chunk.tiles[i].info;
				for (int l = 0; l < 4; ++l) {
					out << "info.tile_neighbours[i]" << '=' << info.tile_neighbours[i] << std::endl;
					out << "info.external_tile_mask[i]" << '=' << info.external_tile_mask[i]
					    << std::endl;
				}

				out << "info.t_xmin" << '=' << info.t_xmin << std::endl;
				out << "info.t_xmax" << '=' << info.t_xmax << std::endl;
				out << "info.t_ymin" << '=' << info.t_ymin << std::endl;
				out << "info.t_ymax" << '=' << info.t_ymax << std::endl;
				out << "info.t_left" << '=' << info.t_left << std::endl;
				out << "info.t_right" << '=' << info.t_right << std::endl;
				out << "info.t_bottom" << '=' << info.t_bottom << std::endl;
				out << "info.t_top" << '=' << info.t_top << std::endl;


				show(out, "density0", fs.density0);
				show(out, "density1", fs.density1);
				show(out, "energy0", fs.energy0);
				show(out, "energy1", fs.energy1);
				show(out, "pressure", fs.pressure);
				show(out, "viscosity", fs.viscosity);
				show(out, "soundspeed", fs.soundspeed);
				show(out, "xvel0", fs.xvel0);
				show(out, "xvel1", fs.xvel1);
				show(out, "yvel0", fs.yvel0);
				show(out, "yvel1", fs.yvel1);
				show(out, "vol_flux_x", fs.vol_flux_x);
				show(out, "vol_flux_y", fs.vol_flux_y);
				show(out, "mass_flux_x", fs.mass_flux_x);
				show(out, "mass_flux_y", fs.mass_flux_y);

				show(out, "work_array1",
				     fs.work_array1); // node_flux, stepbymass, volume_change, pre_vol
				show(out, "work_array2", fs.work_array2); // node_mass_post, post_vol
				show(out, "work_array3", fs.work_array3); // node_mass_pre,pre_mass
				show(out, "work_array4", fs.work_array4); // advec_vel, post_mass
				show(out, "work_array5", fs.work_array5); // mom_flux, advec_vol
				show(out, "work_array6", fs.work_array6); // pre_vol, post_ener
				show(out, "work_array7", fs.work_array7); // post_vol, ener_flux

				show(out, "cellx", fs.cellx);
				show(out, "celldx", fs.celldx);
				show(out, "celly", fs.celly);
				show(out, "celldy", fs.celldy);
				show(out, "vertexx", fs.vertexx);
				show(out, "vertexdx", fs.vertexdx);
				show(out, "vertexy", fs.vertexy);
				show(out, "vertexdy", fs.vertexdy);

				show(out, "volume", fs.volume);
				show(out, "xarea", fs.xarea);
				show(out, "yarea", fs.yarea);
			}
		});

	}


};


#endif


