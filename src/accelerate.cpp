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



#include "accelerate.h"
#include "timer.h"



// @brief Fortran acceleration kernel
// @author Wayne Gaudin
// @details The pressure and viscosity gradients are used to update the 
// velocity field.
void accelerate_kernel(
		bool use_target,
		int x_min, int x_max, int y_min, int y_max,
		double dt,
		clover::Buffer2D<double> &xarea,
		clover::Buffer2D<double> &yarea,
		clover::Buffer2D<double> &volume,
		clover::Buffer2D<double> &density0,
		clover::Buffer2D<double> &pressure,
		clover::Buffer2D<double> &viscosity,
		clover::Buffer2D<double> &xvel0,
		clover::Buffer2D<double> &yvel0,
		clover::Buffer2D<double> &xvel1,
		clover::Buffer2D<double> &yvel1) {

	double halfdt = 0.5 * dt;

	// DO k=y_min,y_max+1
	//   DO j=x_min,x_max+1
//	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1},
//	                                               {x_max + 1 + 2, y_max + 1 + 2});

//for(int j = )


	omp(parallel(2) enable_target(use_target)
			    mapToFrom2D(xarea)
			    mapToFrom2D(yarea)
			    mapToFrom2D(volume)
			    mapToFrom2D(density0)
			    mapToFrom2D(pressure)
			    mapToFrom2D(viscosity)
			    mapToFrom2D(xvel0)
			    mapToFrom2D(yvel0)
			    mapToFrom2D(xvel1)
			    mapToFrom2D(yvel1)
	)
	for (int j = (y_min + 1); j < (y_max + 1 + 2); j++) {
		for (int i = (x_min + 1); i < (x_max + 1 + 2); i++) {
			double stepbymass_s = halfdt / ((idx2(density0, i - 1, j - 1) * idx2(volume, i - 1, j - 1) +
			                                 idx2(density0, i - 1, j + 0) * idx2(volume, i - 1, j + 0) + idx2(density0, i, j) * idx2(volume, i, j) +
			                                 idx2(density0, i + 0, j - 1) * idx2(volume, i + 0, j - 1)) * 0.25);
			idx2(xvel1, i, j) = idx2(xvel0, i, j) -
			                    stepbymass_s * (idx2(xarea, i, j) * (idx2(pressure, i, j) - idx2(pressure, i - 1, j + 0)) +
			                                    idx2(xarea, i + 0, j - 1) * (idx2(pressure, i + 0, j - 1) - idx2(pressure, i - 1, j - 1)));
			idx2(yvel1, i, j) = idx2(yvel0, i, j) -
			                    stepbymass_s * (idx2(yarea, i, j) *
			                                    (idx2(pressure, i, j) - idx2(pressure, i + 0, j - 1)) +
			                                    idx2(yarea, i - 1, j + 0) * (idx2(pressure, i - 1, j + 0) - idx2(pressure, i - 1, j - 1)));
			idx2(xvel1, i, j) = idx2(xvel1, i, j) -
			                    stepbymass_s * (idx2(xarea, i, j) *
			                                    (idx2(viscosity, i, j) -
			                                     idx2(viscosity, i - 1, j + 0)) +
			                                    idx2(xarea, i + 0, j - 1) * (idx2(viscosity, i + 0, j - 1) - idx2(viscosity, i - 1, j - 1)));
			idx2(yvel1, i, j) = idx2(yvel1, i, j) -
			                    stepbymass_s * (idx2(yarea, i, j) *
			                                    (idx2(viscosity, i, j) - idx2(viscosity, i + 0, j - 1)) +
			                                    idx2(yarea, i - 1, j + 0) * (idx2(viscosity, i - 1, j + 0) - idx2(viscosity, i - 1, j - 1)));
		}
	}
}


//  @brief Driver for the acceleration kernels
//  @author Wayne Gaudin
//  @details Calls user requested kernel
void accelerate(global_variables &globals) {

	double kernel_time = 0;
	if (globals.profiler_on) kernel_time = timer();

	#if FLUSH_BUFFER
	globals.hostToDevice();
	#endif

	for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
		tile_type &t = globals.chunk.tiles[tile];

		accelerate_kernel(
				globals.use_target,
				t.info.t_xmin,
				t.info.t_xmax,
				t.info.t_ymin,
				t.info.t_ymax,
				globals.dt,
				t.field.xarea,
				t.field.yarea,
				t.field.volume,
				t.field.density0,
				t.field.pressure,
				t.field.viscosity,
				t.field.xvel0,
				t.field.yvel0,
				t.field.xvel1,
				t.field.yvel1);


	}

	#if FLUSH_BUFFER
	globals.deviceToHost();
	#endif

	if (globals.profiler_on) globals.profiler.acceleration += timer() - kernel_time;

}
