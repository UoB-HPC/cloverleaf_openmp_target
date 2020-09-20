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
		field_type &field) {

	double halfdt = 0.5 * dt;

	// DO k=y_min,y_max+1
	//   DO j=x_min,x_max+1
//	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1},
//	                                               {x_max + 1 + 2, y_max + 1 + 2});

//for(int j = )


	mapToFrom2Df(field, xarea)
	mapToFrom2Df(field, yarea)
	mapToFrom2Df(field, volume)
	mapToFrom2Df(field, density0)
	mapToFrom2Df(field, pressure)
	mapToFrom2Df(field, viscosity)
	mapToFrom2Df(field, xvel0)
	mapToFrom2Df(field, yvel0)
	mapToFrom2Df(field, xvel1)
	mapToFrom2Df(field, yvel1)

	omp(parallel(2) enable_target(use_target))
	for (int j = (y_min + 1); j < (y_max + 1 + 2); j++) {
		for (int i = (x_min + 1); i < (x_max + 1 + 2); i++) {
			double stepbymass_s = halfdt / ((idx2f(field, density0, i - 1, j - 1) * idx2f(field, volume, i - 1, j - 1) +
			                                 idx2f(field, density0, i - 1, j + 0) * idx2f(field, volume, i - 1, j + 0) + idx2f(field, density0, i, j) * idx2f(field, volume, i, j) +
			                                 idx2f(field, density0, i + 0, j - 1) * idx2f(field, volume, i + 0, j - 1)) * 0.25);
			idx2f(field, xvel1, i, j) = idx2f(field, xvel0, i, j) -
			                            stepbymass_s * (idx2f(field, xarea, i, j) * (idx2f(field, pressure, i, j) - idx2f(field, pressure, i - 1, j + 0)) +
			                                            idx2f(field, xarea, i + 0, j - 1) * (idx2f(field, pressure, i + 0, j - 1) - idx2f(field, pressure, i - 1, j - 1)));
			idx2f(field, yvel1, i, j) = idx2f(field, yvel0, i, j) -
			                            stepbymass_s * (idx2f(field, yarea, i, j) *
			                                            (idx2f(field, pressure, i, j) - idx2f(field, pressure, i + 0, j - 1)) +
			                                            idx2f(field, yarea, i - 1, j + 0) * (idx2f(field, pressure, i - 1, j + 0) - idx2f(field, pressure, i - 1, j - 1)));
			idx2f(field, xvel1, i, j) = idx2f(field, xvel1, i, j) -
			                            stepbymass_s * (idx2f(field, xarea, i, j) *
			                                            (idx2f(field, viscosity, i, j) -
			                                             idx2f(field, viscosity, i - 1, j + 0)) +
			                                            idx2f(field, xarea, i + 0, j - 1) * (idx2f(field, viscosity, i + 0, j - 1) - idx2f(field, viscosity, i - 1, j - 1)));
			idx2f(field, yvel1, i, j) = idx2f(field, yvel1, i, j) -
			                            stepbymass_s * (idx2f(field, yarea, i, j) *
			                                            (idx2f(field, viscosity, i, j) - idx2f(field, viscosity, i + 0, j - 1)) +
			                                            idx2f(field, yarea, i - 1, j + 0) * (idx2f(field, viscosity, i - 1, j + 0) - idx2f(field, viscosity, i - 1, j - 1)));
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
				t.field);


	}

	#if FLUSH_BUFFER
	globals.deviceToHost();
	#endif

	if (globals.profiler_on) globals.profiler.acceleration += timer() - kernel_time;

}
