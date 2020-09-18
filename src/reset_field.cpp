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


#include "reset_field.h"
#include "timer.h"


//  @brief Fortran reset field kernel.
//  @author Wayne Gaudin
//  @details Copies all of the final end of step filed data to the begining of
//  step data, ready for the next timestep.
void reset_field_kernel(
		bool use_target,
		int x_min, int x_max, int y_min, int y_max,
		clover::Buffer2D<double> &density0,
		clover::Buffer2D<double> &density1,
		clover::Buffer2D<double> &energy0,
		clover::Buffer2D<double> &energy1,
		clover::Buffer2D<double> &xvel0,
		clover::Buffer2D<double> &xvel1,
		clover::Buffer2D<double> &yvel0,
		clover::Buffer2D<double> &yvel1) {



	// DO k=y_min,y_max
	//   DO j=x_min,x_max
	omp(parallel(2) enable_target(use_target)
			    mapToFrom2D(density0)
			    mapToFrom2D(density1)
			    mapToFrom2D(energy0)
			    mapToFrom2D(energy1)
	)
	for (int j = (y_min + 1); j < (y_max + 2); j++) {
		for (int i = (x_min + 1); i < (x_max + 2); i++) {
			idx2(density0, i, j) = idx2(density1, i, j);
			idx2(energy0, i, j) = idx2(energy1, i, j);
		}
	}




	// DO k=y_min,y_max+1
	//   DO j=x_min,x_max+1
	omp(parallel(2) enable_target(use_target)
			    mapToFrom2D(xvel0)
			    mapToFrom2D(xvel1)
			    mapToFrom2D(yvel0)
			    mapToFrom2D(yvel1)
	)
	for (int j = (y_min + 1); j < (y_max + 1 + 2); j++) {
		for (int i = (x_min + 1); i < (x_max + 1 + 2); i++) {
			idx2(xvel0, i, j) = idx2(xvel1, i, j);
			idx2(yvel0, i, j) = idx2(yvel1, i, j);
		}
	}

}


//  @brief Reset field driver
//  @author Wayne Gaudin
//  @details Invokes the user specified field reset kernel.
void reset_field(global_variables &globals) {

	double kernel_time = 0;
	if (globals.profiler_on) kernel_time = timer();

	#if FLUSH_BUFFER
	globals.hostToDevice();
	#endif

	for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

		tile_type &t = globals.chunk.tiles[tile];
		reset_field_kernel(
				globals.use_target,
				t.info.t_xmin,
				t.info.t_xmax,
				t.info.t_ymin,
				t.info.t_ymax,

				t.field.density0,
				t.field.density1,
				t.field.energy0,
				t.field.energy1,
				t.field.xvel0,
				t.field.xvel1,
				t.field.yvel0,
				t.field.yvel1);
	}

	#if FLUSH_BUFFER
	globals.deviceToHost();
	#endif


	if (globals.profiler_on) globals.profiler.reset += timer() - kernel_time;
}

