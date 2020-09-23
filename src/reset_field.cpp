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
		field_type &field) {



	// DO k=y_min,y_max
	//   DO j=x_min,x_max
	double *density0 = field.density0.data;
	const int density0_sizex = field.density0.sizeX;
	double *density1 = field.density1.data;
	const int density1_sizex = field.density1.sizeX;
	double *energy0 = field.energy0.data;
	const int energy0_sizex = field.energy0.sizeX;
	double *energy1 = field.energy1.data;
	const int energy1_sizex = field.energy1.sizeX;

	#pragma omp target teams distribute parallel for simd collapse(2) if(target: use_target)
	for (int j = (y_min + 1); j < (y_max + 2); j++) {
		for (int i = (x_min + 1); i < (x_max + 2); i++) {
			density0[i + j * density0_sizex] = density1[i + j * density1_sizex];
			energy0[i + j * energy0_sizex] = energy1[i + j * energy1_sizex];
		}
	}




	// DO k=y_min,y_max+1
	//   DO j=x_min,x_max+1
	double *xvel0 = field.xvel0.data;
	const int xvel0_sizex = field.xvel0.sizeX;
	double *xvel1 = field.xvel1.data;
	const int xvel1_sizex = field.xvel1.sizeX;
	double *yvel0 = field.yvel0.data;
	const int yvel0_sizex = field.yvel0.sizeX;
	double *yvel1 = field.yvel1.data;
	const int yvel1_sizex = field.yvel1.sizeX;

	#pragma omp target teams distribute parallel for simd collapse(2) if(target: use_target)
	for (int j = (y_min + 1); j < (y_max + 1 + 2); j++) {
		for (int i = (x_min + 1); i < (x_max + 1 + 2); i++) {
			xvel0[i + j * xvel0_sizex] = xvel1[i + j * xvel1_sizex];
			yvel0[i + j * yvel0_sizex] = yvel1[i + j * yvel1_sizex];
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
				t.field);
	}

	#if FLUSH_BUFFER
	globals.deviceToHost();
	#endif


	if (globals.profiler_on) globals.profiler.reset += timer() - kernel_time;
}

