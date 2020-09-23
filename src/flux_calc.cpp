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



#include "flux_calc.h"
#include "timer.h"



//  @brief Fortran flux kernel.
//  @author Wayne Gaudin
//  @details The edge volume fluxes are calculated based on the velocity fields.
void flux_calc_kernel(
		bool use_target,
		int x_min, int x_max, int y_min, int y_max,
		double dt,
		field_type &field) {

	// DO k=y_min,y_max+1
	//   DO j=x_min,x_max+1
// Note that the loops calculate one extra flux than required, but this
	// allows loop fusion that improves performance
	double *xarea = field.xarea.data;
	const int xarea_sizex = field.xarea.sizeX;
	double *yarea = field.yarea.data;
	const int yarea_sizex = field.yarea.sizeX;
	double *xvel0 = field.xvel0.data;
	const int xvel0_sizex = field.xvel0.sizeX;
	double *yvel0 = field.yvel0.data;
	const int yvel0_sizex = field.yvel0.sizeX;
	double *xvel1 = field.xvel1.data;
	const int xvel1_sizex = field.xvel1.sizeX;
	double *yvel1 = field.yvel1.data;
	const int yvel1_sizex = field.yvel1.sizeX;
	double *vol_flux_x = field.vol_flux_x.data;
	const int vol_flux_x_sizex = field.vol_flux_x.sizeX;
	double *vol_flux_y = field.vol_flux_y.data;
	const int vol_flux_y_sizex = field.vol_flux_y.sizeX;

	#pragma omp target teams distribute parallel for simd collapse(2) if(target: use_target)
	for (int j = (y_min + 1); j < (y_max + 1 + 2); j++) {
		for (int i = (x_min + 1); i < (x_max + 1 + 2); i++) {
			vol_flux_x[i + j * vol_flux_x_sizex] = 0.25 * dt * xarea[i + j * xarea_sizex] *
			                                       (xvel0[i + j * xvel0_sizex] + xvel0[(i + 0) + (j + 1) * xvel0_sizex] + xvel1[i + j * xvel1_sizex] +
			                                        xvel1[(i + 0) + (j + 1) * xvel1_sizex]);
			vol_flux_y[i + j * vol_flux_y_sizex] = 0.25 * dt * yarea[i + j * yarea_sizex] *
			                                       (yvel0[i + j * yvel0_sizex] + yvel0[(i + 1) + (j + 0) * yvel0_sizex] + yvel1[i + j * yvel1_sizex] +
			                                        yvel1[(i + 1) + (j + 0) * yvel1_sizex]);
		}
	}
}

// @brief Driver for the flux kernels
// @author Wayne Gaudin
// @details Invokes the used specified flux kernel
void flux_calc(global_variables &globals) {

	double kernel_time = 0;
	if (globals.profiler_on) kernel_time = timer();


	#if FLUSH_BUFFER
	globals.hostToDevice();
	#endif

	for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

		tile_type &t = globals.chunk.tiles[tile];
		flux_calc_kernel(
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

	if (globals.profiler_on) globals.profiler.flux += timer() - kernel_time;

}

