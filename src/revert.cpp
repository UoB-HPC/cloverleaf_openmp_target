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


#include "revert.h"


//  @brief Fortran revert kernel.
//  @author Wayne Gaudin
//  @details Takes the half step field data used in the predictor and reverts
//  it to the start of step data, ready for the corrector.
//  Note that this does not seem necessary in this proxy-app but should be
//  left in to remain relevant to the full method.
void revert_kernel(
		bool use_target,
		int x_min, int x_max, int y_min, int y_max,
		field_type &field) {

	// DO k=y_min,y_max
	//   DO j=x_min,x_max
	mapToFrom2Df(field, density0)
	mapToFrom2Df(field, density1)
	mapToFrom2Df(field, energy0)
	mapToFrom2Df(field, energy1)

	omp(parallel(2) enable_target(use_target))
	for (int j = (y_min + 1); j < (y_max + 2); j++) {
		for (int i = (x_min + 1); i < (x_max + 2); i++) {
			idx2f(field, density1, i, j) = idx2f(field, density0, i, j);
			idx2f(field, energy1, i, j) = idx2f(field, energy0, i, j);
		}
	}

}


//  @brief Driver routine for the revert kernels.
//  @author Wayne Gaudin
//  @details Invokes the user specified revert kernel.
void revert(global_variables &globals) {

	#if FLUSH_BUFFER
	globals.hostToDevice();
	#endif

	for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
		tile_type &t = globals.chunk.tiles[tile];
		revert_kernel(
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


}

