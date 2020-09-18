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


// @brief Driver for chunk initialisation.
// @author Wayne Gaudin
// @details Invokes the user specified chunk initialisation kernel.
// @brief Fortran chunk initialisation kernel.
// @author Wayne Gaudin
// @details Calculates mesh geometry for the mesh chunk based on the mesh size.


#include "initialise_chunk.h"



void initialise_chunk(const int tile, global_variables &globals) {

	double dx = (globals.config.grid.xmax - globals.config.grid.xmin) /
	            (double) (globals.config.grid.x_cells);
	double dy = (globals.config.grid.ymax - globals.config.grid.ymin) /
	            (double) (globals.config.grid.y_cells);

	double xmin =
			globals.config.grid.xmin + dx * (double) (globals.chunk.tiles[tile].info.t_left - 1);

	double ymin =
			globals.config.grid.ymin + dy * (double) (globals.chunk.tiles[tile].info.t_bottom - 1);


	const int x_min = globals.chunk.tiles[tile].info.t_xmin;
	const int x_max = globals.chunk.tiles[tile].info.t_xmax;
	const int y_min = globals.chunk.tiles[tile].info.t_ymin;
	const int y_max = globals.chunk.tiles[tile].info.t_ymax;

	const int xrange = (x_max + 3) - (x_min - 2) + 1;
	const int yrange = (y_max + 3) - (y_min - 2) + 1;

	// Take a reference to the lowest structure, as Kokkos device cannot necessarily chase through the structure.
	field_type &field = globals.chunk.tiles[tile].field;




	omp(parallel(1) enable_target(globals.use_target)
			    mapToFrom1D(field.vertexx)
			    mapToFrom1D(field.vertexdx)
	)
	for (int j = (0); j < (xrange); j++) {
		idx1(field.vertexx, j) = xmin + dx * (j - 1 - x_min);
		idx1(field.vertexdx, j) = dx;
	}


	omp(parallel(1) enable_target(globals.use_target)
			    mapToFrom1D(field.vertexy)
			    mapToFrom1D(field.vertexdy)
	)
	for (int k = (0); k < (yrange); k++) {
		idx1(field.vertexy, k) = ymin + dy * (k - 1 - y_min);
		idx1(field.vertexdy, k) = dy;
	}


	const int xrange1 = (x_max + 2) - (x_min - 2) + 1;
	const int yrange1 = (y_max + 2) - (y_min - 2) + 1;

	omp(parallel(1) enable_target(globals.use_target)
			    mapToFrom1D(field.cellx)
			    mapToFrom1D(field.celldx)
			    mapToFrom1D(field.vertexx)
	)
	for (int j = (0); j < (xrange1); j++) {
		idx1(field.cellx, j) = 0.5 * (idx1(field.vertexx, j) + idx1(field.vertexx, j + 1));
		idx1(field.celldx, j) = dx;
	}


	omp(parallel(1) enable_target(globals.use_target)
			    mapToFrom1D(field.celly)
			    mapToFrom1D(field.celldy)
			    mapToFrom1D(field.vertexy)
	)
	for (int k = (0); k < (yrange1); k++) {
		idx1(field.celly, k) = 0.5 * (idx1(field.vertexy, k) + idx1(field.vertexy, k + 1));
		idx1(field.celldy, k) = dy;
	}


	omp(parallel(2) enable_target(globals.use_target)
			    mapToFrom2D(field.volume)
			    mapToFrom2D(field.xarea)
			    mapToFrom2D(field.yarea)
			    mapToFrom1D(field.celldx)
			    mapToFrom1D(field.celldy)
	)
	for (int j = (0); j < (yrange1); j++) {
		for (int i = (0); i < (xrange1); i++) {
			idx2(field.volume, i, j) = dx * dy;
			idx2(field.xarea, i, j) = idx1(field.celldy, j);
			idx2(field.yarea, i, j) = idx1(field.celldx, i);
		}
	}


}


