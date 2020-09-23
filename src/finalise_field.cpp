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


// @brief  Allocates the data for each mesh chunk
// @author Wayne Gaudin
// @details The data fields for the mesh chunk are allocated based on the mesh
// size.


#include "finalise_field.h"


// Allocate Kokkos Views for the data arrays
void finalise_field(global_variables &globals) {

	for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

		tile_type &t = globals.chunk.tiles[tile];
		field_type &field = t.field;

		#pragma omp target exit data \
                map(release: field.density0.data[:0]) \
                map(release: field.density1.data[:0]) \
                map(release: field.energy0.data[:0]) \
                map(release: field.energy1.data[:0]) \
                map(release: field.pressure.data[:0]) \
                map(release: field.viscosity.data[:0]) \
                map(release: field.soundspeed.data[:0]) \
                map(release: field.yvel0.data[:0]) \
                map(release: field.yvel1.data[:0]) \
                map(release: field.xvel0.data[:0]) \
                map(release: field.xvel1.data[:0]) \
                map(release: field.vol_flux_x.data[:0]) \
                map(release: field.vol_flux_y.data[:0]) \
                map(release: field.mass_flux_x.data[:0]) \
                map(release: field.mass_flux_y.data[:0]) \
                map(release: field.work_array1.data[:0]) \
                map(release: field.work_array2.data[:0]) \
                map(release: field.work_array3.data[:0]) \
                map(release: field.work_array4.data[:0]) \
                map(release: field.work_array5.data[:0]) \
                map(release: field.work_array6.data[:0]) \
                map(release: field.work_array7.data[:0]) \
                map(release: field.cellx.data[:0]) \
                map(release: field.celldx.data[:0]) \
                map(release: field.celly.data[:0]) \
                map(release: field.celldy.data[:0]) \
                map(release: field.vertexx.data[:0]) \
                map(release: field.vertexdx.data[:0]) \
                map(release: field.vertexy.data[:0]) \
                map(release: field.vertexdy.data[:0]) \
                map(release: field.volume.data[:0]) \
                map(release: field.xarea.data[:0]) \
                map(release: field.yarea.data[:0])

	}

}

