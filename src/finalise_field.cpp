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
                map(from: field.density0.data[:field.density0.N()]) \
                map(from: field.density1.data[:field.density1.N()]) \
                map(from: field.energy0.data[:field.energy0.N()]) \
                map(from: field.energy1.data[:field.energy1.N()]) \
                map(from: field.pressure.data[:field.pressure.N()]) \
                map(from: field.viscosity.data[:field.viscosity.N()]) \
                map(from: field.soundspeed.data[:field.soundspeed.N()]) \
                map(from: field.yvel0.data[:field.yvel0.N()]) \
                map(from: field.yvel1.data[:field.yvel1.N()]) \
                map(from: field.xvel0.data[:field.xvel0.N()]) \
                map(from: field.xvel1.data[:field.xvel1.N()]) \
                map(from: field.vol_flux_x.data[:field.vol_flux_x.N()]) \
                map(from: field.vol_flux_y.data[:field.vol_flux_y.N()]) \
                map(from: field.mass_flux_x.data[:field.mass_flux_x.N()]) \
                map(from: field.mass_flux_y.data[:field.mass_flux_y.N()]) \
                map(from: field.work_array1.data[:field.work_array1.N()]) \
                map(from: field.work_array2.data[:field.work_array2.N()]) \
                map(from: field.work_array3.data[:field.work_array3.N()]) \
                map(from: field.work_array4.data[:field.work_array4.N()]) \
                map(from: field.work_array5.data[:field.work_array5.N()]) \
                map(from: field.work_array6.data[:field.work_array6.N()]) \
                map(from: field.work_array7.data[:field.work_array7.N()]) \
                map(from: field.cellx.data[:field.cellx.N()]) \
                map(from: field.celldx.data[:field.celldx.N()]) \
                map(from: field.celly.data[:field.celly.N()]) \
                map(from: field.celldy.data[:field.celldy.N()]) \
                map(from: field.vertexx.data[:field.vertexx.N()]) \
                map(from: field.vertexdx.data[:field.vertexdx.N()]) \
                map(from: field.vertexy.data[:field.vertexy.N()]) \
                map(from: field.vertexdy.data[:field.vertexdy.N()]) \
                map(from: field.volume.data[:field.volume.N()]) \
                map(from: field.xarea.data[:field.xarea.N()]) \
                map(from: field.yarea.data[:field.yarea.N()])

	}

}

