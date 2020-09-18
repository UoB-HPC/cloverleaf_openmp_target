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


#include "build_field.h"


// Allocate Kokkos Views for the data arrays
void build_field(global_variables &globals) {

	for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

		tile_type &t = globals.chunk.tiles[tile];
		field_type &field = t.field;





//
		#pragma omp target enter data \
                map(alloc: field.density0.data[:field.density0.N()])  map(to: field.density0.sizeX) \
                map(alloc: field.density1.data[:field.density1.N()])  map(to: field.density1.sizeX)  \
                map(alloc: field.energy0.data[:field.energy0.N()])  map(to: field.energy0.sizeX)  \
                map(alloc: field.energy1.data[:field.energy1.N()])  map(to: field.energy1.sizeX)  \
                map(alloc: field.pressure.data[:field.pressure.N()])  map(to: field.pressure.sizeX)  \
                map(alloc: field.viscosity.data[:field.viscosity.N()])  map(to: field.viscosity.sizeX)  \
                map(alloc: field.soundspeed.data[:field.soundspeed.N()])  map(to: field.soundspeed.sizeX)  \
                map(alloc: field.yvel0.data[:field.yvel0.N()])  map(to: field.yvel0.sizeX)  \
                map(alloc: field.yvel1.data[:field.yvel1.N()])  map(to: field.yvel1.sizeX)  \
                map(alloc: field.xvel0.data[:field.xvel0.N()])  map(to: field.xvel0.sizeX)  \
                map(alloc: field.xvel1.data[:field.xvel1.N()])  map(to: field.xvel1.sizeX)  \
                map(alloc: field.vol_flux_x.data[:field.vol_flux_x.N()])  map(to: field.vol_flux_x.sizeX)  \
                map(alloc: field.vol_flux_y.data[:field.vol_flux_y.N()])  map(to: field.vol_flux_y.sizeX)  \
                map(alloc: field.mass_flux_x.data[:field.mass_flux_x.N()])  map(to: field.mass_flux_x.sizeX)  \
                map(alloc: field.mass_flux_y.data[:field.mass_flux_y.N()])  map(to: field.mass_flux_y.sizeX)  \
                map(alloc: field.work_array1.data[:field.work_array1.N()])  map(to: field.work_array1.sizeX)  \
                map(alloc: field.work_array2.data[:field.work_array2.N()])  map(to: field.work_array2.sizeX)  \
                map(alloc: field.work_array3.data[:field.work_array3.N()])  map(to: field.work_array3.sizeX)  \
                map(alloc: field.work_array4.data[:field.work_array4.N()])  map(to: field.work_array4.sizeX)  \
                map(alloc: field.work_array5.data[:field.work_array5.N()])  map(to: field.work_array5.sizeX)  \
                map(alloc: field.work_array6.data[:field.work_array6.N()])  map(to: field.work_array6.sizeX)  \
                map(alloc: field.work_array7.data[:field.work_array7.N()])  map(to: field.work_array7.sizeX)  \
                map(alloc: field.cellx.data[:field.cellx.N()]) \
                map(alloc: field.celldx.data[:field.celldx.N()]) \
                map(alloc: field.celly.data[:field.celly.N()]) \
                map(alloc: field.celldy.data[:field.celldy.N()]) \
                map(alloc: field.vertexx.data[:field.vertexx.N()]) \
                map(alloc: field.vertexdx.data[:field.vertexdx.N()]) \
                map(alloc: field.vertexy.data[:field.vertexy.N()]) \
                map(alloc: field.vertexdy.data[:field.vertexdy.N()]) \
                map(alloc: field.volume.data[:field.volume.N()])  map(to: field.volume.sizeX)  \
                map(alloc: field.xarea.data[:field.xarea.N()])  map(to: field.xarea.sizeX)  \
                map(alloc: field.yarea.data[:field.yarea.N()])  map(to: field.yarea.sizeX)  \

		const int xrange = (t.info.t_xmax + 2) - (t.info.t_xmin - 2) + 1;
		const int yrange = (t.info.t_ymax + 2) - (t.info.t_ymin - 2) + 1;

		// (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+2)

//		t.field.density0 = Buffer2D<double>(range<2>(xrange, yrange));
//		t.field.density1 = Buffer2D<double>(range<2>(xrange, yrange));
//		t.field.energy0 = Buffer2D<double>(range<2>(xrange, yrange));
//		t.field.energy1 = Buffer2D<double>(range<2>(xrange, yrange));
//		t.field.pressure = Buffer2D<double>(range<2>(xrange, yrange));
//		t.field.viscosity = Buffer2D<double>(range<2>(xrange, yrange));
//		t.field.soundspeed = Buffer2D<double>(range<2>(xrange, yrange));
//
//		// (t_xmin-2:t_xmax+3, t_ymin-2:t_ymax+3)
//		t.field.xvel0 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
//		t.field.xvel1 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
//		t.field.yvel0 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
//		t.field.yvel1 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
//
//		// (t_xmin-2:t_xmax+3, t_ymin-2:t_ymax+2)
//		t.field.vol_flux_x = Buffer2D<double>(range<2>(xrange + 1, yrange));
//		t.field.mass_flux_x = Buffer2D<double>(range<2>(xrange + 1, yrange));
//		// (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+3)
//		t.field.vol_flux_y = Buffer2D<double>(range<2>(xrange, yrange + 1));
//		t.field.mass_flux_y = Buffer2D<double>(range<2>(xrange, yrange + 1));
//
//		// (t_xmin-2:t_xmax+3, t_ymin-2:t_ymax+3)
//		t.field.work_array1 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
//		t.field.work_array2 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
//		t.field.work_array3 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
//		t.field.work_array4 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
//		t.field.work_array5 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
//		t.field.work_array6 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
//		t.field.work_array7 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
//
//		// (t_xmin-2:t_xmax+2)
//		t.field.cellx = Buffer1D<double>(range<1>(xrange));
//		t.field.celldx = Buffer1D<double>(range<1>(xrange));
//		// (t_ymin-2:t_ymax+2)
//		t.field.celly = Buffer1D<double>(range<1>(yrange));
//		t.field.celldy = Buffer1D<double>(range<1>(yrange));
//		// (t_xmin-2:t_xmax+3)
//		t.field.vertexx = Buffer1D<double>(range<1>(xrange + 1));
//		t.field.vertexdx = Buffer1D<double>(range<1>(xrange + 1));
//		// (t_ymin-2:t_ymax+3)
//		t.field.vertexy = Buffer1D<double>(range<1>(yrange + 1));
//		t.field.vertexdy = Buffer1D<double>(range<1>(yrange + 1));
//
//		// (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+2)
//		t.field.volume = Buffer2D<double>(range<2>(xrange, yrange));
//		// (t_xmin-2:t_xmax+3, t_ymin-2:t_ymax+2)
//		t.field.xarea = Buffer2D<double>(range<2>(xrange + 1, yrange));
//		// (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+3)
//		t.field.yarea = Buffer2D<double>(range<2>(xrange, yrange + 1));

		// Zeroing isn't strictly necessary but it ensures physical pages
		// are allocated. This prevents first touch overheads in the main code
		// cycle which can skew timings in the first step

		// Take a reference to the lowest structure, as Kokkos device cannot necessarily chase through the structure.

//		Kokkos::MDRangePolicy <Kokkos::Rank<2>> loop_bounds_1({0, 0}, {xrange + 1, yrange + 1});


//		#pragma omp target enter data  map(alloc: field.work_array1.ptr[:field.work_array1.size()])


		// Nested loop over (t_ymin-2:t_ymax+3) and (t_xmin-2:t_xmax+3) inclusive




		omp(parallel(2) enable_target(globals.use_target)
				    mapToFrom2D(field.work_array1)
				    mapToFrom2D(field.work_array2)
				    mapToFrom2D(field.work_array3)
				    mapToFrom2D(field.work_array4)
				    mapToFrom2D(field.work_array5)
				    mapToFrom2D(field.work_array6)
				    mapToFrom2D(field.work_array7)
				    mapToFrom2D(field.xvel0)
				    mapToFrom2D(field.xvel1)
				    mapToFrom2D(field.yvel0)
				    mapToFrom2D(field.yvel1)
		)
		for (int j = (0); j < (yrange + 1); j++) {
			for (int i = (0); i < (xrange + 1); i++) {
				idx2(field.work_array1, i, j) = 0.0;
				idx2(field.work_array2, i, j) = 0.0;
				idx2(field.work_array3, i, j) = 0.0;
				idx2(field.work_array4, i, j) = 0.0;
				idx2(field.work_array5, i, j) = 0.0;
				idx2(field.work_array6, i, j) = 0.0;
				idx2(field.work_array7, i, j) = 0.0;
				idx2(field.xvel0, i, j) = 0.0;
				idx2(field.xvel1, i, j) = 0.0;
				idx2(field.yvel0, i, j) = 0.0;
				idx2(field.yvel1, i, j) = 0.0;
			}
		}

		// Nested loop over (t_ymin-2:t_ymax+2) and (t_xmin-2:t_xmax+2) inclusive
		omp(parallel(2) enable_target(globals.use_target)
				    mapToFrom2D(field.density0)
				    mapToFrom2D(field.density1)
				    mapToFrom2D(field.energy0)
				    mapToFrom2D(field.energy1)
				    mapToFrom2D(field.pressure)
				    mapToFrom2D(field.viscosity)
				    mapToFrom2D(field.soundspeed)
				    mapToFrom2D(field.volume)
		)
		for (int j = (0); j < (yrange); j++) {
			for (int i = (0); i < (xrange); i++) {
				idx2(field.density0, i, j) = 0.0;
				idx2(field.density1, i, j) = 0.0;
				idx2(field.energy0, i, j) = 0.0;
				idx2(field.energy1, i, j) = 0.0;
				idx2(field.pressure, i, j) = 0.0;
				idx2(field.viscosity, i, j) = 0.0;
				idx2(field.soundspeed, i, j) = 0.0;
				idx2(field.volume, i, j) = 0.0;
			}
		}

		// Nested loop over (t_ymin-2:t_ymax+2) and (t_xmin-2:t_xmax+3) inclusive
		omp(parallel(2) enable_target(globals.use_target)
				    mapToFrom2D(field.vol_flux_x)
				    mapToFrom2D(field.mass_flux_x)
				    mapToFrom2D(field.xarea)
		)
		for (int j = (0); j < (yrange); j++) {
			for (int i = (0); i < (xrange); i++) {
				idx2(field.vol_flux_x, i, j) = 0.0;
				idx2(field.mass_flux_x, i, j) = 0.0;
				idx2(field.xarea, i, j) = 0.0;
			}
		}

		// Nested loop over (t_ymin-2:t_ymax+3) and (t_xmin-2:t_xmax+2) inclusive
		omp(parallel(2) enable_target(globals.use_target)
				    mapToFrom2D(field.vol_flux_y)
				    mapToFrom2D(field.mass_flux_y)
				    mapToFrom2D(field.yarea)
		)
		for (int j = (0); j < (yrange + 1); j++) {
			for (int i = (0); i < (xrange); i++) {
				idx2(field.vol_flux_y, i, j) = 0.0;
				idx2(field.mass_flux_y, i, j) = 0.0;
				idx2(field.yarea, i, j) = 0.0;
			}
		}

		// (t_xmin-2:t_xmax+2) inclusive
		omp(parallel(1) enable_target(globals.use_target)
				    mapToFrom1D(field.cellx)
				    mapToFrom1D(field.celldx)
		)
		for (int id = (0); id < (xrange); id++) {
			idx1(field.cellx, id) = 0.0;
			idx1(field.celldx, id) = 0.0;
		}

		// (t_ymin-2:t_ymax+2) inclusive
		omp(parallel(1) enable_target(globals.use_target)
				    mapToFrom1D(field.celly)
				    mapToFrom1D(field.celldy)
		)
		for (int id = (0); id < (yrange); id++) {
			idx1(field.celly, id) = 0.0;
			idx1(field.celldy, id) = 0.0;
		}

		// (t_xmin-2:t_xmax+3) inclusive
		omp(parallel(1) enable_target(globals.use_target)
				    mapToFrom1D(field.vertexx)
				    mapToFrom1D(field.vertexdx)
		)
		for (int id = (0); id < (xrange + 1); id++) {
			idx1(field.vertexx, id) = 0.0;
			idx1(field.vertexdx, id) = 0.0;
		}

		// (t_ymin-2:t_ymax+3) inclusive
		omp(parallel(1) enable_target(globals.use_target)
				    mapToFrom1D(field.vertexy)
				    mapToFrom1D(field.vertexdy)
		)
		for (int id = (0); id < (yrange + 1); id++) {
			idx1(field.vertexy, id) = 0.0;
			idx1(field.vertexdy, id) = 0.0;
		}


	}

}

