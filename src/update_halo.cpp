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



#include "comms.h"
#include "update_halo.h"
#include "update_tile_halo.h"
#include "timer.h"



//   @brief Fortran kernel to update the external halo cells in a chunk.
//   @author Wayne Gaudin
//   @details Updates halo cells for the required fields at the required depth
//   for any halo cells that lie on an external boundary. The location and type
//   of data governs how this is carried out. External boundaries are always
//   reflective.
void update_halo_kernel(
		bool use_target,
		int x_min, int x_max, int y_min, int y_max,
		const std::array<int, 4> &chunk_neighbours,
		const std::array<int, 4> &tile_neighbours,
		field_type &field,
		const int fields[NUM_FIELDS],
		int depth) {


	//  Update values in external halo cells based on depth and fields requested
	//  Even though half of these loops look the wrong way around, it should be noted
	//  that depth is either 1 or 2 so that it is more efficient to always thread
	//  loop along the mesh edge.
	if (fields[field_density0] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *density0 = field.density0.data;
			const int density0_sizex = field.density0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					density0[j + (1 - k) * density0_sizex] = density0[j + (2 + k) * density0_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *density0 = field.density0.data;
			const int density0_sizex = field.density0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					density0[j + (y_max + 2 + k) * density0_sizex] = density0[j + (y_max + 1 - k) * density0_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *density0 = field.density0.data;
			const int density0_sizex = field.density0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					density0[(1 - j) + (k) * density0_sizex] = density0[(2 + j) + (k) * density0_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *density0 = field.density0.data;
			const int density0_sizex = field.density0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					density0[(x_max + 2 + j) + (k) * density0_sizex] = density0[(x_max + 1 - j) + (k) * density0_sizex];
				}
			}

		}
	}


	if (fields[field_density1] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *density1 = field.density1.data;
			const int density1_sizex = field.density1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					density1[j + (1 - k) * density1_sizex] = density1[j + (2 + k) * density1_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *density1 = field.density1.data;
			const int density1_sizex = field.density1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					density1[j + (y_max + 2 + k) * density1_sizex] = density1[j + (y_max + 1 - k) * density1_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *density1 = field.density1.data;
			const int density1_sizex = field.density1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					density1[(1 - j) + (k) * density1_sizex] = density1[(2 + j) + (k) * density1_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *density1 = field.density1.data;
			const int density1_sizex = field.density1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					density1[(x_max + 2 + j) + (k) * density1_sizex] = density1[(x_max + 1 - j) + (k) * density1_sizex];
				}
			}

		}
	}

	if (fields[field_energy0] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			//  DO j=x_min-depth,x_max+depth

			double *energy0 = field.energy0.data;
			const int energy0_sizex = field.energy0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					energy0[j + (1 - k) * energy0_sizex] = energy0[j + (2 + k) * energy0_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth

			double *energy0 = field.energy0.data;
			const int energy0_sizex = field.energy0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					energy0[j + (y_max + 2 + k) * energy0_sizex] = energy0[j + (y_max + 1 - k) * energy0_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth

			double *energy0 = field.energy0.data;
			const int energy0_sizex = field.energy0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					energy0[(1 - j) + (k) * energy0_sizex] = energy0[(2 + j) + (k) * energy0_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth

			double *energy0 = field.energy0.data;
			const int energy0_sizex = field.energy0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					energy0[(x_max + 2 + j) + (k) * energy0_sizex] = energy0[(x_max + 1 - j) + (k) * energy0_sizex];
				}
			}

		}
	}


	if (fields[field_energy1] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *energy1 = field.energy1.data;
			const int energy1_sizex = field.energy1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					energy1[j + (1 - k) * energy1_sizex] = energy1[j + (2 + k) * energy1_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *energy1 = field.energy1.data;
			const int energy1_sizex = field.energy1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					energy1[j + (y_max + 2 + k) * energy1_sizex] = energy1[j + (y_max + 1 - k) * energy1_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *energy1 = field.energy1.data;
			const int energy1_sizex = field.energy1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					energy1[(1 - j) + (k) * energy1_sizex] = energy1[(2 + j) + (k) * energy1_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *energy1 = field.energy1.data;
			const int energy1_sizex = field.energy1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					energy1[(x_max + 2 + j) + (k) * energy1_sizex] = energy1[(x_max + 1 - j) + (k) * energy1_sizex];
				}
			}

		}
	}

	if (fields[field_pressure] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *pressure = field.pressure.data;
			const int pressure_sizex = field.pressure.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					pressure[j + (1 - k) * pressure_sizex] = pressure[j + (2 + k) * pressure_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *pressure = field.pressure.data;
			const int pressure_sizex = field.pressure.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					pressure[j + (y_max + 2 + k) * pressure_sizex] = pressure[j + (y_max + 1 - k) * pressure_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *pressure = field.pressure.data;
			const int pressure_sizex = field.pressure.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					pressure[(1 - j) + (k) * pressure_sizex] = pressure[(2 + j) + (k) * pressure_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *pressure = field.pressure.data;
			const int pressure_sizex = field.pressure.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					pressure[(x_max + 2 + j) + (k) * pressure_sizex] = pressure[(x_max + 1 - j) + (k) * pressure_sizex];
				}
			}

		}
	}

	if (fields[field_viscosity] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *viscosity = field.viscosity.data;
			const int viscosity_sizex = field.viscosity.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					viscosity[j + (1 - k) * viscosity_sizex] = viscosity[j + (2 + k) * viscosity_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *viscosity = field.viscosity.data;
			const int viscosity_sizex = field.viscosity.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					viscosity[j + (y_max + 2 + k) * viscosity_sizex] = viscosity[j + (y_max + 1 - k) * viscosity_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *viscosity = field.viscosity.data;
			const int viscosity_sizex = field.viscosity.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					viscosity[(1 - j) + (k) * viscosity_sizex] = viscosity[(2 + j) + (k) * viscosity_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *viscosity = field.viscosity.data;
			const int viscosity_sizex = field.viscosity.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					viscosity[(x_max + 2 + j) + (k) * viscosity_sizex] = viscosity[(x_max + 1 - j) + (k) * viscosity_sizex];
				}
			}

		}
	}

	if (fields[field_soundspeed] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *soundspeed = field.soundspeed.data;
			const int soundspeed_sizex = field.soundspeed.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					soundspeed[j + (1 - k) * soundspeed_sizex] = soundspeed[j + (+k) * soundspeed_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *soundspeed = field.soundspeed.data;
			const int soundspeed_sizex = field.soundspeed.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					soundspeed[j + (y_max + 2 + k) * soundspeed_sizex] = soundspeed[j + (y_max + 1 - k) * soundspeed_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			//  DO k=y_min-depth,y_max+depth


			double *soundspeed = field.soundspeed.data;
			const int soundspeed_sizex = field.soundspeed.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					soundspeed[(1 - j) + (k) * soundspeed_sizex] = soundspeed[(2 + j) + (k) * soundspeed_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			//  DO k=y_min-depth,y_max+depth


			double *soundspeed = field.soundspeed.data;
			const int soundspeed_sizex = field.soundspeed.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					soundspeed[(x_max + 2 + j) + (k) * soundspeed_sizex] = soundspeed[(x_max + 1 - j) + (k) * soundspeed_sizex];
				}
			}

		}
	}


	if (fields[field_xvel0] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth


			double *xvel0 = field.xvel0.data;
			const int xvel0_sizex = field.xvel0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					xvel0[j + (1 - k) * xvel0_sizex] = xvel0[j + (1 + 2 + k) * xvel0_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth


			double *xvel0 = field.xvel0.data;
			const int xvel0_sizex = field.xvel0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					xvel0[j + (y_max + 1 + 2 + k) * xvel0_sizex] = xvel0[j + (y_max + 1 - k) * xvel0_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth


			double *xvel0 = field.xvel0.data;
			const int xvel0_sizex = field.xvel0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					xvel0[(1 - j) + (k) * xvel0_sizex] = -xvel0[(1 + 2 + j) + (k) * xvel0_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth


			double *xvel0 = field.xvel0.data;
			const int xvel0_sizex = field.xvel0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					xvel0[(x_max + 2 + 1 + j) + (k) * xvel0_sizex] = -xvel0[(x_max + 1 - j) + (k) * xvel0_sizex];
				}
			}

		}
	}

	if (fields[field_xvel1] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth


			double *xvel1 = field.xvel1.data;
			const int xvel1_sizex = field.xvel1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					xvel1[j + (1 - k) * xvel1_sizex] = xvel1[j + (1 + 2 + k) * xvel1_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth


			double *xvel1 = field.xvel1.data;
			const int xvel1_sizex = field.xvel1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					xvel1[j + (y_max + 1 + 2 + k) * xvel1_sizex] = xvel1[j + (y_max + 1 - k) * xvel1_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth


			double *xvel1 = field.xvel1.data;
			const int xvel1_sizex = field.xvel1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					xvel1[(1 - j) + (k) * xvel1_sizex] = -xvel1[(1 + 2 + j) + (k) * xvel1_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth


			double *xvel1 = field.xvel1.data;
			const int xvel1_sizex = field.xvel1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					xvel1[(x_max + 2 + 1 + j) + (k) * xvel1_sizex] = -xvel1[(x_max + 1 - j) + (k) * xvel1_sizex];
				}
			}

		}
	}

	if (fields[field_yvel0] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth


			double *yvel0 = field.yvel0.data;
			const int yvel0_sizex = field.yvel0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					yvel0[j + (1 - k) * yvel0_sizex] = -yvel0[j + (1 + 2 + k) * yvel0_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth


			double *yvel0 = field.yvel0.data;
			const int yvel0_sizex = field.yvel0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					yvel0[j + (y_max + 1 + 2 + k) * yvel0_sizex] = -yvel0[j + (y_max + 1 - k) * yvel0_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth


			double *yvel0 = field.yvel0.data;
			const int yvel0_sizex = field.yvel0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					yvel0[(1 - j) + (k) * yvel0_sizex] = yvel0[(1 + 2 + j) + (k) * yvel0_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth


			double *yvel0 = field.yvel0.data;
			const int yvel0_sizex = field.yvel0.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					yvel0[(x_max + 2 + 1 + j) + (k) * yvel0_sizex] = yvel0[(x_max + 1 - j) + (k) * yvel0_sizex];
				}
			}

		}
	}

	if (fields[field_yvel1] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth


			double *yvel1 = field.yvel1.data;
			const int yvel1_sizex = field.yvel1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					yvel1[j + (1 - k) * yvel1_sizex] = -yvel1[j + (1 + 2 + k) * yvel1_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth


			double *yvel1 = field.yvel1.data;
			const int yvel1_sizex = field.yvel1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					yvel1[j + (y_max + 1 + 2 + k) * yvel1_sizex] = -yvel1[j + (y_max + 1 - k) * yvel1_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth


			double *yvel1 = field.yvel1.data;
			const int yvel1_sizex = field.yvel1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					yvel1[(1 - j) + (k) * yvel1_sizex] = yvel1[(1 + 2 + j) + (k) * yvel1_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth


			double *yvel1 = field.yvel1.data;
			const int yvel1_sizex = field.yvel1.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					yvel1[(x_max + 2 + 1 + j) + (k) * yvel1_sizex] = yvel1[(x_max + 1 - j) + (k) * yvel1_sizex];
				}
			}

		}
	}


	if (fields[field_vol_flux_x] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth


			double *vol_flux_x = field.vol_flux_x.data;
			const int vol_flux_x_sizex = field.vol_flux_x.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					vol_flux_x[j + (1 - k) * vol_flux_x_sizex] = vol_flux_x[j + (1 + 2 + k) * vol_flux_x_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth


			double *vol_flux_x = field.vol_flux_x.data;
			const int vol_flux_x_sizex = field.vol_flux_x.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					vol_flux_x[j + (y_max + 2 + k) * vol_flux_x_sizex] = vol_flux_x[j + (y_max - k) * vol_flux_x_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *vol_flux_x = field.vol_flux_x.data;
			const int vol_flux_x_sizex = field.vol_flux_x.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					vol_flux_x[(1 - j) + (k) * vol_flux_x_sizex] = -vol_flux_x[(1 + 2 + j) + (k) * vol_flux_x_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *vol_flux_x = field.vol_flux_x.data;
			const int vol_flux_x_sizex = field.vol_flux_x.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					vol_flux_x[(x_max + j + 1 + 2) + (k) * vol_flux_x_sizex] = -vol_flux_x[(x_max + 1 - j) + (k) * vol_flux_x_sizex];
				}
			}

		}
	}


	if (fields[field_mass_flux_x] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth


			double *mass_flux_x = field.mass_flux_x.data;
			const int mass_flux_x_sizex = field.mass_flux_x.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					mass_flux_x[j + (1 - k) * mass_flux_x_sizex] = mass_flux_x[j + (1 + 2 + k) * mass_flux_x_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth


			double *mass_flux_x = field.mass_flux_x.data;
			const int mass_flux_x_sizex = field.mass_flux_x.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					mass_flux_x[j + (y_max + 2 + k) * mass_flux_x_sizex] = mass_flux_x[j + (y_max - k) * mass_flux_x_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *mass_flux_x = field.mass_flux_x.data;
			const int mass_flux_x_sizex = field.mass_flux_x.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					mass_flux_x[(1 - j) + (k) * mass_flux_x_sizex] = -mass_flux_x[(1 + 2 + j) + (k) * mass_flux_x_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth


			double *mass_flux_x = field.mass_flux_x.data;
			const int mass_flux_x_sizex = field.mass_flux_x.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					mass_flux_x[(x_max + j + 1 + 2) + (k) * mass_flux_x_sizex] = -mass_flux_x[(x_max + 1 - j) + (k) * mass_flux_x_sizex];
				}
			}

		}
	}


	if (fields[field_vol_flux_y] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *vol_flux_y = field.vol_flux_y.data;
			const int vol_flux_y_sizex = field.vol_flux_y.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					vol_flux_y[j + (1 - k) * vol_flux_y_sizex] = -vol_flux_y[j + (1 + 2 + k) * vol_flux_y_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *vol_flux_y = field.vol_flux_y.data;
			const int vol_flux_y_sizex = field.vol_flux_y.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					vol_flux_y[j + (y_max + k + 1 + 2) * vol_flux_y_sizex] = -vol_flux_y[j + (y_max + 1 - k) * vol_flux_y_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth


			double *vol_flux_y = field.vol_flux_y.data;
			const int vol_flux_y_sizex = field.vol_flux_y.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					vol_flux_y[(1 - j) + (k) * vol_flux_y_sizex] = vol_flux_y[(1 + 2 + j) + (k) * vol_flux_y_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth


			double *vol_flux_y = field.vol_flux_y.data;
			const int vol_flux_y_sizex = field.vol_flux_y.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					vol_flux_y[(x_max + 2 + j) + (k) * vol_flux_y_sizex] = vol_flux_y[(x_max - j) + (k) * vol_flux_y_sizex];
				}
			}

		}
	}

	if (fields[field_mass_flux_y] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *mass_flux_y = field.mass_flux_y.data;
			const int mass_flux_y_sizex = field.mass_flux_y.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					mass_flux_y[j + (1 - k) * mass_flux_y_sizex] = -mass_flux_y[j + (1 + 2 + k) * mass_flux_y_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth


			double *mass_flux_y = field.mass_flux_y.data;
			const int mass_flux_y_sizex = field.mass_flux_y.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				for (int k = 0; k < depth; ++k) {
					mass_flux_y[j + (y_max + k + 1 + 2) * mass_flux_y_sizex] = -mass_flux_y[j + (y_max + 1 - k) * mass_flux_y_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth


			double *mass_flux_y = field.mass_flux_y.data;
			const int mass_flux_y_sizex = field.mass_flux_y.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					mass_flux_y[(1 - j) + (k) * mass_flux_y_sizex] = mass_flux_y[(1 + 2 + j) + (k) * mass_flux_y_sizex];
				}
			}

		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth


			double *mass_flux_y = field.mass_flux_y.data;
			const int mass_flux_y_sizex = field.mass_flux_y.sizeX;
			#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
			for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
				for (int j = 0; j < depth; ++j) {
					mass_flux_y[(x_max + 2 + j) + (k) * mass_flux_y_sizex] = mass_flux_y[(x_max - j) + (k) * mass_flux_y_sizex];
				}
			}

		}
	}

}


//  @brief Driver for the halo updates
//  @author Wayne Gaudin
//  @details Invokes the kernels for the internal and external halo cells for
//  the fields specified.
void update_halo(global_variables &globals, int fields[NUM_FIELDS], int depth) {

	double kernel_time = 0;
	if (globals.profiler_on) kernel_time = timer();
	update_tile_halo(globals, fields, depth);
	if (globals.profiler_on) {
		globals.profiler.tile_halo_exchange += timer() - kernel_time;
		kernel_time = timer();
	}

	clover_exchange(globals, fields, depth);

	if (globals.profiler_on) {
		globals.profiler.mpi_halo_exchange += timer() - kernel_time;
		kernel_time = timer();
	}

	if ((globals.chunk.chunk_neighbours[chunk_left] == external_face) ||
	    (globals.chunk.chunk_neighbours[chunk_right] == external_face) ||
	    (globals.chunk.chunk_neighbours[chunk_bottom] == external_face) ||
	    (globals.chunk.chunk_neighbours[chunk_top] == external_face)) {

		#if FLUSH_BUFFER
		globals.hostToDevice();
		#endif


		for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
			tile_type &t = globals.chunk.tiles[tile];
			update_halo_kernel(
					globals.use_target,
					t.info.t_xmin,
					t.info.t_xmax,
					t.info.t_ymin,
					t.info.t_ymax,
					globals.chunk.chunk_neighbours,
					t.info.tile_neighbours,
					t.field,
					fields,
					depth);
		}

		#if FLUSH_BUFFER
		globals.deviceToHost();
		#endif


	}


	if (globals.profiler_on)
		globals.profiler.self_halo_exchange += timer() - kernel_time;
}

