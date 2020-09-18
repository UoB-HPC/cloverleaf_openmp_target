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

#include "update_tile_halo_kernel.h"


//   @brief Fortran kernel to update the external halo cells in a chunk.
//   @author Wayne Gaudin
//   @details Updates halo cells for the required fields at the required depth
//   for any halo cells that lie on an external boundary. The location and type
//   of data governs how this is carried out. External boundaries are always
//   reflective.

void update_tile_halo_l_kernel(
		bool use_target,
		int x_min, int x_max, int y_min, int y_max,
		clover::Buffer2D<double> &density0, clover::Buffer2D<double> &energy0,
		clover::Buffer2D<double> &pressure, clover::Buffer2D<double> &viscosity,
		clover::Buffer2D<double> &soundspeed, clover::Buffer2D<double> &density1,
		clover::Buffer2D<double> &energy1, clover::Buffer2D<double> &xvel0,
		clover::Buffer2D<double> &yvel0, clover::Buffer2D<double> &xvel1,
		clover::Buffer2D<double> &yvel1, clover::Buffer2D<double> &vol_flux_x,
		clover::Buffer2D<double> &vol_flux_y,
		clover::Buffer2D<double> &mass_flux_x,
		clover::Buffer2D<double> &mass_flux_y, int left_xmin, int left_xmax,
		int left_ymin, int left_ymax, clover::Buffer2D<double> &left_density0,
		clover::Buffer2D<double> &left_energy0,
		clover::Buffer2D<double> &left_pressure,
		clover::Buffer2D<double> &left_viscosity,
		clover::Buffer2D<double> &left_soundspeed,
		clover::Buffer2D<double> &left_density1,
		clover::Buffer2D<double> &left_energy1,
		clover::Buffer2D<double> &left_xvel0,
		clover::Buffer2D<double> &left_yvel0,
		clover::Buffer2D<double> &left_xvel1,
		clover::Buffer2D<double> &left_yvel1,
		clover::Buffer2D<double> &left_vol_flux_x,
		clover::Buffer2D<double> &left_vol_flux_y,
		clover::Buffer2D<double> &left_mass_flux_x,
		clover::Buffer2D<double> &left_mass_flux_y, const int fields[NUM_FIELDS],
		int depth) {
	// Density 0
	if (fields[field_density0] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(density0) mapToFrom2D(left_density0))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(density0, x_min - j, k) = idx2(left_density0, left_xmax + 1 - j, k);
			}
		}
	}

	// Density 1
	if (fields[field_density1] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(density1) mapToFrom2D(left_density1))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(density1, x_min - j, k) = idx2(left_density1, left_xmax + 1 - j, k);
			}
		}
	}

	// Energy 0
	if (fields[field_energy0] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(energy0) mapToFrom2D(left_energy0))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(energy0, x_min - j, k) = idx2(left_energy0, left_xmax + 1 - j, k);
			}
		}
	}

	// Energy 1
	if (fields[field_energy1] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(energy1) mapToFrom2D(left_energy1))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(energy1, x_min - j, k) = idx2(left_energy1, left_xmax + 1 - j, k);
			}
		}
	}

	// Pressure
	if (fields[field_pressure] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(pressure) mapToFrom2D(left_pressure))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(pressure, x_min - j, k) = idx2(left_pressure, left_xmax + 1 - j, k);
			}
		}
	}

	// Viscosity
	if (fields[field_viscosity] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(viscosity) mapToFrom2D(left_viscosity))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(viscosity, x_min - j, k) = idx2(left_viscosity, left_xmax + 1 - j, k);
			}
		}
	}

	// Soundspeed
	if (fields[field_soundspeed] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(soundspeed) mapToFrom2D(left_soundspeed))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(soundspeed, x_min - j, k) = idx2(left_soundspeed, left_xmax + 1 - j, k);
			}
		}
	}

	// XVEL 0
	if (fields[field_xvel0] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(xvel0) mapToFrom2D(left_xvel0))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(xvel0, x_min - j, k) = idx2(left_xvel0, left_xmax + 1 - j, k);
			}
		}
	}

	// XVEL 1
	if (fields[field_xvel1] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(xvel1) mapToFrom2D(left_xvel1))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(xvel1, x_min - j, k) = idx2(left_xvel1, left_xmax + 1 - j, k);
			}
		}
	}

	// YVEL 0
	if (fields[field_yvel0] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(yvel0) mapToFrom2D(left_yvel0))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(yvel0, x_min - j, k) = idx2(left_yvel0, left_xmax + 1 - j, k);
			}
		}
	}

	// YVEL 1
	if (fields[field_yvel1] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(yvel1) mapToFrom2D(left_yvel1))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(yvel1, x_min - j, k) = idx2(left_yvel1, left_xmax + 1 - j, k);
			}
		}
	}

	// VOL_FLUX_X
	if (fields[field_vol_flux_x] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(vol_flux_x) mapToFrom2D(left_vol_flux_x))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(vol_flux_x, x_min - j, k) = idx2(left_vol_flux_x, left_xmax + 1 - j, k);
			}
		}
	}

	// MASS_FLUX_X
	if (fields[field_mass_flux_x] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(mass_flux_x) mapToFrom2D(left_mass_flux_x))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(mass_flux_x, x_min - j, k) = idx2(left_mass_flux_x, left_xmax + 1 - j, k);
			}
		}
	}

	// VOL_FLUX_Y
	if (fields[field_vol_flux_y] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(vol_flux_y) mapToFrom2D(left_vol_flux_y))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(vol_flux_y, x_min - j, k) = idx2(left_vol_flux_y, left_xmax + 1 - j, k);
			}
		}
	}

	// MASS_FLUX_Y
	if (fields[field_mass_flux_y] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(mass_flux_y) mapToFrom2D(left_mass_flux_y))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(mass_flux_y, x_min - j, k) = idx2(left_mass_flux_y, left_xmax + 1 - j, k);
			}
		}
	}
}

void update_tile_halo_r_kernel(
		bool use_target,
		int x_min, int x_max, int y_min, int y_max,
		clover::Buffer2D<double> &density0, clover::Buffer2D<double> &energy0,
		clover::Buffer2D<double> &pressure, clover::Buffer2D<double> &viscosity,
		clover::Buffer2D<double> &soundspeed, clover::Buffer2D<double> &density1,
		clover::Buffer2D<double> &energy1, clover::Buffer2D<double> &xvel0,
		clover::Buffer2D<double> &yvel0, clover::Buffer2D<double> &xvel1,
		clover::Buffer2D<double> &yvel1, clover::Buffer2D<double> &vol_flux_x,
		clover::Buffer2D<double> &vol_flux_y,
		clover::Buffer2D<double> &mass_flux_x,
		clover::Buffer2D<double> &mass_flux_y, int right_xmin, int right_xmax,
		int right_ymin, int right_ymax, clover::Buffer2D<double> &right_density0,
		clover::Buffer2D<double> &right_energy0,
		clover::Buffer2D<double> &right_pressure,
		clover::Buffer2D<double> &right_viscosity,
		clover::Buffer2D<double> &right_soundspeed,
		clover::Buffer2D<double> &right_density1,
		clover::Buffer2D<double> &right_energy1,
		clover::Buffer2D<double> &right_xvel0,
		clover::Buffer2D<double> &right_yvel0,
		clover::Buffer2D<double> &right_xvel1,
		clover::Buffer2D<double> &right_yvel1,
		clover::Buffer2D<double> &right_vol_flux_x,
		clover::Buffer2D<double> &right_vol_flux_y,
		clover::Buffer2D<double> &right_mass_flux_x,
		clover::Buffer2D<double> &right_mass_flux_y, const int fields[NUM_FIELDS],
		int depth) {
	// Density 0
	if (fields[field_density0] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(density0) mapToFrom2D(right_density0))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(density0, x_max + 2 + j, k) = idx2(right_density0, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// Density 1
	if (fields[field_density1] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(density1) mapToFrom2D(right_density1))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(density1, x_max + 2 + j, k) = idx2(right_density1, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// Energy 0
	if (fields[field_energy0] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(energy0) mapToFrom2D(right_energy0))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(energy0, x_max + 2 + j, k) = idx2(right_energy0, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// Energy 1
	if (fields[field_energy1] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(energy1) mapToFrom2D(right_energy1))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(energy1, x_max + 2 + j, k) = idx2(right_energy1, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// Pressure
	if (fields[field_pressure] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(pressure) mapToFrom2D(right_pressure))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(pressure, x_max + 2 + j, k) = idx2(right_pressure, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// Viscosity
	if (fields[field_viscosity] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(viscosity) mapToFrom2D(right_viscosity))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(viscosity, x_max + 2 + j, k) = idx2(right_viscosity, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// Soundspeed
	if (fields[field_soundspeed] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(soundspeed) mapToFrom2D(right_soundspeed))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(soundspeed, x_max + 2 + j, k) = idx2(right_soundspeed, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// XVEL 0
	if (fields[field_xvel0] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(xvel0) mapToFrom2D(right_xvel0))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(xvel0, x_max + 1 + 2 + j, k) = idx2(right_xvel0, right_xmin + 1 - 1 + 2 + j, k);
			}
		}
	}

	// XVEL 1
	if (fields[field_xvel1] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(xvel1) mapToFrom2D(right_xvel1))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(xvel1, x_max + 1 + 2 + j, k) = idx2(right_xvel1, right_xmin + 1 - 1 + 2 + j, k);
			}
		}
	}

	// YVEL 0
	if (fields[field_yvel0] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(yvel0) mapToFrom2D(right_yvel0))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(yvel0, x_max + 1 + 2 + j, k) = idx2(right_yvel0, right_xmin + 1 - 1 + 2 + j, k);
			}
		}
	}

	// YVEL 1
	if (fields[field_yvel1] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(yvel1) mapToFrom2D(right_yvel1))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(yvel1, x_max + 1 + 2 + j, k) = idx2(right_yvel1, right_xmin + 1 - 1 + 2 + j, k);
			}
		}
	}

	// VOL_FLUX_X
	if (fields[field_vol_flux_x] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(vol_flux_x) mapToFrom2D(right_vol_flux_x))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(vol_flux_x, x_max + 1 + 2 + j, k) = idx2(right_vol_flux_x, right_xmin + 1 - 1 + 2 + j, k);
			}
		}
	}

	// MASS_FLUX_X
	if (fields[field_mass_flux_x] == 1) {
		// DO k=y_min-depth,y_max+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(mass_flux_x) mapToFrom2D(right_mass_flux_x))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(mass_flux_x, x_max + 1 + 2 + j, k) = idx2(right_mass_flux_x, right_xmin + 1 - 1 + 2 + j, k);
			}
		}
	}

	// VOL_FLUX_Y
	if (fields[field_vol_flux_y] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(vol_flux_y) mapToFrom2D(right_vol_flux_y))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(vol_flux_y, x_max + 2 + j, k) = idx2(right_vol_flux_y, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// MASS_FLUX_Y
	if (fields[field_mass_flux_y] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		omp(parallel(1) enable_target(use_target) mapToFrom2D(mass_flux_y) mapToFrom2D(right_mass_flux_y))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2(mass_flux_y, x_max + 2 + j, k) = idx2(right_mass_flux_y, right_xmin - 1 + 2 + j, k);
			}
		}
	}
}

//  Top and bottom only do xmin -> xmax
//  This is because the corner ghosts will get communicated in the left right
//  communication

void update_tile_halo_t_kernel(
		bool use_target,
		int x_min, int x_max, int y_min, int y_max,
		clover::Buffer2D<double> &density0, clover::Buffer2D<double> &energy0,
		clover::Buffer2D<double> &pressure, clover::Buffer2D<double> &viscosity,
		clover::Buffer2D<double> &soundspeed, clover::Buffer2D<double> &density1,
		clover::Buffer2D<double> &energy1, clover::Buffer2D<double> &xvel0,
		clover::Buffer2D<double> &yvel0, clover::Buffer2D<double> &xvel1,
		clover::Buffer2D<double> &yvel1, clover::Buffer2D<double> &vol_flux_x,
		clover::Buffer2D<double> &vol_flux_y,
		clover::Buffer2D<double> &mass_flux_x,
		clover::Buffer2D<double> &mass_flux_y, int top_xmin, int top_xmax,
		int top_ymin, int top_ymax, clover::Buffer2D<double> &top_density0,
		clover::Buffer2D<double> &top_energy0,
		clover::Buffer2D<double> &top_pressure,
		clover::Buffer2D<double> &top_viscosity,
		clover::Buffer2D<double> &top_soundspeed,
		clover::Buffer2D<double> &top_density1,
		clover::Buffer2D<double> &top_energy1,
		clover::Buffer2D<double> &top_xvel0, clover::Buffer2D<double> &top_yvel0,
		clover::Buffer2D<double> &top_xvel1, clover::Buffer2D<double> &top_yvel1,
		clover::Buffer2D<double> &top_vol_flux_x,
		clover::Buffer2D<double> &top_vol_flux_y,
		clover::Buffer2D<double> &top_mass_flux_x,
		clover::Buffer2D<double> &top_mass_flux_y, const int fields[NUM_FIELDS],
		int depth) {
	// Density 0
	if (fields[field_density0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(density0) mapToFrom2D(top_density0))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(density0, j, y_max + 2 + k) = idx2(top_density0, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// Density 1
	if (fields[field_density1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(density1) mapToFrom2D(top_density1))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(density1, j, y_max + 2 + k) = idx2(top_density1, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// Energy 0
	if (fields[field_energy0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(energy0) mapToFrom2D(top_energy0))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(energy0, j, y_max + 2 + k) = idx2(top_energy0, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// Energy 1
	if (fields[field_energy1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(energy1) mapToFrom2D(top_energy1))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(energy1, j, y_max + 2 + k) = idx2(top_energy1, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// Pressure
	if (fields[field_pressure] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(pressure) mapToFrom2D(top_pressure))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(pressure, j, y_max + 2 + k) = idx2(top_pressure, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// Viscocity
	if (fields[field_viscosity] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(viscosity) mapToFrom2D(top_viscosity))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(viscosity, j, y_max + 2 + k) = idx2(top_viscosity, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// Soundspeed
	if (fields[field_soundspeed] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(soundspeed) mapToFrom2D(top_soundspeed))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(soundspeed, j, y_max + 2 + k) = idx2(top_soundspeed, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// XVEL 0
	if (fields[field_xvel0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(xvel0) mapToFrom2D(top_xvel0))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2(xvel0, j, y_max + 1 + 2 + k) = idx2(top_xvel0, j, top_ymin + 1 - 1 + 2 + k);
			}
		}
	}

	// XVEL 1
	if (fields[field_xvel1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(xvel1) mapToFrom2D(top_xvel1))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2(xvel1, j, y_max + 1 + 2 + k) = idx2(top_xvel1, j, top_ymin + 1 - 1 + 2 + k);
			}
		}
	}

	// YVEL 0
	if (fields[field_yvel0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(yvel0) mapToFrom2D(top_yvel0))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2(yvel0, j, y_max + 1 + 2 + k) = idx2(top_yvel0, j, top_ymin + 1 - 1 + 2 + k);
			}
		}
	}

	// YVEL 1
	if (fields[field_yvel1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(yvel1) mapToFrom2D(top_yvel1))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2(yvel1, j, y_max + 1 + 2 + k) = idx2(top_yvel1, j, top_ymin + 1 - 1 + 2 + k);
			}
		}
	}

	// VOL_FLUX_X
	if (fields[field_vol_flux_x] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(vol_flux_x) mapToFrom2D(top_vol_flux_x))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2(vol_flux_x, j, y_max + 2 + k) = idx2(top_vol_flux_x, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// MASS_FLUX_X
	if (fields[field_mass_flux_x] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(mass_flux_x) mapToFrom2D(top_mass_flux_x))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2(mass_flux_x, j, y_max + 2 + k) = idx2(top_mass_flux_x, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// VOL_FLUX_Y
	if (fields[field_vol_flux_y] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(vol_flux_y) mapToFrom2D(top_vol_flux_y))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(vol_flux_y, j, y_max + 1 + 2 + k) = idx2(top_vol_flux_y, j, top_ymin + 1 - 1 + 2 + k);
			}
		}
	}

	// MASS_FLUX_Y
	if (fields[field_mass_flux_y] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(mass_flux_y) mapToFrom2D(top_mass_flux_y))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(mass_flux_y, j, y_max + 1 + 2 + k) = idx2(top_mass_flux_y, j, top_ymin + 1 - 1 + 2 + k);
			}
		}
	}
}

void update_tile_halo_b_kernel(
		bool use_target,
		int x_min, int x_max, int y_min, int y_max,
		clover::Buffer2D<double> &density0, clover::Buffer2D<double> &energy0,
		clover::Buffer2D<double> &pressure, clover::Buffer2D<double> &viscosity,
		clover::Buffer2D<double> &soundspeed, clover::Buffer2D<double> &density1,
		clover::Buffer2D<double> &energy1, clover::Buffer2D<double> &xvel0,
		clover::Buffer2D<double> &yvel0, clover::Buffer2D<double> &xvel1,
		clover::Buffer2D<double> &yvel1, clover::Buffer2D<double> &vol_flux_x,
		clover::Buffer2D<double> &vol_flux_y,
		clover::Buffer2D<double> &mass_flux_x,
		clover::Buffer2D<double> &mass_flux_y, int bottom_xmin, int bottom_xmax,
		int bottom_ymin, int bottom_ymax,
		clover::Buffer2D<double> &bottom_density0,
		clover::Buffer2D<double> &bottom_energy0,
		clover::Buffer2D<double> &bottom_pressure,
		clover::Buffer2D<double> &bottom_viscosity,
		clover::Buffer2D<double> &bottom_soundspeed,
		clover::Buffer2D<double> &bottom_density1,
		clover::Buffer2D<double> &bottom_energy1,
		clover::Buffer2D<double> &bottom_xvel0,
		clover::Buffer2D<double> &bottom_yvel0,
		clover::Buffer2D<double> &bottom_xvel1,
		clover::Buffer2D<double> &bottom_yvel1,
		clover::Buffer2D<double> &bottom_vol_flux_x,
		clover::Buffer2D<double> &bottom_vol_flux_y,
		clover::Buffer2D<double> &bottom_mass_flux_x,
		clover::Buffer2D<double> &bottom_mass_flux_y, const int fields[NUM_FIELDS],
		int depth) {
	// Density 0
	if (fields[field_density0] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(density0) mapToFrom2D(bottom_density0))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(density0, j, y_min - k) = idx2(bottom_density0, j, bottom_ymax + 1 - k);
			}
		}
	}

	// Density 1
	if (fields[field_density1] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(density1) mapToFrom2D(bottom_density1))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(density1, j, y_min - k) = idx2(bottom_density1, j, bottom_ymax + 1 - k);
			}
		}
	}

	// Energy 0
	if (fields[field_energy0] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(energy0) mapToFrom2D(bottom_energy0))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(energy0, j, y_min - k) = idx2(bottom_energy0, j, bottom_ymax + 1 - k);
			}
		}
	}

	// Energy 1
	if (fields[field_energy1] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(energy1) mapToFrom2D(bottom_energy1))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(energy1, j, y_min - k) = idx2(bottom_energy1, j, bottom_ymax + 1 - k);
			}
		}
	}

	// Pressure
	if (fields[field_pressure] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(pressure) mapToFrom2D(bottom_pressure))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(pressure, j, y_min - k) = idx2(bottom_pressure, j, bottom_ymax + 1 - k);
			}
		}
	}

	// Viscocity
	if (fields[field_viscosity] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(viscosity) mapToFrom2D(bottom_viscosity))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(viscosity, j, y_min - k) = idx2(bottom_viscosity, j, bottom_ymax + 1 - k);
			}
		}
	}

	// Soundspeed
	if (fields[field_soundspeed] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(soundspeed) mapToFrom2D(bottom_soundspeed))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(soundspeed, j, y_min - k) = idx2(bottom_soundspeed, j, bottom_ymax + 1 - k);
			}
		}
	}

	// XVEL 0
	if (fields[field_xvel0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(xvel0) mapToFrom2D(bottom_xvel0))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2(xvel0, j, y_min - k) = idx2(bottom_xvel0, j, bottom_ymax + 1 - k);
			}
		}
	}

	// XVEL 1
	if (fields[field_xvel1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(xvel1) mapToFrom2D(bottom_xvel1))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2(xvel1, j, y_min - k) = idx2(bottom_xvel1, j, bottom_ymax + 1 - k);
			}
		}
	}

	// YVEL 0
	if (fields[field_yvel0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(yvel0) mapToFrom2D(bottom_yvel0))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2(yvel0, j, y_min - k) = idx2(bottom_yvel0, j, bottom_ymax + 1 - k);
			}
		}
	}

	// YVEL 1
	if (fields[field_yvel1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(yvel1) mapToFrom2D(bottom_yvel1))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2(yvel1, j, y_min - k) = idx2(bottom_yvel1, j, bottom_ymax + 1 - k);
			}
		}
	}

	// VOL_FLUX_X
	if (fields[field_vol_flux_x] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(vol_flux_x) mapToFrom2D(bottom_vol_flux_x))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2(vol_flux_x, j, y_min - k) = idx2(bottom_vol_flux_x, j, bottom_ymax + 1 - k);
			}
		}
	}

	// MASS_FLUX_X
	if (fields[field_mass_flux_x] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(mass_flux_x) mapToFrom2D(bottom_mass_flux_x))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2(mass_flux_x, j, y_min - k) = idx2(bottom_mass_flux_x, j, bottom_ymax + 1 - k);
			}
		}
	}

	// VOL_FLUX_Y
	if (fields[field_vol_flux_y] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(vol_flux_y) mapToFrom2D(bottom_vol_flux_y))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(vol_flux_y, j, y_min - k) = idx2(bottom_vol_flux_y, j, bottom_ymax + 1 - k);
			}
		}
	}

	// MASS_FLUX_Y
	if (fields[field_mass_flux_y] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			omp(parallel(1) enable_target(use_target) mapToFrom2D(mass_flux_y) mapToFrom2D(bottom_mass_flux_y))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2(mass_flux_y, j, y_min - k) = idx2(bottom_mass_flux_y, j, bottom_ymax + 1 - k);
			}
		}
	}
}
