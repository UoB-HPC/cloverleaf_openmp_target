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
		clover::Buffer2D<double> &density0_buffer, clover::Buffer2D<double> &energy0_buffer,
		clover::Buffer2D<double> &pressure_buffer, clover::Buffer2D<double> &viscosity_buffer,
		clover::Buffer2D<double> &soundspeed_buffer, clover::Buffer2D<double> &density1_buffer,
		clover::Buffer2D<double> &energy1_buffer, clover::Buffer2D<double> &xvel0_buffer,
		clover::Buffer2D<double> &yvel0_buffer, clover::Buffer2D<double> &xvel1_buffer,
		clover::Buffer2D<double> &yvel1_buffer, clover::Buffer2D<double> &vol_flux_x_buffer,
		clover::Buffer2D<double> &vol_flux_y_buffer,
		clover::Buffer2D<double> &mass_flux_x_buffer,
		clover::Buffer2D<double> &mass_flux_y_buffer, int left_xmin, int left_xmax,
		int left_ymin, int left_ymax, clover::Buffer2D<double> &left_density0_buffer,
		clover::Buffer2D<double> &left_energy0_buffer,
		clover::Buffer2D<double> &left_pressure_buffer,
		clover::Buffer2D<double> &left_viscosity_buffer,
		clover::Buffer2D<double> &left_soundspeed_buffer,
		clover::Buffer2D<double> &left_density1_buffer,
		clover::Buffer2D<double> &left_energy1_buffer,
		clover::Buffer2D<double> &left_xvel0_buffer,
		clover::Buffer2D<double> &left_yvel0_buffer,
		clover::Buffer2D<double> &left_xvel1_buffer,
		clover::Buffer2D<double> &left_yvel1_buffer,
		clover::Buffer2D<double> &left_vol_flux_x_buffer,
		clover::Buffer2D<double> &left_vol_flux_y_buffer,
		clover::Buffer2D<double> &left_mass_flux_x_buffer,
		clover::Buffer2D<double> &left_mass_flux_y_buffer, const int fields[NUM_FIELDS],
		int depth) {
	// Density 0
	if (fields[field_density0] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(density0_buffer, density0)
		mapToFrom2Dfe(left_density0_buffer, left_density0)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,density0, x_min - j, k) = idx2f(,left_density0, left_xmax + 1 - j, k);
			}
		}
	}

	// Density 1
	if (fields[field_density1] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(density1_buffer, density1)
		mapToFrom2Dfe(left_density1_buffer, left_density1)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,density1, x_min - j, k) = idx2f(,left_density1, left_xmax + 1 - j, k);
			}
		}
	}

	// Energy 0
	if (fields[field_energy0] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(energy0_buffer, energy0)
		mapToFrom2Dfe(left_energy0_buffer, left_energy0)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,energy0, x_min - j, k) = idx2f(,left_energy0, left_xmax + 1 - j, k);
			}
		}
	}

	// Energy 1
	if (fields[field_energy1] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(energy1_buffer, energy1)
		mapToFrom2Dfe(left_energy1_buffer, left_energy1)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,energy1, x_min - j, k) = idx2f(,left_energy1, left_xmax + 1 - j, k);
			}
		}
	}

	// Pressure
	if (fields[field_pressure] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(pressure_buffer, pressure)
		mapToFrom2Dfe(left_pressure_buffer, left_pressure)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,pressure, x_min - j, k) = idx2f(,left_pressure, left_xmax + 1 - j, k);
			}
		}
	}

	// Viscosity
	if (fields[field_viscosity] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(viscosity_buffer, viscosity)
		mapToFrom2Dfe(left_viscosity_buffer, left_viscosity)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,viscosity, x_min - j, k) = idx2f(,left_viscosity, left_xmax + 1 - j, k);
			}
		}
	}

	// Soundspeed
	if (fields[field_soundspeed] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(soundspeed_buffer, soundspeed)
		mapToFrom2Dfe(left_soundspeed_buffer, left_soundspeed)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,soundspeed, x_min - j, k) = idx2f(,left_soundspeed, left_xmax + 1 - j, k);
			}
		}
	}

	// XVEL 0
	if (fields[field_xvel0] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		mapToFrom2Dfe(xvel0_buffer, xvel0)
		mapToFrom2Dfe(left_xvel0_buffer, left_xvel0)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,xvel0, x_min - j, k) = idx2f(,left_xvel0, left_xmax + 1 - j, k);
			}
		}
	}

	// XVEL 1
	if (fields[field_xvel1] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		mapToFrom2Dfe(xvel1_buffer, xvel1)
		mapToFrom2Dfe(left_xvel1_buffer, left_xvel1)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,xvel1, x_min - j, k) = idx2f(,left_xvel1, left_xmax + 1 - j, k);
			}
		}
	}

	// YVEL 0
	if (fields[field_yvel0] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		mapToFrom2Dfe(yvel0_buffer, yvel0)
		mapToFrom2Dfe(left_yvel0_buffer, left_yvel0)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,yvel0, x_min - j, k) = idx2f(,left_yvel0, left_xmax + 1 - j, k);
			}
		}
	}

	// YVEL 1
	if (fields[field_yvel1] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		mapToFrom2Dfe(yvel1_buffer, yvel1)
		mapToFrom2Dfe(left_yvel1_buffer, left_yvel1)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,yvel1, x_min - j, k) = idx2f(,left_yvel1, left_xmax + 1 - j, k);
			}
		}
	}

	// VOL_FLUX_X
	if (fields[field_vol_flux_x] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(vol_flux_x_buffer, vol_flux_x)
		mapToFrom2Dfe(left_vol_flux_x_buffer, left_vol_flux_x)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,vol_flux_x, x_min - j, k) = idx2f(,left_vol_flux_x, left_xmax + 1 - j, k);
			}
		}
	}

	// MASS_FLUX_X
	if (fields[field_mass_flux_x] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(mass_flux_x_buffer, mass_flux_x)
		mapToFrom2Dfe(left_mass_flux_x_buffer, left_mass_flux_x)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,mass_flux_x, x_min - j, k) = idx2f(,left_mass_flux_x, left_xmax + 1 - j, k);
			}
		}
	}

	// VOL_FLUX_Y
	if (fields[field_vol_flux_y] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		mapToFrom2Dfe(vol_flux_y_buffer, vol_flux_y)
		mapToFrom2Dfe(left_vol_flux_y_buffer, left_vol_flux_y)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,vol_flux_y, x_min - j, k) = idx2f(,left_vol_flux_y, left_xmax + 1 - j, k);
			}
		}
	}

	// MASS_FLUX_Y
	if (fields[field_mass_flux_y] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		mapToFrom2Dfe(mass_flux_y_buffer, mass_flux_y)
		mapToFrom2Dfe(left_mass_flux_y_buffer, left_mass_flux_y)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,mass_flux_y, x_min - j, k) = idx2f(,left_mass_flux_y, left_xmax + 1 - j, k);
			}
		}
	}
}

void update_tile_halo_r_kernel(
		bool use_target,
		int x_min, int x_max, int y_min, int y_max,
		clover::Buffer2D<double> &density0_buffer, clover::Buffer2D<double> &energy0_buffer,
		clover::Buffer2D<double> &pressure_buffer, clover::Buffer2D<double> &viscosity_buffer,
		clover::Buffer2D<double> &soundspeed_buffer, clover::Buffer2D<double> &density1_buffer,
		clover::Buffer2D<double> &energy1_buffer, clover::Buffer2D<double> &xvel0_buffer,
		clover::Buffer2D<double> &yvel0_buffer, clover::Buffer2D<double> &xvel1_buffer,
		clover::Buffer2D<double> &yvel1_buffer, clover::Buffer2D<double> &vol_flux_x_buffer,
		clover::Buffer2D<double> &vol_flux_y_buffer,
		clover::Buffer2D<double> &mass_flux_x_buffer,
		clover::Buffer2D<double> &mass_flux_y_buffer, int right_xmin, int right_xmax,
		int right_ymin, int right_ymax, clover::Buffer2D<double> &right_density0_buffer,
		clover::Buffer2D<double> &right_energy0_buffer,
		clover::Buffer2D<double> &right_pressure_buffer,
		clover::Buffer2D<double> &right_viscosity_buffer,
		clover::Buffer2D<double> &right_soundspeed_buffer,
		clover::Buffer2D<double> &right_density1_buffer,
		clover::Buffer2D<double> &right_energy1_buffer,
		clover::Buffer2D<double> &right_xvel0_buffer,
		clover::Buffer2D<double> &right_yvel0_buffer,
		clover::Buffer2D<double> &right_xvel1_buffer,
		clover::Buffer2D<double> &right_yvel1_buffer,
		clover::Buffer2D<double> &right_vol_flux_x_buffer,
		clover::Buffer2D<double> &right_vol_flux_y_buffer,
		clover::Buffer2D<double> &right_mass_flux_x_buffer,
		clover::Buffer2D<double> &right_mass_flux_y_buffer, const int fields[NUM_FIELDS],
		int depth) {
	// Density 0
	if (fields[field_density0] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(density0_buffer, density0)
		mapToFrom2Dfe(right_density0_buffer, right_density0)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,density0, x_max + 2 + j, k) = idx2f(,right_density0, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// Density 1
	if (fields[field_density1] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(density1_buffer, density1)
		mapToFrom2Dfe(right_density1_buffer, right_density1)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,density1, x_max + 2 + j, k) = idx2f(,right_density1, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// Energy 0
	if (fields[field_energy0] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(energy0_buffer, energy0)
		mapToFrom2Dfe(right_energy0_buffer, right_energy0)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,energy0, x_max + 2 + j, k) = idx2f(,right_energy0, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// Energy 1
	if (fields[field_energy1] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(energy1_buffer, energy1)
		mapToFrom2Dfe(right_energy1_buffer, right_energy1)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,energy1, x_max + 2 + j, k) = idx2f(,right_energy1, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// Pressure
	if (fields[field_pressure] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(pressure_buffer, pressure)
		mapToFrom2Dfe(right_pressure_buffer, right_pressure)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,pressure, x_max + 2 + j, k) = idx2f(,right_pressure, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// Viscosity
	if (fields[field_viscosity] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(viscosity_buffer, viscosity)
		mapToFrom2Dfe(right_viscosity_buffer, right_viscosity)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,viscosity, x_max + 2 + j, k) = idx2f(,right_viscosity, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// Soundspeed
	if (fields[field_soundspeed] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(soundspeed_buffer, soundspeed)
		mapToFrom2Dfe(right_soundspeed_buffer, right_soundspeed)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,soundspeed, x_max + 2 + j, k) = idx2f(,right_soundspeed, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// XVEL 0
	if (fields[field_xvel0] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		mapToFrom2Dfe(xvel0_buffer, xvel0)
		mapToFrom2Dfe(right_xvel0_buffer, right_xvel0)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,xvel0, x_max + 1 + 2 + j, k) = idx2f(,right_xvel0, right_xmin + 1 - 1 + 2 + j, k);
			}
		}
	}

	// XVEL 1
	if (fields[field_xvel1] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		mapToFrom2Dfe(xvel1_buffer, xvel1)
		mapToFrom2Dfe(right_xvel1_buffer, right_xvel1)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,xvel1, x_max + 1 + 2 + j, k) = idx2f(,right_xvel1, right_xmin + 1 - 1 + 2 + j, k);
			}
		}
	}

	// YVEL 0
	if (fields[field_yvel0] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		mapToFrom2Dfe(yvel0_buffer, yvel0)
		mapToFrom2Dfe(right_yvel0_buffer, right_yvel0)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,yvel0, x_max + 1 + 2 + j, k) = idx2f(,right_yvel0, right_xmin + 1 - 1 + 2 + j, k);
			}
		}
	}

	// YVEL 1
	if (fields[field_yvel1] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		mapToFrom2Dfe(yvel1_buffer, yvel1)
		mapToFrom2Dfe(right_yvel1_buffer, right_yvel1)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,yvel1, x_max + 1 + 2 + j, k) = idx2f(,right_yvel1, right_xmin + 1 - 1 + 2 + j, k);
			}
		}
	}

	// VOL_FLUX_X
	if (fields[field_vol_flux_x] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(vol_flux_x_buffer, vol_flux_x)
		mapToFrom2Dfe(right_vol_flux_x_buffer, right_vol_flux_x)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,vol_flux_x, x_max + 1 + 2 + j, k) = idx2f(,right_vol_flux_x, right_xmin + 1 - 1 + 2 + j, k);
			}
		}
	}

	// MASS_FLUX_X
	if (fields[field_mass_flux_x] == 1) {
		// DO k=y_min-depth,y_max+depth

		mapToFrom2Dfe(mass_flux_x_buffer, mass_flux_x)
		mapToFrom2Dfe(right_mass_flux_x_buffer, right_mass_flux_x)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,mass_flux_x, x_max + 1 + 2 + j, k) = idx2f(,right_mass_flux_x, right_xmin + 1 - 1 + 2 + j, k);
			}
		}
	}

	// VOL_FLUX_Y
	if (fields[field_vol_flux_y] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		mapToFrom2Dfe(vol_flux_y_buffer, vol_flux_y)
		mapToFrom2Dfe(right_vol_flux_y_buffer, right_vol_flux_y)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,vol_flux_y, x_max + 2 + j, k) = idx2f(,right_vol_flux_y, right_xmin - 1 + 2 + j, k);
			}
		}
	}

	// MASS_FLUX_Y
	if (fields[field_mass_flux_y] == 1) {
		// DO k=y_min-depth,y_max+1+depth

		mapToFrom2Dfe(mass_flux_y_buffer, mass_flux_y)
		mapToFrom2Dfe(right_mass_flux_y_buffer, right_mass_flux_y)
		omp(parallel(1) enable_target(use_target))
		for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
			for (int j = 0; j < depth; ++j) {
				idx2f(,mass_flux_y, x_max + 2 + j, k) = idx2f(,right_mass_flux_y, right_xmin - 1 + 2 + j, k);
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
		clover::Buffer2D<double> &density0_buffer, clover::Buffer2D<double> &energy0_buffer,
		clover::Buffer2D<double> &pressure_buffer, clover::Buffer2D<double> &viscosity_buffer,
		clover::Buffer2D<double> &soundspeed_buffer, clover::Buffer2D<double> &density1_buffer,
		clover::Buffer2D<double> &energy1_buffer, clover::Buffer2D<double> &xvel0_buffer,
		clover::Buffer2D<double> &yvel0_buffer, clover::Buffer2D<double> &xvel1_buffer,
		clover::Buffer2D<double> &yvel1_buffer, clover::Buffer2D<double> &vol_flux_x_buffer,
		clover::Buffer2D<double> &vol_flux_y_buffer,
		clover::Buffer2D<double> &mass_flux_x_buffer,
		clover::Buffer2D<double> &mass_flux_y_buffer, int top_xmin, int top_xmax,
		int top_ymin, int top_ymax, clover::Buffer2D<double> &top_density0_buffer,
		clover::Buffer2D<double> &top_energy0_buffer,
		clover::Buffer2D<double> &top_pressure_buffer,
		clover::Buffer2D<double> &top_viscosity_buffer,
		clover::Buffer2D<double> &top_soundspeed_buffer,
		clover::Buffer2D<double> &top_density1_buffer,
		clover::Buffer2D<double> &top_energy1_buffer,
		clover::Buffer2D<double> &top_xvel0_buffer, clover::Buffer2D<double> &top_yvel0_buffer,
		clover::Buffer2D<double> &top_xvel1_buffer, clover::Buffer2D<double> &top_yvel1_buffer,
		clover::Buffer2D<double> &top_vol_flux_x_buffer,
		clover::Buffer2D<double> &top_vol_flux_y_buffer,
		clover::Buffer2D<double> &top_mass_flux_x_buffer,
		clover::Buffer2D<double> &top_mass_flux_y_buffer, const int fields[NUM_FIELDS],
		int depth) {
	// Density 0
	if (fields[field_density0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(density0_buffer, density0)
			mapToFrom2Dfe(top_density0_buffer, top_density0)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,density0, j, y_max + 2 + k) = idx2f(,top_density0, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// Density 1
	if (fields[field_density1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(density1_buffer, density1)
			mapToFrom2Dfe(top_density1_buffer, top_density1)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,density1, j, y_max + 2 + k) = idx2f(,top_density1, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// Energy 0
	if (fields[field_energy0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(energy0_buffer, energy0)
			mapToFrom2Dfe(top_energy0_buffer, top_energy0)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,energy0, j, y_max + 2 + k) = idx2f(,top_energy0, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// Energy 1
	if (fields[field_energy1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(energy1_buffer, energy1)
			mapToFrom2Dfe(top_energy1_buffer, top_energy1)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,energy1, j, y_max + 2 + k) = idx2f(,top_energy1, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// Pressure
	if (fields[field_pressure] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(pressure_buffer, pressure)
			mapToFrom2Dfe(top_pressure_buffer, top_pressure)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,pressure, j, y_max + 2 + k) = idx2f(,top_pressure, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// Viscocity
	if (fields[field_viscosity] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(viscosity_buffer, viscosity)
			mapToFrom2Dfe(top_viscosity_buffer, top_viscosity)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,viscosity, j, y_max + 2 + k) = idx2f(,top_viscosity, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// Soundspeed
	if (fields[field_soundspeed] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(soundspeed_buffer, soundspeed)
			mapToFrom2Dfe(top_soundspeed_buffer, top_soundspeed)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,soundspeed, j, y_max + 2 + k) = idx2f(,top_soundspeed, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// XVEL 0
	if (fields[field_xvel0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			mapToFrom2Dfe(xvel0_buffer, xvel0)
			mapToFrom2Dfe(top_xvel0_buffer, top_xvel0)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2f(,xvel0, j, y_max + 1 + 2 + k) = idx2f(,top_xvel0, j, top_ymin + 1 - 1 + 2 + k);
			}
		}
	}

	// XVEL 1
	if (fields[field_xvel1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			mapToFrom2Dfe(xvel1_buffer, xvel1)
			mapToFrom2Dfe(top_xvel1_buffer, top_xvel1)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2f(,xvel1, j, y_max + 1 + 2 + k) = idx2f(,top_xvel1, j, top_ymin + 1 - 1 + 2 + k);
			}
		}
	}

	// YVEL 0
	if (fields[field_yvel0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			mapToFrom2Dfe(yvel0_buffer, yvel0)
			mapToFrom2Dfe(top_yvel0_buffer, top_yvel0)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2f(,yvel0, j, y_max + 1 + 2 + k) = idx2f(,top_yvel0, j, top_ymin + 1 - 1 + 2 + k);
			}
		}
	}

	// YVEL 1
	if (fields[field_yvel1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			mapToFrom2Dfe(yvel1_buffer, yvel1)
			mapToFrom2Dfe(top_yvel1_buffer, top_yvel1)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2f(,yvel1, j, y_max + 1 + 2 + k) = idx2f(,top_yvel1, j, top_ymin + 1 - 1 + 2 + k);
			}
		}
	}

	// VOL_FLUX_X
	if (fields[field_vol_flux_x] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			mapToFrom2Dfe(vol_flux_x_buffer, vol_flux_x)
			mapToFrom2Dfe(top_vol_flux_x_buffer, top_vol_flux_x)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2f(,vol_flux_x, j, y_max + 2 + k) = idx2f(,top_vol_flux_x, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// MASS_FLUX_X
	if (fields[field_mass_flux_x] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			mapToFrom2Dfe(mass_flux_x_buffer, mass_flux_x)
			mapToFrom2Dfe(top_mass_flux_x_buffer, top_mass_flux_x)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2f(,mass_flux_x, j, y_max + 2 + k) = idx2f(,top_mass_flux_x, j, top_ymin - 1 + 2 + k);
			}
		}
	}

	// VOL_FLUX_Y
	if (fields[field_vol_flux_y] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(vol_flux_y_buffer, vol_flux_y)
			mapToFrom2Dfe(top_vol_flux_y_buffer, top_vol_flux_y)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,vol_flux_y, j, y_max + 1 + 2 + k) = idx2f(,top_vol_flux_y, j, top_ymin + 1 - 1 + 2 + k);
			}
		}
	}

	// MASS_FLUX_Y
	if (fields[field_mass_flux_y] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(mass_flux_y_buffer, mass_flux_y)
			mapToFrom2Dfe(top_mass_flux_y_buffer, top_mass_flux_y)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,mass_flux_y, j, y_max + 1 + 2 + k) = idx2f(,top_mass_flux_y, j, top_ymin + 1 - 1 + 2 + k);
			}
		}
	}
}

void update_tile_halo_b_kernel(
		bool use_target,
		int x_min, int x_max, int y_min, int y_max,
		clover::Buffer2D<double> &density0_buffer, clover::Buffer2D<double> &energy0_buffer,
		clover::Buffer2D<double> &pressure_buffer, clover::Buffer2D<double> &viscosity_buffer,
		clover::Buffer2D<double> &soundspeed_buffer, clover::Buffer2D<double> &density1_buffer,
		clover::Buffer2D<double> &energy1_buffer, clover::Buffer2D<double> &xvel0_buffer,
		clover::Buffer2D<double> &yvel0_buffer, clover::Buffer2D<double> &xvel1_buffer,
		clover::Buffer2D<double> &yvel1_buffer, clover::Buffer2D<double> &vol_flux_x_buffer,
		clover::Buffer2D<double> &vol_flux_y_buffer,
		clover::Buffer2D<double> &mass_flux_x_buffer,
		clover::Buffer2D<double> &mass_flux_y_buffer, int bottom_xmin, int bottom_xmax,
		int bottom_ymin, int bottom_ymax,
		clover::Buffer2D<double> &bottom_density0_buffer,
		clover::Buffer2D<double> &bottom_energy0_buffer,
		clover::Buffer2D<double> &bottom_pressure_buffer,
		clover::Buffer2D<double> &bottom_viscosity_buffer,
		clover::Buffer2D<double> &bottom_soundspeed_buffer,
		clover::Buffer2D<double> &bottom_density1_buffer,
		clover::Buffer2D<double> &bottom_energy1_buffer,
		clover::Buffer2D<double> &bottom_xvel0_buffer,
		clover::Buffer2D<double> &bottom_yvel0_buffer,
		clover::Buffer2D<double> &bottom_xvel1_buffer,
		clover::Buffer2D<double> &bottom_yvel1_buffer,
		clover::Buffer2D<double> &bottom_vol_flux_x_buffer,
		clover::Buffer2D<double> &bottom_vol_flux_y_buffer,
		clover::Buffer2D<double> &bottom_mass_flux_x_buffer,
		clover::Buffer2D<double> &bottom_mass_flux_y_buffer, const int fields[NUM_FIELDS],
		int depth) {
	// Density 0
	if (fields[field_density0] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(density0_buffer, density0)
			mapToFrom2Dfe(bottom_density0_buffer, bottom_density0)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,density0, j, y_min - k) = idx2f(,bottom_density0, j, bottom_ymax + 1 - k);
			}
		}
	}

	// Density 1
	if (fields[field_density1] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(density1_buffer, density1)
			mapToFrom2Dfe(bottom_density1_buffer, bottom_density1)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,density1, j, y_min - k) = idx2f(,bottom_density1, j, bottom_ymax + 1 - k);
			}
		}
	}

	// Energy 0
	if (fields[field_energy0] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(energy0_buffer, energy0)
			mapToFrom2Dfe(bottom_energy0_buffer, bottom_energy0)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,energy0, j, y_min - k) = idx2f(,bottom_energy0, j, bottom_ymax + 1 - k);
			}
		}
	}

	// Energy 1
	if (fields[field_energy1] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(energy1_buffer, energy1)
			mapToFrom2Dfe(bottom_energy1_buffer, bottom_energy1)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,energy1, j, y_min - k) = idx2f(,bottom_energy1, j, bottom_ymax + 1 - k);
			}
		}
	}

	// Pressure
	if (fields[field_pressure] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(pressure_buffer, pressure)
			mapToFrom2Dfe(bottom_pressure_buffer, bottom_pressure)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,pressure, j, y_min - k) = idx2f(,bottom_pressure, j, bottom_ymax + 1 - k);
			}
		}
	}

	// Viscocity
	if (fields[field_viscosity] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(viscosity_buffer, viscosity)
			mapToFrom2Dfe(bottom_viscosity_buffer, bottom_viscosity)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,viscosity, j, y_min - k) = idx2f(,bottom_viscosity, j, bottom_ymax + 1 - k);
			}
		}
	}

	// Soundspeed
	if (fields[field_soundspeed] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(soundspeed_buffer, soundspeed)
			mapToFrom2Dfe(bottom_soundspeed_buffer, bottom_soundspeed)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,soundspeed, j, y_min - k) = idx2f(,bottom_soundspeed, j, bottom_ymax + 1 - k);
			}
		}
	}

	// XVEL 0
	if (fields[field_xvel0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			mapToFrom2Dfe(xvel0_buffer, xvel0)
			mapToFrom2Dfe(bottom_xvel0_buffer, bottom_xvel0)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2f(,xvel0, j, y_min - k) = idx2f(,bottom_xvel0, j, bottom_ymax + 1 - k);
			}
		}
	}

	// XVEL 1
	if (fields[field_xvel1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			mapToFrom2Dfe(xvel1_buffer, xvel1)
			mapToFrom2Dfe(bottom_xvel1_buffer, bottom_xvel1)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2f(,xvel1, j, y_min - k) = idx2f(,bottom_xvel1, j, bottom_ymax + 1 - k);
			}
		}
	}

	// YVEL 0
	if (fields[field_yvel0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			mapToFrom2Dfe(yvel0_buffer, yvel0)
			mapToFrom2Dfe(bottom_yvel0_buffer, bottom_yvel0)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2f(,yvel0, j, y_min - k) = idx2f(,bottom_yvel0, j, bottom_ymax + 1 - k);
			}
		}
	}

	// YVEL 1
	if (fields[field_yvel1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			mapToFrom2Dfe(yvel1_buffer, yvel1)
			mapToFrom2Dfe(bottom_yvel1_buffer, bottom_yvel1)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2f(,yvel1, j, y_min - k) = idx2f(,bottom_yvel1, j, bottom_ymax + 1 - k);
			}
		}
	}

	// VOL_FLUX_X
	if (fields[field_vol_flux_x] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			mapToFrom2Dfe(vol_flux_x_buffer, vol_flux_x)
			mapToFrom2Dfe(bottom_vol_flux_x_buffer, bottom_vol_flux_x)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2f(,vol_flux_x, j, y_min - k) = idx2f(,bottom_vol_flux_x, j, bottom_ymax + 1 - k);
			}
		}
	}

	// MASS_FLUX_X
	if (fields[field_mass_flux_x] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth

			mapToFrom2Dfe(mass_flux_x_buffer, mass_flux_x)
			mapToFrom2Dfe(bottom_mass_flux_x_buffer, bottom_mass_flux_x)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + 1 + depth + 2); j++) {
				idx2f(,mass_flux_x, j, y_min - k) = idx2f(,bottom_mass_flux_x, j, bottom_ymax + 1 - k);
			}
		}
	}

	// VOL_FLUX_Y
	if (fields[field_vol_flux_y] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(vol_flux_y_buffer, vol_flux_y)
			mapToFrom2Dfe(bottom_vol_flux_y_buffer, bottom_vol_flux_y)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,vol_flux_y, j, y_min - k) = idx2f(,bottom_vol_flux_y, j, bottom_ymax + 1 - k);
			}
		}
	}

	// MASS_FLUX_Y
	if (fields[field_mass_flux_y] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth

			mapToFrom2Dfe(mass_flux_y_buffer, mass_flux_y)
			mapToFrom2Dfe(bottom_mass_flux_y_buffer, bottom_mass_flux_y)
			omp(parallel(1) enable_target(use_target))
			for (int j = (x_min - depth + 1); j < (x_max + depth + 2); j++) {
				idx2f(,mass_flux_y, j, y_min - k) = idx2f(,bottom_mass_flux_y, j, bottom_ymax + 1 - k);
			}
		}
	}
}
