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


#include <cmath>
#include "advec_cell.h"



//  @brief Fortran cell advection kernel.
//  @author Wayne Gaudin
//  @details Performs a second order advective remap using van-Leer limiting
//  with directional splitting.
void advec_cell_kernel(
		bool use_target,
		int x_min,
		int x_max,
		int y_min,
		int y_max,
		int dir,
		int sweep_number,
		field_type &field) {

	const double one_by_six = 1.0 / 6.0;

	if (dir == g_xdir) {

		// DO k=y_min-2,y_max+2
		//   DO j=x_min-2,x_max+2

		if (sweep_number == 1) {


			mapToFrom2Df(field, volume)
			mapToFrom2Df(field, vol_flux_x)
			mapToFrom2Df(field, vol_flux_y)
			mapToFrom2Dfn(field, work_array1, pre_vol)
			mapToFrom2Dfn(field, work_array2, post_vol)

			omp(parallel(2) enable_target(use_target))
			for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
				for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
					idx2f(field, pre_vol, i, j) = idx2f(field, volume, i, j) +
					                              (idx2f(field, vol_flux_x, i + 1, j + 0) - idx2f(field, vol_flux_x, i, j) + idx2f(field, vol_flux_y, i + 0, j + 1) -
					                               idx2f(field, vol_flux_y, i, j));
					idx2f(field, post_vol, i, j) = idx2f(field, pre_vol, i, j) - (idx2f(field, vol_flux_x, i + 1, j + 0) - idx2f(field, vol_flux_x, i, j));
				}
			}


		} else {


			mapToFrom2Df(field, volume)
			mapToFrom2Df(field, vol_flux_x)
			mapToFrom2Dfn(field, work_array1, pre_vol)
			mapToFrom2Dfn(field, work_array2, post_vol)

			omp(parallel(2) enable_target(use_target))
			for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
				for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
					idx2f(field, pre_vol, i, j) = idx2f(field, volume, i, j) + idx2f(field, vol_flux_x, i + 1, j + 0) - idx2f(field, vol_flux_x, i, j);
					idx2f(field, post_vol, i, j) = idx2f(field, volume, i, j);
				}
			}

		}

		// DO k=y_min,y_max
		//   DO j=x_min,x_max+2
		mapToFrom1Df(field, vertexdx)
		mapToFrom2Df(field, density1)
		mapToFrom2Df(field, energy1)
		mapToFrom2Df(field, mass_flux_x)
		mapToFrom2Df(field, vol_flux_x)
		mapToFrom2Dfn(field, work_array1, pre_vol)
		mapToFrom2Dfn(field, work_array7, ener_flux)

		omp(parallel(2) enable_target(use_target))
		for (int j = (y_min + 1); j < (y_max + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2 + 2); i++)
				({
					int upwind, donor, downwind, dif;
					double sigmat, sigma3, sigma4, sigmav, sigma, sigmam, diffuw, diffdw, limiter, wind;
					if (idx2f(field, vol_flux_x, i, j) > 0.0) {
						upwind = i - 2;
						donor = i - 1;
						downwind = i;
						dif = donor;
					} else {
						upwind = std::min(i + 1, x_max + 2);
						donor = i;
						downwind = i - 1;
						dif = upwind;
					}
					sigmat = std::fabs(idx2f(field, vol_flux_x, i, j)) / idx2f(field, pre_vol, donor, j);
					sigma3 = (1.0 + sigmat) * (idx1f(field, vertexdx, i) / idx1f(field, vertexdx, dif));
					sigma4 = 2.0 - sigmat;
//					sigma = sigmat;
					sigmav = sigmat;
					diffuw = idx2f(field, density1, donor, j) - idx2f(field, density1, upwind, j);
					diffdw = idx2f(field, density1, downwind, j) - idx2f(field, density1, donor, j);
					wind = 1.0;
					if (diffdw <= 0.0)wind = -1.0;
					if (diffuw * diffdw > 0.0) {
						limiter = (1.0 - sigmav) * wind *
						          std::fmin(std::fmin(
								          std::fabs(diffuw),
								          std::fabs(diffdw)),
						                    one_by_six * (sigma3 * std::fabs(diffuw) + sigma4 * std::fabs(diffdw)));
					} else {
						limiter = 0.0;
					}
					idx2f(field, mass_flux_x, i, j) = idx2f(field, vol_flux_x, i, j) * (idx2f(field, density1, donor, j) + limiter);
					sigmam = std::fabs(idx2f(field, mass_flux_x, i, j)) / (idx2f(field, density1, donor, j) * idx2f(field, pre_vol, donor, j));
					diffuw = idx2f(field, energy1, donor, j) - idx2f(field, energy1, upwind, j);
					diffdw = idx2f(field, energy1, downwind, j) - idx2f(field, energy1, donor, j);
					wind = 1.0;
					if (diffdw <= 0.0)wind = -1.0;
					if (diffuw * diffdw > 0.0) {
						limiter = (1.0 - sigmam) *
						          wind *
						          std::fmin(std::fmin(
								          std::fabs(diffuw),
								          std::fabs(diffdw)),
						                    one_by_six * (sigma3 * std::fabs(diffuw) + sigma4 * std::fabs(diffdw)));
					} else {
						limiter = 0.0;
					}
					idx2f(field, ener_flux, i, j) = idx2f(field, mass_flux_x, i, j) * (idx2f(field, energy1, donor, j) + limiter);
				});
		}



		// DO k=y_min,y_max
		//   DO j=x_min,x_max



		omp(parallel(2) enable_target(use_target))
		for (int j = (y_min + 1); j < (y_max + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2); i++) {
				double pre_mass_s = idx2f(field, density1, i, j) * idx2f(field, pre_vol, i, j);
				double post_mass_s = pre_mass_s + idx2f(field, mass_flux_x, i, j) - idx2f(field, mass_flux_x, i + 1, j + 0);
				double post_ener_s = (idx2f(field, energy1, i, j) * pre_mass_s + idx2f(field, ener_flux, i, j) - idx2f(field, ener_flux, i + 1, j + 0)) / post_mass_s;
				double advec_vol_s = idx2f(field, pre_vol, i, j) + idx2f(field, vol_flux_x, i, j) - idx2f(field, vol_flux_x, i + 1, j + 0);
				idx2f(field, density1, i, j) = post_mass_s / advec_vol_s;
				idx2f(field, energy1, i, j) = post_ener_s;
			}
		}

	} else if (dir == g_ydir) {

		// DO k=y_min-2,y_max+2
		//   DO j=x_min-2,x_max+2

		if (sweep_number == 1) {


			mapToFrom2Df(field, volume)
			mapToFrom2Df(field, vol_flux_x)
			mapToFrom2Df(field, vol_flux_y)
			mapToFrom2Dfn(field, work_array1, pre_vol)
			mapToFrom2Dfn(field, work_array2, post_vol)

			omp(parallel(2) enable_target(use_target))
			for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
				for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
					idx2f(field, pre_vol, i, j) = idx2f(field, volume, i, j) +
					                              (idx2f(field, vol_flux_y, i + 0, j + 1) - idx2f(field, vol_flux_y, i, j) + idx2f(field, vol_flux_x, i + 1, j + 0) -
					                               idx2f(field, vol_flux_x, i, j));
					idx2f(field, post_vol, i, j) = idx2f(field, pre_vol, i, j) - (idx2f(field, vol_flux_y, i + 0, j + 1) - idx2f(field, vol_flux_y, i, j));
				}
			}


		} else {


			mapToFrom2Df(field, volume)
			mapToFrom2Df(field, vol_flux_y)
			mapToFrom2Dfn(field, work_array1, pre_vol)
			mapToFrom2Dfn(field, work_array2, post_vol)

			omp(parallel(2) enable_target(use_target))
			for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
				for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
					idx2f(field, pre_vol, i, j) = idx2f(field, volume, i, j) + idx2f(field, vol_flux_y, i + 0, j + 1) - idx2f(field, vol_flux_y, i, j);
					idx2f(field, post_vol, i, j) = idx2f(field, volume, i, j);
				}
			}


		}


		// DO k=y_min,y_max+2
		//   DO j=x_min,x_max
		mapToFrom1Df(field, vertexdy)
		mapToFrom2Df(field, density1)
		mapToFrom2Df(field, energy1)
		mapToFrom2Df(field, mass_flux_y)
		mapToFrom2Df(field, vol_flux_y)
		mapToFrom2Dfn(field, work_array1, pre_vol)
		mapToFrom2Dfn(field, work_array7, ener_flux)
		omp(parallel(2) enable_target(use_target))
		for (int j = (y_min + 1); j < (y_max + 2 + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2); i++)
				({
					int upwind, donor, downwind, dif;
					double sigmat, sigma3, sigma4, sigmav, sigma, sigmam, diffuw, diffdw, limiter, wind;
					if (idx2f(field, vol_flux_y, i, j) > 0.0) {
						upwind = j - 2;
						donor = j - 1;
						downwind = j;
						dif = donor;
					} else {
						upwind = std::min(j + 1, y_max + 2);
						donor = j;
						downwind = j - 1;
						dif = upwind;
					}
					sigmat = std::fabs(idx2f(field, vol_flux_y, i, j)) / idx2f(field, pre_vol, i, donor);
					sigma3 = (1.0 + sigmat) * (idx1f(field, vertexdy, j) / idx1f(field, vertexdy, dif));
					sigma4 = 2.0 - sigmat;
//					sigma = sigmat;
					sigmav = sigmat;
					diffuw = idx2f(field, density1, i, donor) - idx2f(field, density1, i, upwind);
					diffdw = idx2f(field, density1, i, downwind) - idx2f(field, density1, i, donor);
					wind = 1.0;
					if (diffdw <= 0.0)wind = -1.0;
					if (diffuw * diffdw > 0.0) {
						limiter = (1.0 - sigmav) * wind *
						          std::fmin(std::fmin(
								          std::fabs(diffuw),
								          std::fabs(diffdw)),
						                    one_by_six * (sigma3 * std::fabs(diffuw) + sigma4 * std::fabs(diffdw)));
					} else {
						limiter = 0.0;
					}
					idx2f(field, mass_flux_y, i, j) = idx2f(field, vol_flux_y, i, j) * (idx2f(field, density1, i, donor) + limiter);
					sigmam = std::fabs(idx2f(field, mass_flux_y, i, j)) / (idx2f(field, density1, i, donor) * idx2f(field, pre_vol, i, donor));
					diffuw = idx2f(field, energy1, i, donor) - idx2f(field, energy1, i, upwind);
					diffdw = idx2f(field, energy1, i, downwind) - idx2f(field, energy1, i, donor);
					wind = 1.0;
					if (diffdw <= 0.0)wind = -1.0;
					if (diffuw * diffdw > 0.0) {
						limiter = (1.0 - sigmam) * wind *
						          std::fmin(std::fmin(
								          std::fabs(diffuw),
								          std::fabs(diffdw)),
						                    one_by_six * (sigma3 * std::fabs(diffuw) + sigma4 * std::fabs(diffdw)));
					} else {
						limiter = 0.0;
					}
					idx2f(field, ener_flux, i, j) = idx2f(field, mass_flux_y, i, j) * (idx2f(field, energy1, i, donor) + limiter);
				});
		}


		// DO k=y_min,y_max
		//   DO j=x_min,x_max


		omp(parallel(2) enable_target(use_target))
		for (int j = (y_min + 1); j < (y_max + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2); i++) {
				double pre_mass_s = idx2f(field, density1, i, j) * idx2f(field, pre_vol, i, j);
				double post_mass_s = pre_mass_s + idx2f(field, mass_flux_y, i, j) - idx2f(field, mass_flux_y, i + 0, j + 1);
				double post_ener_s = (idx2f(field, energy1, i, j) * pre_mass_s + idx2f(field, ener_flux, i, j) - idx2f(field, ener_flux, i + 0, j + 1)) / post_mass_s;
				double advec_vol_s = idx2f(field, pre_vol, i, j) + idx2f(field, vol_flux_y, i, j) - idx2f(field, vol_flux_y, i + 0, j + 1);
				idx2f(field, density1, i, j) = post_mass_s / advec_vol_s;
				idx2f(field, energy1, i, j) = post_ener_s;
			}
		}

	}

}


//  @brief Cell centred advection driver.
//  @author Wayne Gaudin
//  @details Invokes the user selected advection kernel.
void advec_cell_driver(global_variables &globals, int tile, int sweep_number, int direction) {

	#if FLUSH_BUFFER
	globals.hostToDevice();
	#endif

	tile_type &t = globals.chunk.tiles[tile];
	advec_cell_kernel(
			globals.use_target,
			t.info.t_xmin,
			t.info.t_xmax,
			t.info.t_ymin,
			t.info.t_ymax,
			direction,
			sweep_number,
			t.field);

	#if FLUSH_BUFFER
	globals.deviceToHost();
	#endif

}

