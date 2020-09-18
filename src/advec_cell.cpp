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
		clover::Buffer1D<double> &vertexdx,
		clover::Buffer1D<double> &vertexdy,
		clover::Buffer2D<double> &volume,
		clover::Buffer2D<double> &density1,
		clover::Buffer2D<double> &energy1,
		clover::Buffer2D<double> &mass_flux_x,
		clover::Buffer2D<double> &vol_flux_x,
		clover::Buffer2D<double> &mass_flux_y,
		clover::Buffer2D<double> &vol_flux_y,
		clover::Buffer2D<double> &pre_vol,
		clover::Buffer2D<double> &post_vol,
		clover::Buffer2D<double> &pre_mass,
		clover::Buffer2D<double> &post_mass,
		clover::Buffer2D<double> &advec_vol,
		clover::Buffer2D<double> &post_ener,
		clover::Buffer2D<double> &ener_flux) {

	const double one_by_six = 1.0 / 6.0;

	if (dir == g_xdir) {

		// DO k=y_min-2,y_max+2
		//   DO j=x_min-2,x_max+2

		if (sweep_number == 1) {


			omp(parallel(2) enable_target(use_target)
					    mapToFrom2D(volume)
					    mapToFrom2D(vol_flux_x)
					    mapToFrom2D(vol_flux_y)
					    mapToFrom2D(pre_vol)
					    mapToFrom2D(post_vol)
			)
			for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
				for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
					idx2(pre_vol, i, j) = idx2(volume, i, j) + (idx2(vol_flux_x, i + 1, j + 0) - idx2(vol_flux_x, i, j) + idx2(vol_flux_y, i + 0, j + 1) - idx2(vol_flux_y, i, j));
					idx2(post_vol, i, j) = idx2(pre_vol, i, j) - (idx2(vol_flux_x, i + 1, j + 0) - idx2(vol_flux_x, i, j));
				}
			}


		} else {


			omp(parallel(2) enable_target(use_target)
					    mapToFrom2D(volume)
					    mapToFrom2D(vol_flux_x)
					    mapToFrom2D(pre_vol)
					    mapToFrom2D(post_vol)
			)
			for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
				for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
					idx2(pre_vol, i, j) = idx2(volume, i, j) + idx2(vol_flux_x, i + 1, j + 0) - idx2(vol_flux_x, i, j);
					idx2(post_vol, i, j) = idx2(volume, i, j);
				}
			}

		}

		// DO k=y_min,y_max
		//   DO j=x_min,x_max+2
		omp(parallel(2) enable_target(use_target)
				    mapToFrom1D(vertexdx)
				    mapToFrom2D(density1)
				    mapToFrom2D(energy1)
				    mapToFrom2D(mass_flux_x)
				    mapToFrom2D(vol_flux_x)
				    mapToFrom2D(pre_vol)
				    mapToFrom2D(ener_flux)
		)
		for (int j = (y_min + 1); j < (y_max + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2 + 2); i++)
				({
					int upwind, donor, downwind, dif;
					double sigmat, sigma3, sigma4, sigmav, sigma, sigmam, diffuw, diffdw, limiter, wind;
					if (idx2(vol_flux_x, i, j) > 0.0) {
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
					sigmat = std::fabs(idx2(vol_flux_x, i, j)) / idx2(pre_vol, donor, j);
					sigma3 = (1.0 + sigmat) * (idx1(vertexdx, i) / idx1(vertexdx, dif));
					sigma4 = 2.0 - sigmat;
//					sigma = sigmat;
					sigmav = sigmat;
					diffuw = idx2(density1, donor, j) - idx2(density1, upwind, j);
					diffdw = idx2(density1, downwind, j) - idx2(density1, donor, j);
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
					idx2(mass_flux_x, i, j) = idx2(vol_flux_x, i, j) * (idx2(density1, donor, j) + limiter);
					sigmam = std::fabs(idx2(mass_flux_x, i, j)) / (idx2(density1, donor, j) * idx2(pre_vol, donor, j));
					diffuw = idx2(energy1, donor, j) - idx2(energy1, upwind, j);
					diffdw = idx2(energy1, downwind, j) - idx2(energy1, donor, j);
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
					idx2(ener_flux, i, j) = idx2(mass_flux_x, i, j) * (idx2(energy1, donor, j) + limiter);
				});
		}



		// DO k=y_min,y_max
		//   DO j=x_min,x_max

		omp(parallel(2) enable_target(use_target)
				    mapToFrom2D(density1)
				    mapToFrom2D(energy1)
				    mapToFrom2D(mass_flux_x)
				    mapToFrom2D(vol_flux_x)
				    mapToFrom2D(pre_vol)
				    mapToFrom2D(ener_flux)
		)
		for (int j = (y_min + 1); j < (y_max + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2); i++) {
				double pre_mass_s = idx2(density1, i, j) * idx2(pre_vol, i, j);
				double post_mass_s = pre_mass_s + idx2(mass_flux_x, i, j) - idx2(mass_flux_x, i + 1, j + 0);
				double post_ener_s = (idx2(energy1, i, j) * pre_mass_s + idx2(ener_flux, i, j) - idx2(ener_flux, i + 1, j + 0)) / post_mass_s;
				double advec_vol_s = idx2(pre_vol, i, j) + idx2(vol_flux_x, i, j) - idx2(vol_flux_x, i + 1, j + 0);
				idx2(density1, i, j) = post_mass_s / advec_vol_s;
				idx2(energy1, i, j) = post_ener_s;
			}
		}

	} else if (dir == g_ydir) {

		// DO k=y_min-2,y_max+2
		//   DO j=x_min-2,x_max+2

		if (sweep_number == 1) {


			omp(parallel(2) enable_target(use_target)
					    mapToFrom2D(volume)
					    mapToFrom2D(vol_flux_x)
					    mapToFrom2D(vol_flux_y)
					    mapToFrom2D(pre_vol)
					    mapToFrom2D(post_vol)
			)
			for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
				for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
					idx2(pre_vol, i, j) = idx2(volume, i, j) + (idx2(vol_flux_y, i + 0, j + 1) - idx2(vol_flux_y, i, j) + idx2(vol_flux_x, i + 1, j + 0) - idx2(vol_flux_x, i, j));
					idx2(post_vol, i, j) = idx2(pre_vol, i, j) - (idx2(vol_flux_y, i + 0, j + 1) - idx2(vol_flux_y, i, j));
				}
			}


		} else {


			omp(parallel(2) enable_target(use_target)
					    mapToFrom2D(volume)
					    mapToFrom2D(vol_flux_y)
					    mapToFrom2D(pre_vol)
					    mapToFrom2D(post_vol)
			)
			for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
				for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
					idx2(pre_vol, i, j) = idx2(volume, i, j) + idx2(vol_flux_y, i + 0, j + 1) - idx2(vol_flux_y, i, j);
					idx2(post_vol, i, j) = idx2(volume, i, j);
				}
			}


		}


		// DO k=y_min,y_max+2
		//   DO j=x_min,x_max
		omp(parallel(2) enable_target(use_target)
				    mapToFrom1D(vertexdy)
				    mapToFrom2D(density1)
				    mapToFrom2D(energy1)
				    mapToFrom2D(mass_flux_y)
				    mapToFrom2D(vol_flux_y)
				    mapToFrom2D(pre_vol)
				    mapToFrom2D(ener_flux)
		)
		for (int j = (y_min + 1); j < (y_max + 2 + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2); i++)
				({
					int upwind, donor, downwind, dif;
					double sigmat, sigma3, sigma4, sigmav, sigma, sigmam, diffuw, diffdw, limiter, wind;
					if (idx2(vol_flux_y, i, j) > 0.0) {
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
					sigmat = std::fabs(idx2(vol_flux_y, i, j)) / idx2(pre_vol, i, donor);
					sigma3 = (1.0 + sigmat) * (idx1(vertexdy, j) / idx1(vertexdy, dif));
					sigma4 = 2.0 - sigmat;
//					sigma = sigmat;
					sigmav = sigmat;
					diffuw = idx2(density1, i, donor) - idx2(density1, i, upwind);
					diffdw = idx2(density1, i, downwind) - idx2(density1, i, donor);
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
					idx2(mass_flux_y, i, j) = idx2(vol_flux_y, i, j) * (idx2(density1, i, donor) + limiter);
					sigmam = std::fabs(idx2(mass_flux_y, i, j)) / (idx2(density1, i, donor) * idx2(pre_vol, i, donor));
					diffuw = idx2(energy1, i, donor) - idx2(energy1, i, upwind);
					diffdw = idx2(energy1, i, downwind) - idx2(energy1, i, donor);
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
					idx2(ener_flux, i, j) = idx2(mass_flux_y, i, j) * (idx2(energy1, i, donor) + limiter);
				});
		}


		// DO k=y_min,y_max
		//   DO j=x_min,x_max
		omp(parallel(2) enable_target(use_target)
				    mapToFrom2D(density1)
				    mapToFrom2D(energy1)
				    mapToFrom2D(mass_flux_y)
				    mapToFrom2D(vol_flux_y)
				    mapToFrom2D(pre_vol)
				    mapToFrom2D(ener_flux)
		)
		for (int j = (y_min + 1); j < (y_max + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2); i++) {
				double pre_mass_s = idx2(density1, i, j) * idx2(pre_vol, i, j);
				double post_mass_s = pre_mass_s + idx2(mass_flux_y, i, j) - idx2(mass_flux_y, i + 0, j + 1);
				double post_ener_s = (idx2(energy1, i, j) * pre_mass_s + idx2(ener_flux, i, j) - idx2(ener_flux, i + 0, j + 1)) / post_mass_s;
				double advec_vol_s = idx2(pre_vol, i, j) + idx2(vol_flux_y, i, j) - idx2(vol_flux_y, i + 0, j + 1);
				idx2(density1, i, j) = post_mass_s / advec_vol_s;
				idx2(energy1, i, j) = post_ener_s;
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
			t.field.vertexdx,
			t.field.vertexdy,
			t.field.volume,
			t.field.density1,
			t.field.energy1,
			t.field.mass_flux_x,
			t.field.vol_flux_x,
			t.field.mass_flux_y,
			t.field.vol_flux_y,
			t.field.work_array1,
			t.field.work_array2,
			t.field.work_array3,
			t.field.work_array4,
			t.field.work_array5,
			t.field.work_array6,
			t.field.work_array7);

	#if FLUSH_BUFFER
	globals.deviceToHost();
	#endif

}

