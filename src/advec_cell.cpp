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


#define MIN(x, y) (((x) < (y)) ? (x) : (y))


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


			double *volume = field.volume.data;
			const int volume_sizex = field.volume.sizeX;
			double *vol_flux_x = field.vol_flux_x.data;
			const int vol_flux_x_sizex = field.vol_flux_x.sizeX;
			double *vol_flux_y = field.vol_flux_y.data;
			const int vol_flux_y_sizex = field.vol_flux_y.sizeX;
			double *pre_vol = field.work_array1.data;
			const int pre_vol_sizex = field.work_array1.sizeX;
			double *post_vol = field.work_array2.data;
			const int post_vol_sizex = field.work_array2.sizeX;

			#pragma omp target teams distribute parallel for simd collapse(2) if(target: use_target)
			for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
				for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
					pre_vol[i + j * pre_vol_sizex] = volume[i + j * volume_sizex] +
					                                 (vol_flux_x[(i + 1) + (j + 0) * vol_flux_x_sizex] - vol_flux_x[i + j * vol_flux_x_sizex] +
					                                  vol_flux_y[(i + 0) + (j + 1) * vol_flux_y_sizex] -
					                                  vol_flux_y[i + j * vol_flux_y_sizex]);
					post_vol[i + j * post_vol_sizex] = pre_vol[i + j * pre_vol_sizex] - (vol_flux_x[(i + 1) + (j + 0) * vol_flux_x_sizex] - vol_flux_x[i + j * vol_flux_x_sizex]);
				}
			}


		} else {


			double *volume = field.volume.data;
			const int volume_sizex = field.volume.sizeX;
			double *vol_flux_x = field.vol_flux_x.data;
			const int vol_flux_x_sizex = field.vol_flux_x.sizeX;
			double *pre_vol = field.work_array1.data;
			const int pre_vol_sizex = field.work_array1.sizeX;
			double *post_vol = field.work_array2.data;
			const int post_vol_sizex = field.work_array2.sizeX;

			#pragma omp target teams distribute parallel for simd collapse(2) if(target: use_target)
			for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
				for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
					pre_vol[i + j * pre_vol_sizex] = volume[i + j * volume_sizex] + vol_flux_x[(i + 1) + (j + 0) * vol_flux_x_sizex] - vol_flux_x[i + j * vol_flux_x_sizex];
					post_vol[i + j * post_vol_sizex] = volume[i + j * volume_sizex];
				}
			}

		}

		// DO k=y_min,y_max
		//   DO j=x_min,x_max+2
		double *vertexdx = field.vertexdx.data;
		double *density1 = field.density1.data;
		const int density1_sizex = field.density1.sizeX;
		double *energy1 = field.energy1.data;
		const int energy1_sizex = field.energy1.sizeX;
		double *mass_flux_x = field.mass_flux_x.data;
		const int mass_flux_x_sizex = field.mass_flux_x.sizeX;
		double *vol_flux_x = field.vol_flux_x.data;
		const int vol_flux_x_sizex = field.vol_flux_x.sizeX;
		double *pre_vol = field.work_array1.data;
		const int pre_vol_sizex = field.work_array1.sizeX;
		double *ener_flux = field.work_array7.data;
		const int ener_flux_sizex = field.work_array7.sizeX;

		#pragma omp target teams distribute parallel for simd collapse(2) if(target: use_target)
		for (int j = (y_min + 1); j < (y_max + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2 + 2); i++)
				({
					int upwind, donor, downwind, dif;
					double sigmat, sigma3, sigma4, sigmav, sigma, sigmam, diffuw, diffdw, limiter, wind;
					if (vol_flux_x[i + j * vol_flux_x_sizex] > 0.0) {
						upwind = i - 2;
						donor = i - 1;
						downwind = i;
						dif = donor;
					} else {
						upwind = MIN(i + 1, x_max + 2);
						donor = i;
						downwind = i - 1;
						dif = upwind;
					}
					sigmat = fabs(vol_flux_x[i + j * vol_flux_x_sizex]) / pre_vol[donor + j * pre_vol_sizex];
					sigma3 = (1.0 + sigmat) * (vertexdx[i] / vertexdx[dif]);
					sigma4 = 2.0 - sigmat;
//					sigma = sigmat;
					sigmav = sigmat;
					diffuw = density1[donor + j * density1_sizex] - density1[upwind + j * density1_sizex];
					diffdw = density1[downwind + j * density1_sizex] - density1[donor + j * density1_sizex];
					wind = 1.0;
					if (diffdw <= 0.0)wind = -1.0;
					if (diffuw * diffdw > 0.0) {
						limiter = (1.0 - sigmav) * wind *
						          fmin(fmin(
								          fabs(diffuw),
								          fabs(diffdw)),
						               one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
					} else {
						limiter = 0.0;
					}
					mass_flux_x[i + j * mass_flux_x_sizex] = vol_flux_x[i + j * vol_flux_x_sizex] * (density1[donor + j * density1_sizex] + limiter);
					sigmam = fabs(mass_flux_x[i + j * mass_flux_x_sizex]) / (density1[donor + j * density1_sizex] * pre_vol[donor + j * pre_vol_sizex]);
					diffuw = energy1[donor + j * energy1_sizex] - energy1[upwind + j * energy1_sizex];
					diffdw = energy1[downwind + j * energy1_sizex] - energy1[donor + j * energy1_sizex];
					wind = 1.0;
					if (diffdw <= 0.0)wind = -1.0;
					if (diffuw * diffdw > 0.0) {
						limiter = (1.0 - sigmam) *
						          wind *
						          fmin(fmin(
								          fabs(diffuw),
								          fabs(diffdw)),
						               one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
					} else {
						limiter = 0.0;
					}
					ener_flux[i + j * ener_flux_sizex] = mass_flux_x[i + j * mass_flux_x_sizex] * (energy1[donor + j * energy1_sizex] + limiter);
				});
		}



		// DO k=y_min,y_max
		//   DO j=x_min,x_max



		#pragma omp target teams distribute parallel for simd collapse(2) if(target: use_target)
		for (int j = (y_min + 1); j < (y_max + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2); i++) {
				double pre_mass_s = density1[i + j * density1_sizex] * pre_vol[i + j * pre_vol_sizex];
				double post_mass_s = pre_mass_s + mass_flux_x[i + j * mass_flux_x_sizex] - mass_flux_x[(i + 1) + (j + 0) * mass_flux_x_sizex];
				double post_ener_s = (energy1[i + j * energy1_sizex] * pre_mass_s + ener_flux[i + j * ener_flux_sizex] - ener_flux[(i + 1) + (j + 0) * ener_flux_sizex]) / post_mass_s;
				double advec_vol_s = pre_vol[i + j * pre_vol_sizex] + vol_flux_x[i + j * vol_flux_x_sizex] - vol_flux_x[(i + 1) + (j + 0) * vol_flux_x_sizex];
				density1[i + j * density1_sizex] = post_mass_s / advec_vol_s;
				energy1[i + j * energy1_sizex] = post_ener_s;
			}
		}

	} else if (dir == g_ydir) {

		// DO k=y_min-2,y_max+2
		//   DO j=x_min-2,x_max+2

		if (sweep_number == 1) {


			double *volume = field.volume.data;
			const int volume_sizex = field.volume.sizeX;
			double *vol_flux_x = field.vol_flux_x.data;
			const int vol_flux_x_sizex = field.vol_flux_x.sizeX;
			double *vol_flux_y = field.vol_flux_y.data;
			const int vol_flux_y_sizex = field.vol_flux_y.sizeX;
			double *pre_vol = field.work_array1.data;
			const int pre_vol_sizex = field.work_array1.sizeX;
			double *post_vol = field.work_array2.data;
			const int post_vol_sizex = field.work_array2.sizeX;

			#pragma omp target teams distribute parallel for simd collapse(2) if(target: use_target)
			for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
				for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
					pre_vol[i + j * pre_vol_sizex] = volume[i + j * volume_sizex] +
					                                 (vol_flux_y[(i + 0) + (j + 1) * vol_flux_y_sizex] - vol_flux_y[i + j * vol_flux_y_sizex] +
					                                  vol_flux_x[(i + 1) + (j + 0) * vol_flux_x_sizex] -
					                                  vol_flux_x[i + j * vol_flux_x_sizex]);
					post_vol[i + j * post_vol_sizex] = pre_vol[i + j * pre_vol_sizex] - (vol_flux_y[(i + 0) + (j + 1) * vol_flux_y_sizex] - vol_flux_y[i + j * vol_flux_y_sizex]);
				}
			}


		} else {


			double *volume = field.volume.data;
			const int volume_sizex = field.volume.sizeX;
			double *vol_flux_y = field.vol_flux_y.data;
			const int vol_flux_y_sizex = field.vol_flux_y.sizeX;
			double *pre_vol = field.work_array1.data;
			const int pre_vol_sizex = field.work_array1.sizeX;
			double *post_vol = field.work_array2.data;
			const int post_vol_sizex = field.work_array2.sizeX;

			#pragma omp target teams distribute parallel for simd collapse(2) if(target: use_target)
			for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
				for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
					pre_vol[i + j * pre_vol_sizex] = volume[i + j * volume_sizex] + vol_flux_y[(i + 0) + (j + 1) * vol_flux_y_sizex] - vol_flux_y[i + j * vol_flux_y_sizex];
					post_vol[i + j * post_vol_sizex] = volume[i + j * volume_sizex];
				}
			}


		}


		// DO k=y_min,y_max+2
		//   DO j=x_min,x_max
		double *vertexdy = field.vertexdy.data;
		double *density1 = field.density1.data;
		const int density1_sizex = field.density1.sizeX;
		double *energy1 = field.energy1.data;
		const int energy1_sizex = field.energy1.sizeX;
		double *mass_flux_y = field.mass_flux_y.data;
		const int mass_flux_y_sizex = field.mass_flux_y.sizeX;
		double *vol_flux_y = field.vol_flux_y.data;
		const int vol_flux_y_sizex = field.vol_flux_y.sizeX;
		double *pre_vol = field.work_array1.data;
		const int pre_vol_sizex = field.work_array1.sizeX;
		double *ener_flux = field.work_array7.data;
		const int ener_flux_sizex = field.work_array7.sizeX;
		#pragma omp target teams distribute parallel for simd collapse(2) if(target: use_target)
		for (int j = (y_min + 1); j < (y_max + 2 + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2); i++)
				({
					int upwind, donor, downwind, dif;
					double sigmat, sigma3, sigma4, sigmav, sigma, sigmam, diffuw, diffdw, limiter, wind;
					if (vol_flux_y[i + j * vol_flux_y_sizex] > 0.0) {
						upwind = j - 2;
						donor = j - 1;
						downwind = j;
						dif = donor;
					} else {
						upwind = MIN(j + 1, y_max + 2);
						donor = j;
						downwind = j - 1;
						dif = upwind;
					}
					sigmat = fabs(vol_flux_y[i + j * vol_flux_y_sizex]) / pre_vol[i + donor * pre_vol_sizex];
					sigma3 = (1.0 + sigmat) * (vertexdy[j] / vertexdy[dif]);
					sigma4 = 2.0 - sigmat;
//					sigma = sigmat;
					sigmav = sigmat;
					diffuw = density1[i + donor * density1_sizex] - density1[i + upwind * density1_sizex];
					diffdw = density1[i + downwind * density1_sizex] - density1[i + donor * density1_sizex];
					wind = 1.0;
					if (diffdw <= 0.0)wind = -1.0;
					if (diffuw * diffdw > 0.0) {
						limiter = (1.0 - sigmav) * wind *
						          fmin(fmin(
								          fabs(diffuw),
								          fabs(diffdw)),
						               one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
					} else {
						limiter = 0.0;
					}
					mass_flux_y[i + j * mass_flux_y_sizex] = vol_flux_y[i + j * vol_flux_y_sizex] * (density1[i + donor * density1_sizex] + limiter);
					sigmam = fabs(mass_flux_y[i + j * mass_flux_y_sizex]) / (density1[i + donor * density1_sizex] * pre_vol[i + donor * pre_vol_sizex]);
					diffuw = energy1[i + donor * energy1_sizex] - energy1[i + upwind * energy1_sizex];
					diffdw = energy1[i + downwind * energy1_sizex] - energy1[i + donor * energy1_sizex];
					wind = 1.0;
					if (diffdw <= 0.0)wind = -1.0;
					if (diffuw * diffdw > 0.0) {
						limiter = (1.0 - sigmam) * wind *
						          fmin(fmin(
								          fabs(diffuw),
								          fabs(diffdw)),
						               one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
					} else {
						limiter = 0.0;
					}
					ener_flux[i + j * ener_flux_sizex] = mass_flux_y[i + j * mass_flux_y_sizex] * (energy1[i + donor * energy1_sizex] + limiter);
				});
		}


		// DO k=y_min,y_max
		//   DO j=x_min,x_max


		#pragma omp target teams distribute parallel for simd collapse(2) if(target: use_target)
		for (int j = (y_min + 1); j < (y_max + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2); i++) {
				double pre_mass_s = density1[i + j * density1_sizex] * pre_vol[i + j * pre_vol_sizex];
				double post_mass_s = pre_mass_s + mass_flux_y[i + j * mass_flux_y_sizex] - mass_flux_y[(i + 0) + (j + 1) * mass_flux_y_sizex];
				double post_ener_s = (energy1[i + j * energy1_sizex] * pre_mass_s + ener_flux[i + j * ener_flux_sizex] - ener_flux[(i + 0) + (j + 1) * ener_flux_sizex]) / post_mass_s;
				double advec_vol_s = pre_vol[i + j * pre_vol_sizex] + vol_flux_y[i + j * vol_flux_y_sizex] - vol_flux_y[(i + 0) + (j + 1) * vol_flux_y_sizex];
				density1[i + j * density1_sizex] = post_mass_s / advec_vol_s;
				energy1[i + j * energy1_sizex] = post_ener_s;
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

