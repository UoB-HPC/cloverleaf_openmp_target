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
#include "PdV.h"
#include "timer.h"
#include "comms.h"
#include "report.h"
#include "ideal_gas.h"
#include "update_halo.h"
#include "revert.h"


//  @brief Fortran PdV kernel.
//  @author Wayne Gaudin
//  @details Calculates the change in energy and density in a cell using the
//  change on cell volume due to the velocity gradients in a cell. The time
//  level of the velocity data depends on whether it is invoked as the
//  predictor or corrector.
void PdV_kernel(
		bool use_target,
		bool predict,
		int x_min, int x_max, int y_min, int y_max,
		double dt,
		field_type &field
) {


	// DO k=y_min,y_max
	//   DO j=x_min,x_max

	if (predict) {

		double *xarea = field.xarea.data;
		const int xarea_sizex = field.xarea.sizeX;
		double *yarea = field.yarea.data;
		const int yarea_sizex = field.yarea.sizeX;
		double *volume = field.volume.data;
		const int volume_sizex = field.volume.sizeX;
		double *density0 = field.density0.data;
		const int density0_sizex = field.density0.sizeX;
		double *density1 = field.density1.data;
		const int density1_sizex = field.density1.sizeX;
		double *energy0 = field.energy0.data;
		const int energy0_sizex = field.energy0.sizeX;
		double *energy1 = field.energy1.data;
		const int energy1_sizex = field.energy1.sizeX;
		double *pressure = field.pressure.data;
		const int pressure_sizex = field.pressure.sizeX;
		double *viscosity = field.viscosity.data;
		const int viscosity_sizex = field.viscosity.sizeX;
		double *xvel0 = field.xvel0.data;
		const int xvel0_sizex = field.xvel0.sizeX;
		double *xvel1 = field.xvel1.data;
		const int xvel1_sizex = field.xvel1.sizeX;
		double *yvel0 = field.yvel0.data;
		const int yvel0_sizex = field.yvel0.sizeX;
		double *yvel1 = field.yvel1.data;
		const int yvel1_sizex = field.yvel1.sizeX;
		double *volume_change = field.work_array1.data;
		const int volume_change_sizex = field.work_array1.sizeX;

		#pragma omp target teams distribute parallel for simd collapse(2) omp_use_target(use_target)
		for (int j = (y_min + 1); j < (y_max + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2); i++) {
				double left_flux = (xarea[i + j * xarea_sizex] * (xvel0[i + j * xvel0_sizex] +
				                                                  xvel0[(i + 0) + (j + 1) * xvel0_sizex] +
				                                                  xvel0[i + j * xvel0_sizex] +
				                                                  xvel0[(i + 0) + (j + 1) * xvel0_sizex])) * 0.25 * dt * 0.5;
				double right_flux = (xarea[(i + 1) + (j + 0) * xarea_sizex] * (xvel0[(i + 1) + (j + 0) * xvel0_sizex] +
				                                                               xvel0[(i + 1) + (j + 1) * xvel0_sizex] +
				                                                               xvel0[(i + 1) + (j + 0) * xvel0_sizex] +
				                                                               xvel0[(i + 1) + (j + 1) * xvel0_sizex])) * 0.25 * dt * 0.5;
				double bottom_flux = (yarea[i + j * yarea_sizex] * (yvel0[i + j * yvel0_sizex] +
				                                                    yvel0[(i + 1) + (j + 0) * yvel0_sizex] +
				                                                    yvel0[i + j * yvel0_sizex] +
				                                                    yvel0[(i + 1) + (j + 0) * yvel0_sizex])) * 0.25 * dt * 0.5;
				double top_flux = (yarea[(i + 0) + (j + 1) * yarea_sizex] * (yvel0[(i + 0) + (j + 1) * yvel0_sizex] +
				                                                             yvel0[(i + 1) + (j + 1) * yvel0_sizex] +
				                                                             yvel0[(i + 0) + (j + 1) * yvel0_sizex] +
				                                                             yvel0[(i + 1) + (j + 1) * yvel0_sizex])) * 0.25 * dt * 0.5;
				double total_flux = right_flux - left_flux + top_flux - bottom_flux;
				double volume_change_s = volume[i + j * volume_sizex] / (volume[i + j * volume_sizex] + total_flux);
				double min_cell_volume = fmin(fmin(volume[i + j * volume_sizex] + right_flux - left_flux + top_flux - bottom_flux, volume[i + j * volume_sizex] + right_flux - left_flux),
				                              volume[i + j * volume_sizex] + top_flux - bottom_flux);
				double recip_volume = 1.0 / volume[i + j * volume_sizex];
				double energy_change =
						(pressure[i + j * pressure_sizex] / density0[i + j * density0_sizex] + viscosity[i + j * viscosity_sizex] / density0[i + j * density0_sizex]) * total_flux *
						recip_volume;
				energy1[i + j * energy1_sizex] = energy0[i + j * energy0_sizex] - energy_change;
				density1[i + j * density1_sizex] = density0[i + j * density0_sizex] * volume_change_s;
			}
		}

	} else {

		double *xarea = field.xarea.data;
		const int xarea_sizex = field.xarea.sizeX;
		double *yarea = field.yarea.data;
		const int yarea_sizex = field.yarea.sizeX;
		double *volume = field.volume.data;
		const int volume_sizex = field.volume.sizeX;
		double *density0 = field.density0.data;
		const int density0_sizex = field.density0.sizeX;
		double *density1 = field.density1.data;
		const int density1_sizex = field.density1.sizeX;
		double *energy0 = field.energy0.data;
		const int energy0_sizex = field.energy0.sizeX;
		double *energy1 = field.energy1.data;
		const int energy1_sizex = field.energy1.sizeX;
		double *pressure = field.pressure.data;
		const int pressure_sizex = field.pressure.sizeX;
		double *viscosity = field.viscosity.data;
		const int viscosity_sizex = field.viscosity.sizeX;
		double *xvel0 = field.xvel0.data;
		const int xvel0_sizex = field.xvel0.sizeX;
		double *xvel1 = field.xvel1.data;
		const int xvel1_sizex = field.xvel1.sizeX;
		double *yvel0 = field.yvel0.data;
		const int yvel0_sizex = field.yvel0.sizeX;
		double *yvel1 = field.yvel1.data;
		const int yvel1_sizex = field.yvel1.sizeX;
		double *volume_change = field.work_array1.data;
		const int volume_change_sizex = field.work_array1.sizeX;

		#pragma omp target teams distribute parallel for simd collapse(2) omp_use_target(use_target)
		for (int j = (y_min + 1); j < (y_max + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2); i++) {
				double left_flux = (xarea[i + j * xarea_sizex] * (xvel0[i + j * xvel0_sizex] +
				                                                  xvel0[(i + 0) + (j + 1) * xvel0_sizex] +
				                                                  xvel1[i + j * xvel1_sizex] +
				                                                  xvel1[(i + 0) + (j + 1) * xvel1_sizex])) * 0.25 * dt;
				double right_flux = (xarea[(i + 1) + (j + 0) * xarea_sizex] * (xvel0[(i + 1) + (j + 0) * xvel0_sizex] +
				                                                               xvel0[(i + 1) + (j + 1) * xvel0_sizex] +
				                                                               xvel1[(i + 1) + (j + 0) * xvel1_sizex] +
				                                                               xvel1[(i + 1) + (j + 1) * xvel1_sizex])) * 0.25 * dt;
				double bottom_flux = (yarea[i + j * yarea_sizex] * (yvel0[i + j * yvel0_sizex] +
				                                                    yvel0[(i + 1) + (j + 0) * yvel0_sizex] +
				                                                    yvel1[i + j * yvel1_sizex] +
				                                                    yvel1[(i + 1) + (j + 0) * yvel1_sizex])) * 0.25 * dt;
				double top_flux = (yarea[(i + 0) + (j + 1) * yarea_sizex] * (yvel0[(i + 0) + (j + 1) * yvel0_sizex] +
				                                                             yvel0[(i + 1) + (j + 1) * yvel0_sizex] +
				                                                             yvel1[(i + 0) + (j + 1) * yvel1_sizex] + yvel1[(i + 1) + (j + 1) * yvel1_sizex])) * 0.25 * dt;
				double total_flux = right_flux - left_flux + top_flux - bottom_flux;
				double volume_change_s = volume[i + j * volume_sizex] / (volume[i + j * volume_sizex] + total_flux);
				double min_cell_volume = fmin(fmin(
						volume[i + j * volume_sizex] + right_flux - left_flux + top_flux - bottom_flux, volume[i + j * volume_sizex] + right_flux - left_flux),
				                              volume[i + j * volume_sizex] + top_flux - bottom_flux);
				double recip_volume = 1.0 / volume[i + j * volume_sizex];
				double energy_change =
						(pressure[i + j * pressure_sizex] / density0[i + j * density0_sizex] + viscosity[i + j * viscosity_sizex] / density0[i + j * density0_sizex]) * total_flux *
						recip_volume;
				energy1[i + j * energy1_sizex] = energy0[i + j * energy0_sizex] - energy_change;
				density1[i + j * density1_sizex] = density0[i + j * density0_sizex] * volume_change_s;
			}
		}
	}

}


//  @brief Driver for the PdV update.
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the PdV update.
void PdV(global_variables &globals, bool predict) {

	double kernel_time = 0;
	if (globals.profiler_on) kernel_time = timer();

	globals.error_condition = 0;

	#if FLUSH_BUFFER
	globals.hostToDevice();
	#endif

	for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
		tile_type &t = globals.chunk.tiles[tile];
		PdV_kernel(globals.use_target,
		           predict,
		           t.info.t_xmin,
		           t.info.t_xmax,
		           t.info.t_ymin,
		           t.info.t_ymax,
		           globals.dt,
		           t.field);
	}

	#if FLUSH_BUFFER
	globals.deviceToHost();
	#endif

	clover_check_error(globals.error_condition);
	if (globals.profiler_on) globals.profiler.PdV += timer() - kernel_time;

	if (globals.error_condition == 1) {
		report_error((char *) "PdV", (char *) "error in PdV");
	}

	if (predict) {
		if (globals.profiler_on) kernel_time = timer();
		for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
			ideal_gas(globals, tile, true);
		}

		if (globals.profiler_on) globals.profiler.ideal_gas += timer() - kernel_time;

		int fields[NUM_FIELDS];
		for (int &field : fields) field = 0;
		fields[field_pressure] = 1;
		update_halo(globals, fields, 1);
	}

	if (predict) {
		if (globals.profiler_on) kernel_time = timer();
		revert(globals);
		if (globals.profiler_on) globals.profiler.revert += timer() - kernel_time;
	}

}


