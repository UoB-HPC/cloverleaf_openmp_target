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

		mapToFrom2Df(field, xarea)
		mapToFrom2Df(field, yarea)
		mapToFrom2Df(field, volume)
		mapToFrom2Df(field, density0)
		mapToFrom2Df(field, density1)
		mapToFrom2Df(field, energy0)
		mapToFrom2Df(field, energy1)
		mapToFrom2Df(field, pressure)
		mapToFrom2Df(field, viscosity)
		mapToFrom2Df(field, xvel0)
		mapToFrom2Df(field, xvel1)
		mapToFrom2Df(field, yvel0)
		mapToFrom2Df(field, yvel1)
		mapToFrom2Dfn(field, work_array1, volume_change)

		omp(parallel(2) enable_target(use_target))
		for (int j = (y_min + 1); j < (y_max + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2); i++) {
				double left_flux = (idx2f(field, xarea, i, j) * (idx2f(field, xvel0, i, j) +
				                                                 idx2f(field, xvel0, i + 0, j + 1) +
				                                                 idx2f(field, xvel0, i, j) +
				                                                 idx2f(field, xvel0, i + 0, j + 1))) * 0.25 * dt * 0.5;
				double right_flux = (idx2f(field, xarea, i + 1, j + 0) * (idx2f(field, xvel0, i + 1, j + 0) +
				                                                          idx2f(field, xvel0, i + 1, j + 1) +
				                                                          idx2f(field, xvel0, i + 1, j + 0) +
				                                                          idx2f(field, xvel0, i + 1, j + 1))) * 0.25 * dt * 0.5;
				double bottom_flux = (idx2f(field, yarea, i, j) * (idx2f(field, yvel0, i, j) +
				                                                   idx2f(field, yvel0, i + 1, j + 0) +
				                                                   idx2f(field, yvel0, i, j) +
				                                                   idx2f(field, yvel0, i + 1, j + 0))) * 0.25 * dt * 0.5;
				double top_flux = (idx2f(field, yarea, i + 0, j + 1) * (idx2f(field, yvel0, i + 0, j + 1) +
				                                                        idx2f(field, yvel0, i + 1, j + 1) +
				                                                        idx2f(field, yvel0, i + 0, j + 1) +
				                                                        idx2f(field, yvel0, i + 1, j + 1))) * 0.25 * dt * 0.5;
				double total_flux = right_flux - left_flux + top_flux - bottom_flux;
				double volume_change_s = idx2f(field, volume, i, j) / (idx2f(field, volume, i, j) + total_flux);
				double min_cell_volume = std::fmin(std::fmin(idx2f(field, volume, i, j) + right_flux - left_flux + top_flux - bottom_flux, idx2f(field, volume, i, j) + right_flux - left_flux),
				                                   idx2f(field, volume, i, j) + top_flux - bottom_flux);
				double recip_volume = 1.0 / idx2f(field, volume, i, j);
				double energy_change = (idx2f(field, pressure, i, j) / idx2f(field, density0, i, j) + idx2f(field, viscosity, i, j) / idx2f(field, density0, i, j)) * total_flux * recip_volume;
				idx2f(field, energy1, i, j) = idx2f(field, energy0, i, j) - energy_change;
				idx2f(field, density1, i, j) = idx2f(field, density0, i, j) * volume_change_s;
			}
		}

	} else {

		mapToFrom2Df(field, xarea)
		mapToFrom2Df(field, yarea)
		mapToFrom2Df(field, volume)
		mapToFrom2Df(field, density0)
		mapToFrom2Df(field, density1)
		mapToFrom2Df(field, energy0)
		mapToFrom2Df(field, energy1)
		mapToFrom2Df(field, pressure)
		mapToFrom2Df(field, viscosity)
		mapToFrom2Df(field, xvel0)
		mapToFrom2Df(field, xvel1)
		mapToFrom2Df(field, yvel0)
		mapToFrom2Df(field, yvel1)
		mapToFrom2Dfn(field, work_array1, volume_change)

		omp(parallel(2) enable_target(use_target))
		for (int j = (y_min + 1); j < (y_max + 2); j++) {
			for (int i = (x_min + 1); i < (x_max + 2); i++) {
				double left_flux = (idx2f(field, xarea, i, j) * (idx2f(field, xvel0, i, j) +
				                                                 idx2f(field, xvel0, i + 0, j + 1) +
				                                                 idx2f(field, xvel1, i, j) +
				                                                 idx2f(field, xvel1, i + 0, j + 1))) * 0.25 * dt;
				double right_flux = (idx2f(field, xarea, i + 1, j + 0) * (idx2f(field, xvel0, i + 1, j + 0) +
				                                                          idx2f(field, xvel0, i + 1, j + 1) +
				                                                          idx2f(field, xvel1, i + 1, j + 0) +
				                                                          idx2f(field, xvel1, i + 1, j + 1))) * 0.25 * dt;
				double bottom_flux = (idx2f(field, yarea, i, j) * (idx2f(field, yvel0, i, j) +
				                                                   idx2f(field, yvel0, i + 1, j + 0) +
				                                                   idx2f(field, yvel1, i, j) +
				                                                   idx2f(field, yvel1, i + 1, j + 0))) * 0.25 * dt;
				double top_flux = (idx2f(field, yarea, i + 0, j + 1) * (idx2f(field, yvel0, i + 0, j + 1) +
				                                                        idx2f(field, yvel0, i + 1, j + 1) +
				                                                        idx2f(field, yvel1, i + 0, j + 1) + idx2f(field, yvel1, i + 1, j + 1))) * 0.25 * dt;
				double total_flux = right_flux - left_flux + top_flux - bottom_flux;
				double volume_change_s = idx2f(field, volume, i, j) / (idx2f(field, volume, i, j) + total_flux);
				double min_cell_volume = std::fmin(std::fmin(
						idx2f(field, volume, i, j) + right_flux - left_flux + top_flux - bottom_flux, idx2f(field, volume, i, j) + right_flux - left_flux),
				                                   idx2f(field, volume, i, j) + top_flux - bottom_flux);
				double recip_volume = 1.0 / idx2f(field, volume, i, j);
				double energy_change = (idx2f(field, pressure, i, j) / idx2f(field, density0, i, j) + idx2f(field, viscosity, i, j) / idx2f(field, density0, i, j)) * total_flux * recip_volume;
				idx2f(field, energy1, i, j) = idx2f(field, energy0, i, j) - energy_change;
				idx2f(field, density1, i, j) = idx2f(field, density0, i, j) * volume_change_s;
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


