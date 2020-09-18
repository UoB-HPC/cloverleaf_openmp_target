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
#include "viscosity.h"

//  @brief Fortran viscosity kernel.
//  @author Wayne Gaudin
//  @details Calculates an artificial viscosity using the Wilkin's method to
//  smooth out shock front and prevent oscillations around discontinuities.
//  Only cells in compression will have a non-zero value.

void viscosity_kernel(
		bool use_target,
		int x_min, int x_max, int y_min, int y_max,
		clover::Buffer1D<double> &celldx,
		clover::Buffer1D<double> &celldy,
		clover::Buffer2D<double> &density0,
		clover::Buffer2D<double> &pressure,
		clover::Buffer2D<double> &viscosity,
		clover::Buffer2D<double> &xvel0,
		clover::Buffer2D<double> &yvel0) {

	// DO k=y_min,y_max
	//   DO j=x_min,x_max
	omp(parallel(2) enable_target(use_target)
			    mapToFrom1D(celldx)
			    mapToFrom1D(celldy)
			    mapToFrom2D(density0)
			    mapToFrom2D(pressure)
			    mapToFrom2D(viscosity)
			    mapToFrom2D(xvel0)
			    mapToFrom2D(yvel0)
	)
	for (int j = (y_min + 1); j < (y_max + 2); j++) {
		for (int i = (x_min + 1); i < (x_max + 2); i++) {
			double ugrad = (idx2(xvel0, i + 1, j + 0) + idx2(xvel0, i + 1, j + 1)) - (idx2(xvel0, i, j) + idx2(xvel0, i + 0, j + 1));
			double vgrad = (idx2(yvel0, i + 0, j + 1) + idx2(yvel0, i + 1, j + 1)) - (idx2(yvel0, i, j) + idx2(yvel0, i + 1, j + 0));
			double div = (idx1(celldx, i) * (ugrad) + idx1(celldy, j) * (vgrad));
			double strain2 = 0.5 * (idx2(xvel0, i + 0, j + 1) +
			                        idx2(xvel0, i + 1, j + 1) -
			                        idx2(xvel0, i, j) -
			                        idx2(xvel0, i + 1, j + 0)) / idx1(celldy, j) +
			                 0.5 * (idx2(yvel0, i + 1, j + 0) +
			                        idx2(yvel0, i + 1, j + 1) -
			                        idx2(yvel0, i, j) -
			                        idx2(yvel0, i + 0, j + 1)) / idx1(celldx, i);
			double pgradx = (idx2(pressure, i + 1, j + 0) - idx2(pressure, i - 1, j + 0)) / (idx1(celldx, i) + idx1(celldx, i + 1));
			double pgrady = (idx2(pressure, i + 0, j + 1) - idx2(pressure, i + 0, j - 1)) / (idx1(celldy, j) + idx1(celldy, j + 2));
			double pgradx2 = pgradx * pgradx;
			double pgrady2 = pgrady * pgrady;
			double limiter = ((0.5 * (ugrad) / idx1(celldx, i)) * pgradx2 +
			                  (0.5 * (vgrad) / idx1(celldy, j)) * pgrady2 + strain2 * pgradx * pgrady) /
			                 std::fmax(pgradx2 + pgrady2, g_small);
			if ((limiter > 0.0) || (div >= 0.0)) { idx2(viscosity, i, j) = 0.0; }
			else {
				double dirx = 1.0;
				if (pgradx < 0.0)dirx = -1.0;
				pgradx = dirx * std::fmax(g_small, std::fabs(pgradx));
				double diry = 1.0;
				if (pgradx < 0.0)diry = -1.0;
				pgrady = diry * std::fmax(g_small, std::fabs(pgrady));
				double pgrad = std::sqrt(pgradx * pgradx + pgrady * pgrady);
				double xgrad = std::fabs(idx1(celldx, i) * pgrad / pgradx);
				double ygrad = std::fabs(idx1(celldy, j) * pgrad / pgrady);
				double grad = std::fmin(xgrad, ygrad);
				double grad2 = grad * grad;
				idx2(viscosity, i, j) = 2.0 * idx2(density0, i, j) * grad2 * limiter * limiter;
			}
		}
	}
}

//  @brief Driver for the viscosity kernels
//  @author Wayne Gaudin
//  @details Selects the user specified kernel to caluclate the artificial
//  viscosity.
void viscosity(global_variables &globals) {

	#if FLUSH_BUFFER
	globals.hostToDevice();
	#endif


	for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
		tile_type &t = globals.chunk.tiles[tile];
		viscosity_kernel(globals.use_target,
		                 t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax,
		                 t.field.celldx, t.field.celldy, t.field.density0,
		                 t.field.pressure, t.field.viscosity, t.field.xvel0,
		                 t.field.yvel0);
	}

	#if FLUSH_BUFFER
	globals.deviceToHost();
	#endif

}
