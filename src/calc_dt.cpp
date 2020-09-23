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



#include <string>
#include "calc_dt.h"

#include <cmath>

//  @brief Fortran timestep kernel
//  @author Wayne Gaudin
//  @details Calculates the minimum timestep on the mesh chunk based on the CFL
//  condition, the velocity gradient and the velocity divergence. A safety
//  factor is used to ensure numerical stability.


void calc_dt_kernel(
		bool use_target,
		int x_min, int x_max, int y_min, int y_max,
		double dtmin,
		double dtc_safe,
		double dtu_safe,
		double dtv_safe,
		double dtdiv_safe,
		field_type &field,
		double &dt_min_val,
		int &dtl_control,
		double &xl_pos,
		double &yl_pos,
		int &jldt,
		int &kldt,
		int &small) {


	small = 0;
	dt_min_val = g_big;
	double jk_control = 1.1;

	// DO k=y_min,y_max
	//   DO j=x_min,x_max
//	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});


	double *xarea = field.xarea.data;
	const int xarea_sizex = field.xarea.sizeX;
	double *yarea = field.yarea.data;
	const int yarea_sizex = field.yarea.sizeX;
	double *celldx = field.celldx.data;
	double *celldy = field.celldy.data;
	double *volume = field.volume.data;
	const int volume_sizex = field.volume.sizeX;
	double *density0 = field.density0.data;
	const int density0_sizex = field.density0.sizeX;
	double *viscosity = field.viscosity.data;
	const int viscosity_sizex = field.viscosity.sizeX;
	double *soundspeed = field.soundspeed.data;
	const int soundspeed_sizex = field.soundspeed.sizeX;
	double *xvel0 = field.xvel0.data;
	const int xvel0_sizex = field.xvel0.sizeX;
	double *yvel0 = field.yvel0.data;
	const int yvel0_sizex = field.yvel0.sizeX;


	#pragma omp target teams distribute parallel for simd collapse(2) if(target: use_target) map(tofrom:dt_min_val) reduction(min:dt_min_val)
	for (int j = (y_min + 1); j < (y_max + 2); j++) {
		for (int i = (x_min + 1); i < (x_max + 2); i++) {
			double dsx = celldx[i];
			double dsy = celldy[j];
			double cc = soundspeed[i + j * soundspeed_sizex] * soundspeed[i + j * soundspeed_sizex];
			cc = cc + 2.0 * viscosity[i + j * viscosity_sizex] / density0[i + j * density0_sizex];
			cc = fmax(sqrt(cc), g_small);
			double dtct = dtc_safe * fmin(dsx, dsy) / cc;
			double div = 0.0;
			double dv1 = (xvel0[i + j * xvel0_sizex] + xvel0[(i + 0) + (j + 1) * xvel0_sizex]) * xarea[i + j * xarea_sizex];
			double dv2 = (xvel0[(i + 1) + (j + 0) * xvel0_sizex] + xvel0[(i + 1) + (j + 1) * xvel0_sizex]) * xarea[(i + 1) + (j + 0) * xarea_sizex];
			div = div + dv2 - dv1;
			double dtut = dtu_safe * 2.0 * volume[i + j * volume_sizex] / fmax(fmax(fabs(dv1), fabs(dv2)), g_small * volume[i + j * volume_sizex]);
			dv1 = (yvel0[i + j * yvel0_sizex] + yvel0[(i + 1) + (j + 0) * yvel0_sizex]) * yarea[i + j * yarea_sizex];
			dv2 = (yvel0[(i + 0) + (j + 1) * yvel0_sizex] + yvel0[(i + 1) + (j + 1) * yvel0_sizex]) * yarea[(i + 0) + (j + 1) * yarea_sizex];
			div = div + dv2 - dv1;
			double dtvt = dtv_safe * 2.0 * volume[i + j * volume_sizex] / fmax(fmax(fabs(dv1), fabs(dv2)), g_small * volume[i + j * volume_sizex]);
			div = div / (2.0 * volume[i + j * volume_sizex]);
			double dtdivt;
			if (div < -g_small) {
				dtdivt = dtdiv_safe * (-1.0 / div);
			} else {
				dtdivt = g_big;
			}
			double mins = fmin(dtct, fmin(dtut, fmin(dtvt, fmin(dtdivt, g_big))));
			dt_min_val = fmin(mins, dt_min_val);
		}
	}


	dtl_control = static_cast<int>(10.01 * (jk_control - static_cast<int>(jk_control)));
	jk_control = jk_control - (jk_control - (int) (jk_control));
	jldt = ((int) jk_control) % x_max;
	kldt = static_cast<int>(1.f + (jk_control / x_max));

	if (dt_min_val < dtmin) small = 1;


	if (small != 0) {

		auto &cellx_acc = field.cellx;
		auto &celly_acc = field.celly;
		auto &density0_acc = field.density0;
		auto &energy0_acc = field.energy0;
		auto &pressure_acc = field.pressure;
		auto &soundspeed_acc = field.soundspeed;
		auto &xvel0_acc = field.xvel0;
		auto &yvel0_acc = field.yvel0;

		std::cout
				<< "Timestep information:" << std::endl
				<< "j, k                 : " << jldt << " " << kldt << std::endl
				<< "x, y                 : " << hostidx1(cellx_acc, jldt) << " " << hostidx1(celly_acc, kldt)
				<< std::endl
				<< "timestep : " << dt_min_val << std::endl
				<< "Cell velocities;" << std::endl
				<< hostidx2(xvel0_acc, jldt, kldt) << " " << hostidx2(yvel0_acc, jldt, kldt) << std::endl
				<< hostidx2(xvel0_acc, jldt + 1, kldt) << " " << hostidx2(yvel0_acc, jldt + 1, kldt) << std::endl
				<< hostidx2(xvel0_acc, jldt + 1, kldt + 1) << " " << hostidx2(yvel0_acc, jldt + 1, kldt + 1)
				<< std::endl
				<< hostidx2(xvel0_acc, jldt, kldt + 1) << " " << hostidx2(yvel0_acc, jldt, kldt + 1) << std::endl
				<< "density, energy, pressure, soundspeed " << std::endl
				<< hostidx2(density0_acc, jldt, kldt) << " " << hostidx2(energy0_acc, jldt, kldt) << " "
				<< hostidx2(pressure_acc, jldt, kldt)
				<< " " << hostidx2(soundspeed_acc, jldt, kldt) << std::endl;
	}


}


//  @brief Driver for the timestep kernels
//  @author Wayne Gaudin
//  @details Invokes the user specified timestep kernel.
void calc_dt(global_variables &globals, int tile, double &local_dt, std::string &local_control,
             double &xl_pos, double &yl_pos, int &jldt, int &kldt) {

	local_dt = g_big;

	int l_control;
	int small = 0;

	#if FLUSH_BUFFER
	globals.hostToDevice();
	#endif

	tile_type &t = globals.chunk.tiles[tile];
	calc_dt_kernel(
			globals.use_target,
			t.info.t_xmin,
			t.info.t_xmax,
			t.info.t_ymin,
			t.info.t_ymax,
			globals.config.dtmin,
			globals.config.dtc_safe,
			globals.config.dtu_safe,
			globals.config.dtv_safe,
			globals.config.dtdiv_safe,
			t.field,
			local_dt,
			l_control,
			xl_pos,
			yl_pos,
			jldt,
			kldt,
			small
	);

	#if FLUSH_BUFFER
	globals.deviceToHost();
	#endif


	if (l_control == 1) local_control = "sound";
	if (l_control == 2) local_control = "xvel";
	if (l_control == 3) local_control = "yvel";
	if (l_control == 4) local_control = "div";

}

