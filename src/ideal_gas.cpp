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
#include "ideal_gas.h"

#include "comms.h"

#define IDX(buffer, x, y) buffer[idx[(x)]][idx[(y)]]

#define SWP(x, y) (x), (y)

int N = 0;

//  @brief Fortran ideal gas kernel.
//  @author Wayne Gaudin
//  @details Calculates the pressure and sound speed for the mesh chunk using
//  the ideal gas equation of state, with a fixed gamma of 1.4.
void ideal_gas_kernel(
		bool use_target,
		int x_min, int x_max, int y_min, int y_max,
		clover::Buffer2D<double> &density_buffer,
		clover::Buffer2D<double> &energy_buffer,
		clover::Buffer2D<double> &pressure_buffer,
		clover::Buffer2D<double> &soundspeed_buffer) {

	//std::cout <<" ideal_gas(" << x_min+1 << ","<< y_min+1<< ","<< x_max+2<< ","<< y_max +2  << ")" << std::endl;
	// DO k=y_min,y_max
	//   DO j=x_min,x_max

//	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});

	double *density = density_buffer.data;
	const int density_sizex = density_buffer.sizeX;
	double *energy = energy_buffer.data;
	const int energy_sizex = energy_buffer.sizeX;
	double *pressure = pressure_buffer.data;
	const int pressure_sizex = pressure_buffer.sizeX;
	double *soundspeed = soundspeed_buffer.data;
	const int soundspeed_sizex = soundspeed_buffer.sizeX;

	#pragma omp target teams distribute parallel for simd collapse(2) omp_use_target(use_target)
	for (int j = (y_min + 1); j < (y_max + 2); j++) {
		for (int i = (x_min + 1); i < (x_max + 2); i++) {
			double v = 1.0 / density[i + j * density_sizex];
			pressure[i + j * pressure_sizex] = (1.4 - 1.0) * density[i + j * density_sizex] * energy[i + j * energy_sizex];
			double pressurebyenergy = (1.4 - 1.0) * density[i + j * density_sizex];
			double pressurebyvolume = -density[i + j * density_sizex] * pressure[i + j * pressure_sizex];
			double sound_speed_squared = v * v * (pressure[i + j * pressure_sizex] * pressurebyenergy - pressurebyvolume);
			soundspeed[i + j * soundspeed_sizex] = std::sqrt(sound_speed_squared);
		}
	};

}

//  @brief Ideal gas kernel driver
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the ideal gas equation of
//  state using the specified time level data.

void ideal_gas(global_variables &globals, const int tile, bool predict) {
	if (DEBUG) std::cout << "ideal_gas(tile " << tile << ")" << std::endl;

	tile_type &t = globals.chunk.tiles[tile];

	#if FLUSH_BUFFER
	globals.hostToDevice();
	#endif

	if (!predict) {
		ideal_gas_kernel(
				globals.use_target,
				t.info.t_xmin,
				t.info.t_xmax,
				t.info.t_ymin,
				t.info.t_ymax,
				t.field.density0,
				t.field.energy0,
				t.field.pressure,
				t.field.soundspeed
		);
	} else {
		ideal_gas_kernel(
				globals.use_target,
				t.info.t_xmin,
				t.info.t_xmax,
				t.info.t_ymin,
				t.info.t_ymax,
				t.field.density1,
				t.field.energy1,
				t.field.pressure,
				t.field.soundspeed
		);
	}

	#if FLUSH_BUFFER
	globals.deviceToHost();
	#endif

}

