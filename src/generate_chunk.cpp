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


//  @brief Mesh chunk generation driver
//  @author Wayne Gaudin
//  @details Invoked the users specified chunk generator.
//  @brief Mesh chunk generation driver
//  @author Wayne Gaudin
//  @details Invoked the users specified chunk generator.

#include <cmath>
#include "generate_chunk.h"

void generate_chunk(const int tile, global_variables &globals) {


	// Need to copy the host array of state input data into a device array
	clover::Buffer1D<double> state_density_buffer(globals.config.number_of_states);
	clover::Buffer1D<double> state_energy_buffer(globals.config.number_of_states);
	clover::Buffer1D<double> state_xvel_buffer(globals.config.number_of_states);
	clover::Buffer1D<double> state_yvel_buffer(globals.config.number_of_states);
	clover::Buffer1D<double> state_xmin_buffer(globals.config.number_of_states);
	clover::Buffer1D<double> state_xmax_buffer(globals.config.number_of_states);
	clover::Buffer1D<double> state_ymin_buffer(globals.config.number_of_states);
	clover::Buffer1D<double> state_ymax_buffer(globals.config.number_of_states);
	clover::Buffer1D<double> state_radius_buffer(globals.config.number_of_states);
	clover::Buffer1D<int> state_geometry_buffer(globals.config.number_of_states);


	// Copy the data to the new views
	for (int state = 0; state < globals.config.number_of_states; ++state) {
		idx1(state_density_buffer, state) = globals.config.states[state].density;
		idx1(state_energy_buffer, state) = globals.config.states[state].energy;
		idx1(state_xvel_buffer, state) = globals.config.states[state].xvel;
		idx1(state_yvel_buffer, state) = globals.config.states[state].yvel;
		idx1(state_xmin_buffer, state) = globals.config.states[state].xmin;
		idx1(state_xmax_buffer, state) = globals.config.states[state].xmax;
		idx1(state_ymin_buffer, state) = globals.config.states[state].ymin;
		idx1(state_ymax_buffer, state) = globals.config.states[state].ymax;
		idx1(state_radius_buffer, state) = globals.config.states[state].radius;
		idx1(state_geometry_buffer, state) = globals.config.states[state].geometry;
	}

	// Kokkos::deep_copy (TO, FROM)

	const int x_min = globals.chunk.tiles[tile].info.t_xmin;
	const int x_max = globals.chunk.tiles[tile].info.t_xmax;
	const int y_min = globals.chunk.tiles[tile].info.t_ymin;
	const int y_max = globals.chunk.tiles[tile].info.t_ymax;

	int xrange = (x_max + 2) - (x_min - 2) + 1;
	int yrange = (y_max + 2) - (y_min - 2) + 1;

	// Take a reference to the lowest structure, as Kokkos device cannot necessarily chase through the structure.



	field_type &field = globals.chunk.tiles[tile].field;


	const double state_energy_0 = idx1(state_energy_buffer, 0);
	const double state_density_0 = idx1(state_density_buffer, 0);
	const double state_xvel_0 = idx1(state_xvel_buffer, 0);
	const double state_yvel_0 = idx1(state_yvel_buffer, 0);

	// State 1 is always the background state
	mapToFrom2Df(field, energy0)
	mapToFrom2Df(field, density0)
	mapToFrom2Df(field, xvel0)
	mapToFrom2Df(field, yvel0)

	omp(parallel(2) enable_target(globals.use_target))
	for (int j = (0); j < (yrange); j++) {
		for (int i = (0); i < (xrange); i++) {
			idx2f(field, energy0, i, j) = state_energy_0;
			idx2f(field, density0, i, j) = state_density_0;
			idx2f(field, xvel0, i, j) = state_xvel_0;
			idx2f(field, yvel0, i, j) = state_yvel_0;
		}
	}


	for (int state = 1; state < globals.config.number_of_states; ++state) {

		mapToFrom1Df(field, cellx)
		mapToFrom1Df(field, celly)

		mapToFrom1Df(field, vertexx)
		mapToFrom1Df(field, vertexy)

		const double *state_density = state_density_buffer.data;
		const double *state_energy = state_energy_buffer.data;
		const double *state_xvel = state_xvel_buffer.data;
		const double *state_yvel = state_yvel_buffer.data;
		const double *state_xmin = state_xmin_buffer.data;
		const double *state_xmax = state_xmax_buffer.data;
		const double *state_ymin = state_ymin_buffer.data;
		const double *state_ymax = state_ymax_buffer.data;
		const double *state_radius = state_radius_buffer.data;
		const int *state_geometry = state_geometry_buffer.data;

		omp(parallel(2) enable_target(globals.use_target)
				    map(to : state_density[:state_density_buffer.N()])
				    map(to : state_energy[:state_energy_buffer.N()])
				    map(to : state_xvel[:state_xvel_buffer.N()])
				    map(to : state_yvel[:state_yvel_buffer.N()])
				    map(to : state_xmin[:state_xmin_buffer.N()])
				    map(to : state_xmax[:state_xmax_buffer.N()])
				    map(to : state_ymin[:state_ymin_buffer.N()])
				    map(to : state_ymax[:state_ymax_buffer.N()])
				    map(to : state_radius[:state_radius_buffer.N()])
				    map(to : state_geometry[:state_geometry_buffer.N()])
		)
		for (int j = (0); j < (yrange); j++) {
			for (int i = (0); i < (xrange); i++) {
				double x_cent = state_xmin[state];
				double y_cent = state_ymin[state];
				if (state_geometry[state] == g_rect) {
					if (idx1f(field, vertexx, i + 1) >= state_xmin[state] && idx1f(field, vertexx, i) < state_xmax[state]) {
						if (idx1f(field, vertexy, j + 1) >= state_ymin[state] && idx1f(field, vertexy, j) < state_ymax[state]) {
							idx2f(field, energy0, i, j) = state_energy[state];
							idx2f(field, density0, i, j) = state_density[state];
							for (int kt = j; kt <= j + 1; ++kt) {
								for (int jt = i; jt <= i + 1; ++jt) {
									idx2f(field, xvel0, jt, kt) = state_xvel[state];
									idx2f(field, yvel0, jt, kt) = state_yvel[state];
								}
							}
						}
					}
				} else if (state_geometry[state] == g_circ) {
					double radius = sqrt((idx1f(field, cellx, i) - x_cent) *
					                     (idx1f(field, cellx, i) - x_cent) + (idx1f(field, celly, j) - y_cent) * (idx1f(field, celly, j) - y_cent));
					if (radius <= state_radius[state]) {
						idx2f(field, energy0, i, j) = state_energy[state];
						idx2f(field, density0, i, j) = state_density[state];
						for (int kt = j; kt <= j + 1; ++kt) {
							for (int jt = i; jt <= i + 1; ++jt) {
								idx2f(field, xvel0, jt, kt) = state_xvel[state];
								idx2f(field, yvel0, jt, kt) = state_yvel[state];
							}
						}
					}
				} else if (state_geometry[state] == g_point) {
					if (idx1f(field, vertexx, i) == x_cent && idx1f(field, vertexy, j) == y_cent) {
						idx2f(field, energy0, i, j) = state_energy[state];
						idx2f(field, density0, i, j) = state_density[state];
						for (int kt = j; kt <= j + 1; ++kt) {
							for (int jt = i; jt <= i + 1; ++jt) {
								idx2f(field, xvel0, jt, kt) = state_xvel[state];
								idx2f(field, yvel0, jt, kt) = state_yvel[state];
							}
						}
					}
				}
			}
		}

	}


}

