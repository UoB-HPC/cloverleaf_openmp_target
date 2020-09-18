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

#include "comms.h"

void generate_chunk(const int tile, global_variables &globals) {


	// Need to copy the host array of state input data into a device array
	clover::Buffer1D<double> state_density(globals.config.number_of_states);
	clover::Buffer1D<double> state_energy(globals.config.number_of_states);
	clover::Buffer1D<double> state_xvel(globals.config.number_of_states);
	clover::Buffer1D<double> state_yvel(globals.config.number_of_states);
	clover::Buffer1D<double> state_xmin(globals.config.number_of_states);
	clover::Buffer1D<double> state_xmax(globals.config.number_of_states);
	clover::Buffer1D<double> state_ymin(globals.config.number_of_states);
	clover::Buffer1D<double> state_ymax(globals.config.number_of_states);
	clover::Buffer1D<double> state_radius(globals.config.number_of_states);
	clover::Buffer1D<int> state_geometry(globals.config.number_of_states);


	// Copy the data to the new views
	for (int state = 0; state < globals.config.number_of_states; ++state) {
		idx1(state_density, state) = globals.config.states[state].density;
		idx1(state_energy, state) = globals.config.states[state].energy;
		idx1(state_xvel, state) = globals.config.states[state].xvel;
		idx1(state_yvel, state) = globals.config.states[state].yvel;
		idx1(state_xmin, state) = globals.config.states[state].xmin;
		idx1(state_xmax, state) = globals.config.states[state].xmax;
		idx1(state_ymin, state) = globals.config.states[state].ymin;
		idx1(state_ymax, state) = globals.config.states[state].ymax;
		idx1(state_radius, state) = globals.config.states[state].radius;
		idx1(state_geometry, state) = globals.config.states[state].geometry;
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


	const double state_energy_0 = idx1(state_energy, 0);
	const double state_density_0 = idx1(state_density, 0);
	const double state_xvel_0 = idx1(state_xvel, 0);
	const double state_yvel_0 = idx1(state_yvel, 0);

	// State 1 is always the background state
	omp(parallel(2) enable_target(globals.use_target)
			    mapToFrom2D(field.energy0)
			    mapToFrom2D(field.density0)
			    mapToFrom2D(field.xvel0)
			    mapToFrom2D(field.yvel0)
	)
	for (int j = (0); j < (yrange); j++) {
		for (int i = (0); i < (xrange); i++) {
			idx2(field.energy0, i, j) = state_energy_0;
			idx2(field.density0, i, j) = state_density_0;
			idx2(field.xvel0, i, j) = state_xvel_0;
			idx2(field.yvel0, i, j) = state_yvel_0;
		}
	}


	for (int state = 1; state < globals.config.number_of_states; ++state) {
		omp(parallel(2) enable_target(globals.use_target)
				    mapToFrom2D(field.density0)
				    mapToFrom2D(field.xvel0)
				    mapToFrom2D(field.yvel0)
				    mapToFrom2D(field.energy0)

				    mapToFrom1D(field.cellx)
				    mapToFrom1D(field.celly)

				    mapToFrom1D(field.vertexx)
				    mapToFrom1D(field.vertexy)

				    mapTo1D(state_density)
				    mapTo1D(state_energy)
				    mapTo1D(state_xvel)
				    mapTo1D(state_yvel)
				    mapTo1D(state_xmin)
				    mapTo1D(state_xmax)
				    mapTo1D(state_ymin)
				    mapTo1D(state_ymax)
				    mapTo1D(state_radius)
				    mapTo1D(state_geometry)

		)
		for (int j = (0); j < (yrange); j++) {
			for (int i = (0); i < (xrange); i++) {
				double x_cent = idx1(state_xmin, state);
				double y_cent = idx1(state_ymin, state);
				if (idx1(state_geometry, state) == g_rect) {
					if (idx1(field.vertexx, i + 1) >= idx1(state_xmin, state) && idx1(field.vertexx, i) < idx1(state_xmax, state)) {
						if (idx1(field.vertexy, j + 1) >= idx1(state_ymin, state) && idx1(field.vertexy, j) < idx1(state_ymax, state)) {
							idx2(field.energy0, i, j) = idx1(state_energy, state);
							idx2(field.density0, i, j) = idx1(state_density, state);
							for (int kt = j; kt <= j + 1; ++kt) {
								for (int jt = i; jt <= i + 1; ++jt) {
									idx2(field.xvel0, jt, kt) = idx1(state_xvel, state);
									idx2(field.yvel0, jt, kt) = idx1(state_yvel, state);
								}
							}
						}
					}
				} else if (idx1(state_geometry, state) == g_circ) {
					double radius = std::sqrt((idx1(field.cellx, i) - x_cent) *
					                          (idx1(field.cellx, i) - x_cent) + (idx1(field.celly, j) - y_cent) * (idx1(field.celly, j) - y_cent));
					if (radius <= idx1(state_radius, state)) {
						idx2(field.energy0, i, j) = idx1(state_energy, state);
						idx2(field.density0, i, j) = idx1(state_density, state);
						for (int kt = j; kt <= j + 1; ++kt) {
							for (int jt = i; jt <= i + 1; ++jt) {
								idx2(field.xvel0, jt, kt) = idx1(state_xvel, state);
								idx2(field.yvel0, jt, kt) = idx1(state_yvel, state);
							}
						}
					}
				} else if (idx1(state_geometry, state) == g_point) {
					if (idx1(field.vertexx, i) == x_cent && idx1(field.vertexy, j) == y_cent) {
						idx2(field.energy0, i, j) = idx1(state_energy, state);
						idx2(field.density0, i, j) = idx1(state_density, state);
						for (int kt = j; kt <= j + 1; ++kt) {
							for (int jt = i; jt <= i + 1; ++jt) {
								idx2(field.xvel0, jt, kt) = idx1(state_xvel, state);
								idx2(field.yvel0, jt, kt) = idx1(state_yvel, state);
							}
						}
					}
				}
			}
		}

	}


}

