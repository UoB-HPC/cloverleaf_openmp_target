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


//  @brief Fortran mpi buffer packing kernel
//  @author Wayne Gaudin
//  @details Packs/unpacks mpi send and receive buffers


#include "pack_kernel.h"


void clover_pack_message_left(bool use_target, int x_min, int x_max, int y_min, int y_max,
                              clover::Buffer2D<double> &field_buffer,
                              clover::Buffer1D<double> &left_snd_buffer,
                              int cell_data, int vertex_data, int x_face_data, int y_face_data,
                              int depth, int field_type, int buffer_offset) {

	// Pack

	int x_inc, y_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		x_inc = 0;
		y_inc = 0;
	}
	if (field_type == vertex_data) {
		x_inc = 1;
		y_inc = 1;
	}
	if (field_type == x_face_data) {
		x_inc = 1;
		y_inc = 0;
	}
	if (field_type == y_face_data) {
		x_inc = 0;
		y_inc = 1;
	}

	// DO k=y_min-depth,y_max+y_inc+depth


	double *left_snd = left_snd_buffer.data;
	double *field = field_buffer.data;
	const int field_sizex = field_buffer.sizeX;
	#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
	for (int k = (y_min - depth + 1); k < (y_max + y_inc + depth + 2); k++) {
		for (int j = 0; j < depth; ++j) {
			int index = buffer_offset + j + (k + depth - 1) * depth;
			left_snd[index] = field[(x_min + x_inc - 1 + j) + (k) * field_sizex];
		}
	}


}


void clover_unpack_message_left(bool use_target, int x_min, int x_max, int y_min, int y_max,
                                clover::Buffer2D<double> &field_buffer,
                                clover::Buffer1D<double> &left_rcv_buffer,
                                int cell_data, int vertex_data, int x_face_data, int y_face_data,
                                int depth, int field_type, int buffer_offset) {

	// Upnack

	int y_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		y_inc = 0;
	}
	if (field_type == vertex_data) {
		y_inc = 1;
	}
	if (field_type == x_face_data) {
		y_inc = 0;
	}
	if (field_type == y_face_data) {
		y_inc = 1;
	}

	// DO k=y_min-depth,y_max+y_inc+depth




	double *field = field_buffer.data;
	const int field_sizex = field_buffer.sizeX;
	double *left_rcv = left_rcv_buffer.data;
	#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
	for (int k = (y_min - depth + 1); k < (y_max + y_inc + depth + 2); k++) {
		for (int j = 0; j < depth; ++j) {
			int index = buffer_offset + j + (k + depth - 1) * depth;
			field[(x_min - j) + (k) * field_sizex] = left_rcv[index];
		}
	}


}


void clover_pack_message_right(bool use_target, int x_min, int x_max, int y_min, int y_max,
                               clover::Buffer2D<double> &field_buffer,
                               clover::Buffer1D<double> &right_snd_buffer,
                               int cell_data, int vertex_data, int x_face_data, int y_face_data,
                               int depth, int field_type, int buffer_offset) {

	// Pack

	int y_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		y_inc = 0;
	}
	if (field_type == vertex_data) {
		y_inc = 1;
	}
	if (field_type == x_face_data) {
		y_inc = 0;
	}
	if (field_type == y_face_data) {
		y_inc = 1;
	}

	// DO k=y_min-depth,y_max+y_inc+depth
	double *right_snd = right_snd_buffer.data;
	double *field = field_buffer.data;
	const int field_sizex = field_buffer.sizeX;
	#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
	for (int k = (y_min - depth + 1); k < (y_max + y_inc + depth + 2); k++) {
		for (int j = 0; j < depth; ++j) {
			int index = buffer_offset + j + (k + depth - 1) * depth;
			right_snd[index] = field[(x_min + 1 + j) + (k) * field_sizex];
		}
	}


}


void clover_unpack_message_right(bool use_target, int x_min, int x_max, int y_min, int y_max,
                                 clover::Buffer2D<double> &field_buffer,
                                 clover::Buffer1D<double> &right_rcv_buffer,
                                 int cell_data, int vertex_data, int x_face_data, int y_face_data,
                                 int depth, int field_type, int buffer_offset) {

	// Upnack

	int x_inc, y_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		x_inc = 0;
		y_inc = 0;
	}
	if (field_type == vertex_data) {
		x_inc = 1;
		y_inc = 1;
	}
	if (field_type == x_face_data) {
		x_inc = 1;
		y_inc = 0;
	}
	if (field_type == y_face_data) {
		x_inc = 0;
		y_inc = 1;
	}

	// DO k=y_min-depth,y_max+y_inc+depth
	double *right_rcv = right_rcv_buffer.data;
	double *field = field_buffer.data;
	const int field_sizex = field_buffer.sizeX;
	#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
	for (int k = (y_min - depth + 1); k < (y_max + y_inc + depth + 2); k++) {
		for (int j = 0; j < depth; ++j) {
			int index = buffer_offset + j + (k + depth - 1) * depth;
			right_rcv[index] = field[(x_max + x_inc + j) + (k) * field_sizex];
		}
	}


}

void clover_pack_message_top(bool use_target, int x_min, int x_max, int y_min, int y_max,
                             clover::Buffer2D<double> &field_buffer,
                             clover::Buffer1D<double> &top_snd_buffer,
                             int cell_data, int vertex_data, int x_face_data, int y_face_data,
                             int depth, int field_type, int buffer_offset) {

	// Pack

	int x_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		x_inc = 0;
	}
	if (field_type == vertex_data) {
		x_inc = 1;
	}
	if (field_type == x_face_data) {
		x_inc = 1;
	}
	if (field_type == y_face_data) {
		x_inc = 0;
	}

	for (int k = 0; k < depth; ++k) {
		// DO j=x_min-depth,x_max+x_inc+depth

		double *top_snd = top_snd_buffer.data;
		double *field = field_buffer.data;
		const int field_sizex = field_buffer.sizeX;
		#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
		for (int j = (x_min - depth + 1); j < (x_max + x_inc + depth + 2); j++) {
			int index = buffer_offset + k + (j + depth - 1) * depth;
			top_snd[index] = field[j + (y_max + 1 - k) * field_sizex];
		}
	}
}

void clover_unpack_message_top(bool use_target, int x_min, int x_max, int y_min, int y_max,
                               clover::Buffer2D<double> &field_buffer,
                               clover::Buffer1D<double> &top_rcv_buffer,
                               int cell_data, int vertex_data, int x_face_data, int y_face_data,
                               int depth, int field_type, int buffer_offset) {

	// Unpack

	int x_inc, y_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		x_inc = 0;
		y_inc = 0;
	}
	if (field_type == vertex_data) {
		x_inc = 1;
		y_inc = 1;
	}
	if (field_type == x_face_data) {
		x_inc = 1;
		y_inc = 0;
	}
	if (field_type == y_face_data) {
		x_inc = 0;
		y_inc = 1;
	}

	for (int k = 0; k < depth; ++k) {
		// DO j=x_min-depth,x_max+x_inc+depth


		double *field = field_buffer.data;
		const int field_sizex = field_buffer.sizeX;
		double *top_rcv = top_rcv_buffer.data;
		#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
		for (int j = (x_min - depth + 1); j < (x_max + x_inc + depth + 2); j++) {
			int index = buffer_offset + k + (j + depth - 1) * depth;
			field[j + (y_max + y_inc + k) * field_sizex] = top_rcv[index];
		}
	}
}


void clover_pack_message_bottom(bool use_target, int x_min, int x_max, int y_min, int y_max,
                                clover::Buffer2D<double> &field_buffer,
                                clover::Buffer1D<double> &bottom_snd_buffer,
                                int cell_data, int vertex_data, int x_face_data, int y_face_data,
                                int depth, int field_type, int buffer_offset) {

	// Pack

	int x_inc, y_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		x_inc = 0;
		y_inc = 0;
	}
	if (field_type == vertex_data) {
		x_inc = 1;
		y_inc = 1;
	}
	if (field_type == x_face_data) {
		x_inc = 1;
		y_inc = 0;
	}
	if (field_type == y_face_data) {
		x_inc = 0;
		y_inc = 1;
	}

	for (int k = 0; k < depth; ++k) {
		// DO j=x_min-depth,x_max+x_inc+depth

		double *bottom_snd = bottom_snd_buffer.data;
		double *field = field_buffer.data;
		const int field_sizex = field_buffer.sizeX;
		#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
		for (int j = (x_min - depth + 1); j < (x_max + x_inc + depth + 2); j++) {
			int index = buffer_offset + k + (j + depth - 1) * depth;
			bottom_snd[index] = field[j + (y_min + y_inc - 1 + k) * field_sizex];
		}
	}
}

void clover_unpack_message_bottom(bool use_target, int x_min, int x_max, int y_min, int y_max,
                                  clover::Buffer2D<double> &field_buffer,
                                  clover::Buffer1D<double> &bottom_rcv_buffer,
                                  int cell_data, int vertex_data, int x_face_data, int y_face_data,
                                  int depth, int field_type, int buffer_offset) {

	// Unpack

	int x_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		x_inc = 0;
	}
	if (field_type == vertex_data) {
		x_inc = 1;
	}
	if (field_type == x_face_data) {
		x_inc = 1;
	}
	if (field_type == y_face_data) {
		x_inc = 0;
	}

	for (int k = 0; k < depth; ++k) {
		// DO j=x_min-depth,x_max+x_inc+depth

		double *field = field_buffer.data;
		const int field_sizex = field_buffer.sizeX;
		double *bottom_rcv = bottom_rcv_buffer.data;
		#pragma omp target teams distribute parallel for simd omp_use_target(use_target)
		for (int j = (x_min - depth + 1); j < (x_max + x_inc + depth + 2); j++) {
			int index = buffer_offset + k + (j + depth - 1) * depth;
			field[j + (y_min - k) * field_sizex] = bottom_rcv[index];
		}
	}
}
