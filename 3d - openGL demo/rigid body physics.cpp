#include <windows.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <GL/gl.h>
#include <GL/glu.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <random>
#include <assert.h>
#include "rigid_body_physics.h"

#define DETER_SUBSAMPLE -1
#define U_RANDOM_SUBSAMPLE -2
#define SUBSAMPLE_GAP 3 

#define SUBDIV_DEPTH 2

#define BOUNDING_BOX 100
#define BOUNDING_SPHERE 200
#define BOUNDING_CONVEX 300
#define SUB_MESH 400

#define TRIANGLE_POL 3
#define QUAD_POL 4

void translate_vertices_LEGACY_GL(std::vector<std::vector<std::vector<GLfloat>>>& vertices_per_object,
	std::vector<std::vector<GLfloat>>& vertex_vec) {
	for (int i = 0; i < (int)vertices_per_object.size(); i++) {
		for (int j = 0; j < (int)vertices_per_object[i].size(); j++) {
			vertex_vec.push_back(vertices_per_object[i][j]);
		}
	}
}

void sortByIndex(GLfloat vertex_data[], GLushort index[], int ISIZE, std::vector<std::vector<GLfloat>>& vertex_vec) {
	int vec_size = 3;
	for (int i = 0; i < ISIZE; i++) {
		std::vector<GLfloat> vertex;
		vertex.push_back(vertex_data[index[i] * vec_size]);
		vertex.push_back(vertex_data[index[i] * vec_size + 1]);
		vertex.push_back(vertex_data[index[i] * vec_size + 2]);
		vertex_vec.push_back(vertex);
	}
}

void generate_tinyBOX_mesh(std::vector<std::vector<GLfloat>>& vertex_vec, GLfloat center[], GLfloat epsilon) {
	GLfloat x_0 = center[0];
	GLfloat y_0 = center[1];
	GLfloat z_0 = center[2];

	GLfloat cube_vertices[] = {
		-epsilon + x_0, -epsilon + y_0,  epsilon + z_0,
		 epsilon + x_0, -epsilon + y_0,  epsilon + z_0,
		 epsilon + x_0,  epsilon + y_0,  epsilon + z_0,
		-epsilon + x_0,  epsilon + y_0,  epsilon + z_0,
		-epsilon + x_0, -epsilon + y_0, -epsilon + z_0,
		 epsilon + x_0, -epsilon + y_0, -epsilon + z_0,
		 epsilon + x_0,  epsilon + y_0, -epsilon + z_0,
		-epsilon + x_0,  epsilon + y_0, -epsilon + z_0
	};
	// starting from 0
	GLushort cube_index[] = {
		0, 1, 2,
		2, 3, 0,
		1, 5, 6,
		6, 2, 1,
		7, 6, 5,
		5, 4, 7,
		4, 0, 3,
		3, 7, 4,
		4, 5, 1,
		1, 0, 4,
		3, 2, 6,
		6, 7, 3
	};
	int index_size = 12; // for a trianglular polygon cube
	sortByIndex(cube_vertices, cube_index, index_size, vertex_vec);
}

void generate_tinyCONVEX_mesh(std::vector<std::vector<GLfloat>>& mesh, GLfloat center[], GLfloat epsilon) {
	// icosahedron mesh 
	GLfloat X = 0.5257 * epsilon;
	GLfloat Z = 0.8507 * epsilon;

	GLfloat x_0 = center[0];
	GLfloat y_0 = center[1];
	GLfloat z_0 = center[2];

	GLfloat vdata[] = {
	   -X + x_0, 0.0 + y_0, Z + z_0,
	   X + x_0, 0.0 + y_0, Z + z_0,
	   -X + x_0, 0.0 + y_0, -Z + z_0,
	   X + x_0, 0.0 + y_0, -Z + z_0,
	   0.0 + x_0, Z + y_0, X + z_0,
	   0.0 + x_0, Z + y_0, -X + z_0,
	   0.0 + x_0, -Z + y_0, X + z_0,
	   0.0 + x_0, -Z + y_0, -X + z_0,
	   Z + x_0, X + y_0, 0.0 + z_0,
	   -Z + x_0, X + y_0, 0.0 + z_0,
	   Z + x_0, -X + y_0, 0.0 + z_0,
	   -Z + x_0, -X + y_0, 0.0 + z_0
	};
	GLushort indices[] = {
	   0,4,1,
	   0,9,4,
	   9,5,4,
	   4,5,8,
	   4,8,1,
	   8,10,1,
	   8,3,10,
	   5,3,8,
	   5,2,3,
	   2,7,3,
	   7,10,3,
	   7,6,10,
	   7,11,6,
	   11,0,6,
	   0,1,6,
	   6,1,10,
	   9,0,11,
	   9,11,2,
	   9,2,5,
	   7,2,11
	};
	int index_size = 20; // for a trianglular polygon convex solid (icosahedron)
	sortByIndex(vdata, indices, index_size, mesh);
}

// uses halley's method of approximating square root
float fast_sqrt(float num) {
	// initial guess
	float x = 10.0;
	int iters = 3;

	int i = 0;
	while (i < iters) {
		x = (x * x * x + 3 * num * x) / (3 * x * x + num);
	}

	return x;
}

// generating a "sphere-like" mesh by using subdivision
void normalize(float v[3]) {
	// mathematically normalize vectors v -> v/||v||
	GLfloat d = fast_sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	if (d == 0.0) {
		std::cout << "zero length vector";
		return;
	}
	v[0] /= d;
	v[1] /= d;
	v[2] /= d;
}

void PolygonSubdivision(float* v1, float* v2, float* v3, long depth, std::vector<std::vector<GLfloat>>& mesh) {

	GLfloat v12[3], v23[3], v31[3];
	GLint i;
	// stopping condition for recursive subdividing
	if (depth == 0) {
		// take every vertex made
		std::vector<GLfloat> vec1, vec2, vec3;
		vec1.assign(v1, v1 + 3);
		vec2.assign(v2, v2 + 3);
		vec3.assign(v3, v3 + 3);
		mesh.push_back(vec1);
		mesh.push_back(vec2);
		mesh.push_back(vec3);
		return;
	}
	/* the method of dividing the triangles is by choosing a point on the edges between each of the vetrices,
	taking the vector from the origin.
	to that specific point and normalizing the vector,
	and using those vectors as normals and vetrices for the new triangles.
	the process is done recursively untill we reach depth 0 in the recursive tree.
	the depth of the tree is defined by hand*/
	for (i = 0; i < 3; i++) {
		v12[i] = v1[i] + v2[i];
		v23[i] = v2[i] + v3[i];
		v31[i] = v3[i] + v1[i];
	}
	normalize(v12);
	normalize(v23);
	normalize(v31);
	// there are 4 initial branches of the tree because creating 3 middle points in a triangle creates 4 different triangles.
	PolygonSubdivision(v1, v12, v31, depth - 1, mesh);
	PolygonSubdivision(v2, v23, v12, depth - 1, mesh);
	PolygonSubdivision(v3, v31, v23, depth - 1, mesh);
	PolygonSubdivision(v12, v23, v31, depth - 1, mesh);
}

void generate_tinySPHERE_mesh(std::vector<std::vector<GLfloat>>& mesh, GLfloat center[3], GLfloat epsilon, int depth) {
	// icosahedron mesh 
	GLfloat X = 0.5257 * epsilon;
	GLfloat Z = 0.8507 * epsilon;

	GLfloat x_0 = center[0];
	GLfloat y_0 = center[1];
	GLfloat z_0 = center[2];

	static GLfloat vdata[12][3] = {
	   {-X + x_0, 0.0 + y_0, Z + z_0},
	   {X + x_0, 0.0 + y_0, Z + z_0},
	   {-X + x_0, 0.0 + y_0, -Z + z_0},
	   {X + x_0, 0.0 + y_0, -Z + z_0},
	   {0.0 + x_0, Z + y_0, X + z_0},
	   {0.0 + x_0, Z + y_0, -X + z_0},
	   {0.0 + x_0, -Z + y_0, X + z_0},
	   {0.0 + x_0, -Z + y_0, -X + z_0},
	   {Z + x_0, X + y_0, 0.0 + z_0},
	   {-Z + x_0, X + y_0, 0.0 + z_0},
	   {Z + x_0, -X + y_0, 0.0 + z_0},
	   {-Z + x_0, -X + y_0, 0.0 + z_0}
	};
	static GLuint indices[20][3] = {
	   {0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},
	   {8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},
	   {7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6},
	   {6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11} };

	for (int i = 0; i < 20; i++) {
		PolygonSubdivision(&vdata[indices[i][0]][0],
			&vdata[indices[i][1]][0],
			&vdata[indices[i][2]][0], depth, mesh);
	}
}

// good only for low poly models
auto generate_random_number_INT(int min, int max) {
	std::random_device rd;     // only used once to initialise (seed) engine
	std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
	std::uniform_int_distribution<int> uniform(min, max); // guaranteed unbiased

	auto random_integer = uniform(rng);
	return random_integer;
}

// good only for low poly models
void generate_SUBSAMPLE_mesh(std::vector<std::vector<GLfloat>>& mesh,
	std::vector<std::vector<GLfloat>>& orig_mesh,
	int sample_mode) {

	int i1 = 0;
	int i2 = 0;
	if (sample_mode == DETER_SUBSAMPLE) {
		while (i1 < (int)orig_mesh.size()) {
			mesh.push_back(orig_mesh[i1]);
			i1 += SUBSAMPLE_GAP;
		}
	}
	else if (sample_mode == U_RANDOM_SUBSAMPLE) {
		double size_ratio = (double)((int)orig_mesh.size() / SUBSAMPLE_GAP);
		int mesh_size = (int)ceil(size_ratio);
		while (i2 < mesh_size) {
			int rand_num = generate_random_number_INT(0, (int)orig_mesh.size() - 1);
			mesh.push_back(orig_mesh[rand_num]);
			i2++;
		}
	}
}

void init_3Dvec(std::vector<GLfloat>& v) {
	v.push_back(0.0);
	v.push_back(0.0);
	v.push_back(0.0);
}

void add_3Dvectors(std::vector<GLfloat>& v_r, std::vector<GLfloat>& v_a) {
	v_r[0] = v_r[0] + v_a[0];
	v_r[1] = v_r[1] + v_a[1];
	v_r[2] = v_r[2] + v_a[2];
}

void scale_3Dvectors(std::vector<GLfloat>& v, GLfloat scale) {
	v[0] *= scale;
	v[1] *= scale;
	v[2] *= scale;
}

void add_3Dvec_to_mesh(std::vector<std::vector<GLfloat>>& mesh, std::vector<GLfloat>& v) {
	for (int i = 0; i < (int)mesh.size(); i++) {
		add_3Dvectors(mesh[i], v);
	}
}

void scale_3Dmesh(std::vector<std::vector<GLfloat>>& mesh, GLfloat scale) {
	for (int i = 0; i < (int)mesh.size(); i++) {
		scale_3Dvectors(mesh[i], scale);
	}
}

GLfloat* geometrig_center(std::vector<std::vector<GLfloat>>& model_mesh) {
	std::vector<GLfloat> center(3);
	init_3Dvec(center);
	std::vector<std::vector<GLfloat>>::iterator v = model_mesh.begin();
	while (v != model_mesh.end()) {
		add_3Dvectors(center, *v);
		v++;
	}
	GLfloat scale = (GLfloat)(1 / model_mesh.size());
	scale_3Dvectors(center, scale);

	GLfloat geomCenter[3];
	std::copy(center.begin(), center.end(), geomCenter);

	return geomCenter;
}

bool detect_boundries(std::vector<std::vector<GLfloat>>& model_mesh,
	std::vector<std::vector<GLfloat>>& bounding_mesh, GLfloat center[]) {

	std::vector<GLfloat> geomCenter(center, center + 3);
	scale_3Dvectors(geomCenter, -1.0);
	add_3Dvec_to_mesh(model_mesh, geomCenter);
	add_3Dvec_to_mesh(bounding_mesh, geomCenter);

	bool broke = false;
	for (int l1 = 0; l1 < (int)model_mesh.size(); l1++) {
		float d1 = sqrt((float)(model_mesh[l1][0] * model_mesh[l1][0]) +
			(float)(model_mesh[l1][1] * model_mesh[l1][1]) +
			(float)(model_mesh[l1][2] * model_mesh[l1][2]));
		for (int l2 = 0; l2 < (int)bounding_mesh.size(); l2++) {
			float d2 = sqrt((float)(bounding_mesh[l2][0] * bounding_mesh[l2][0]) +
				(float)(bounding_mesh[l2][1] * bounding_mesh[l2][1]) +
				(float)(bounding_mesh[l2][2] * bounding_mesh[l2][2]));
			if (d1 <= d2) {
				broke = true;
				break;
			}
		}
		if (broke) {
			break;
		}
	}
	return broke;
}

void inflate_mesh(std::vector<std::vector<GLfloat>>& model_mesh,
	std::vector<std::vector<GLfloat>>& bounding_mesh,
	GLfloat center[],
	GLfloat init_scale,
	GLfloat step) {

	GLfloat scale = init_scale;
	while (!detect_boundries(model_mesh, bounding_mesh, center)) {
		scale += step;
		scale_3Dmesh(bounding_mesh, scale);
	}
}

void fitMesh(std::vector<std::vector<GLfloat>>& model_mesh, std::vector<std::vector<GLfloat>>& bounding_mesh, int mesh_type) {
	GLfloat epsilon = 0.1;
	GLfloat step = 0.0005;
	GLfloat scaleFactor = 1.0;

	GLfloat* geomCenter = geometrig_center(model_mesh);

	switch (mesh_type) {
	case BOUNDING_BOX:
		generate_tinyBOX_mesh(bounding_mesh, geomCenter, epsilon);
		inflate_mesh(model_mesh, bounding_mesh, geomCenter, scaleFactor, step);
	case BOUNDING_CONVEX:
		generate_tinyCONVEX_mesh(bounding_mesh, geomCenter, epsilon);
		inflate_mesh(model_mesh, bounding_mesh, geomCenter, scaleFactor, step);
	case BOUNDING_SPHERE:
		generate_tinySPHERE_mesh(bounding_mesh, geomCenter, epsilon, SUBDIV_DEPTH);
		inflate_mesh(model_mesh, bounding_mesh, geomCenter, scaleFactor, step);
	case SUB_MESH:
		generate_SUBSAMPLE_mesh(bounding_mesh, model_mesh, DETER_SUBSAMPLE);
	}
}

void fitMesh_from_orig_Model_LEGACY_GL(std::vector<std::vector<std::vector<GLfloat>>>& vertices,
	std::vector<std::vector<GLfloat>>& bounding_mesh,
	int mesh_type) {

	std::vector<std::vector<GLfloat>> model_mesh;
	translate_vertices_LEGACY_GL(vertices, model_mesh);
	fitMesh(model_mesh, bounding_mesh, mesh_type);
}

void find_max_min_Coords_vals(std::vector<std::vector<GLfloat>>& box, 
	GLfloat& x, GLfloat& y, GLfloat& z,
	GLfloat& mx, GLfloat& my, GLfloat& mz) {

	GLfloat max_X = box[0][0];
	GLfloat max_Y = box[0][1];
	GLfloat max_Z = box[0][2];

	GLfloat min_X = box[0][0];
	GLfloat min_Y = box[0][1];
	GLfloat min_Z = box[0][2];

	std::vector<std::vector<GLfloat>>::iterator v = box.begin();
	while (v != box.end()) {
		std::vector<GLfloat> vertex = *v;
		if (vertex[0] > max_X) { max_X = vertex[0]; }
		if (vertex[1] > max_Y) { max_Y = vertex[1]; }
		if (vertex[2] > max_Z) { max_Z = vertex[2]; }
		
		if (vertex[0] < min_X) { min_X = vertex[0]; }
		if (vertex[1] < min_Y) { min_Y = vertex[1]; }
		if (vertex[2] < min_Z) { min_Z = vertex[2]; }
		v++;
	}

	x = max_X;
	y = max_Y;
	z = max_Z;

	mx = min_X;
	my = min_Y;
	mz = min_Z;
}

bool detect_collision_BOX_vs_BOX(std::vector<std::vector<GLfloat>>& box1, std::vector<std::vector<GLfloat>>& box2) {
	GLfloat x1, y1, z1, mx1, my1, mz1;
	GLfloat x2, y2, z2, mx2, my2, mz2;
	find_max_min_Coords_vals(box1, x1, y1, z1, mx1, my1, mz1);
	find_max_min_Coords_vals(box2, x2, y2, z2, mx2, my2, mz2);

	if ((mx1 <= x2 && x1 >= mx2) && (my1 <= y2 && y1 >= my2) && (mz1 <= z2 && z1 >= mz2)) {
		return true;
	}
	return false;
}

float radius_squared(std::vector<GLfloat>& point, GLfloat center[]) {
	std::vector<GLfloat> geomCenter(center, center + 3);
	std::vector<GLfloat> relative_vec = point;

	// subtract two vectors
	scale_3Dvectors(geomCenter, -1.0);
	add_3Dvectors(relative_vec, geomCenter);

	// calculating the x, y, z components squared
	float x_c_sq = (float)relative_vec[0] * relative_vec[0];
	float y_c_sq = (float)relative_vec[1] * relative_vec[1];
	float z_c_sq = (float)relative_vec[2] * relative_vec[2];

	return x_c_sq + y_c_sq + z_c_sq;
}

float approx_radius(std::vector<GLfloat>& point, GLfloat center[]) {

	return fast_sqrt(radius_squared(point, center));
}

float approx_distance(GLfloat point1[], GLfloat point2[]) {
	float x_sq = (float)(point2[0] - point1[0]) * (point2[0] - point1[0]);
	float y_sq = (float)(point2[1] - point1[1]) * (point2[1] - point1[1]);
	float z_sq = (float)(point2[2] - point1[2]) * (point2[2] - point1[2]);

	return fast_sqrt(x_sq + y_sq + z_sq);
}

bool detect_collision_SPHERE_vs_SPHERE(std::vector<std::vector<GLfloat>>& sphere1,
	std::vector<std::vector<GLfloat>>& sphere2,
	GLfloat center1[],
	GLfloat center2[]) {
	std::vector<GLfloat> refPoint1 = sphere1[0];
	std::vector<GLfloat> refPoint2 = sphere2[0];
	if (approx_distance(center1, center2) <= approx_radius(refPoint1, center1) + approx_radius(refPoint2, center2)) {
		return true;
	}
	return false;
}

bool detect_collision_BOX_vs_SPHERE(std::vector<std::vector<GLfloat>>& box, std::vector<std::vector<GLfloat>>& sphere, GLfloat center[]) {
	GLfloat box_max[3];
	GLfloat box_min[3];
	find_max_min_Coords_vals(box, box_max[0], box_max[1], box_max[2], box_min[0], box_min[1], box_min[2]);

	std::vector<GLfloat> ref_p = sphere[0];
	float r_sq = radius_squared(ref_p, center);
	float d_min_sq = 0;
	for (int i = 0; i < 3; i++) {
		if (center[i] < box_min[i]) { d_min_sq += (float)(center[i] - box_min[i]) * (center[i] - box_min[i]); }
		else if (center[i] > box_max[i]) { d_min_sq += (float)(center[i] - box_max[i]) * (center[i] - box_max[i]); }
	}
	return d_min_sq < r_sq;
}

void cross_product(std::vector<GLfloat>& v1, std::vector<GLfloat>& v2, std::vector<GLfloat>& res) {
	GLfloat c_x = v1[1] * v2[2] - v1[2] * v2[1];
	GLfloat c_y = v1[2] * v2[0] - v1[0] * v2[2];
	GLfloat c_z = v1[0] * v2[1] - v1[1] * v2[0];
	res.push_back(c_x);
	res.push_back(c_y);
	res.push_back(c_z);
}

void normalize_vec(std::vector<GLfloat>& v) {
	float v_arr[3];
	std::copy(v.begin(), v.end(), v_arr);
	normalize(v_arr);
	std::vector<GLfloat> temp_v(v_arr, v_arr + 3);
	v = temp_v;
}

// for shapes built out of triangles only - suitble for all bounding meshes
void f_normal_per_polygon(std::vector<std::vector<GLfloat>>& polygon, std::vector<GLfloat>& normal) {	
	std::vector<GLfloat> vec1 = polygon[1];
	std::vector<GLfloat> vec2 = polygon[2];

	std::vector<GLfloat> base = polygon[0];
	scale_3Dvectors(base, -1.0);
	add_3Dvectors(vec1, base);
	add_3Dvectors(vec2, base);

	cross_product(vec1, vec2, normal);
	normalize_vec(normal);
}

void f_normals(std::vector<std::vector<GLfloat>>& mesh, std::vector<std::vector<GLfloat>>& f_normals, int pol_type) {

	std::vector<std::vector<GLfloat>> f_n;
	int mesh_size = (int)mesh.size();
	int polygon_count = (int)(mesh_size / pol_type);
	for (int i = 0; i < polygon_count; i++) {
		std::vector<GLfloat> normal;

		std::vector<std::vector<GLfloat>>::iterator m_st = mesh.begin() + i * pol_type;
		std::vector<std::vector<GLfloat>>::iterator m_end = mesh.begin() + i * pol_type + 3;
		std::vector<std::vector<GLfloat>> polygon(m_st, m_end);

		f_normal_per_polygon(polygon, normal);
		f_n.push_back(normal);
	}
	f_normals = f_n;
}

GLfloat scalar_mult(std::vector<GLfloat>& v1, std::vector<GLfloat>& v2) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

void project_3D_along_axis(std::vector<std::vector<GLfloat>>& obj,
	std::vector<GLfloat>& axis,
	std::vector<std::vector<GLfloat>>& proj) {

	std::vector<std::vector<GLfloat>>::iterator v_obj = obj.begin();
	while (v_obj != obj.end()) {
		std::vector<GLfloat> res = axis;
		GLfloat scalar_product = scalar_mult(*v_obj, axis);
		scale_3Dvectors(res, scalar_product);
		proj.push_back(res);
		v_obj++;
	}
}

float dist_sq_vec(std::vector<GLfloat>& p1, std::vector<GLfloat>& p2) {
	float x_sq = (float)(p2[0] - p1[0]) * (p2[0] - p1[0]);
	float y_sq = (float)(p2[1] - p1[1]) * (p2[1] - p1[1]);
	float z_sq = (float)(p2[2] - p1[2]) * (p2[2] - p1[2]);

	return x_sq + y_sq + z_sq;
}

void find_Min_Max_line(std::vector<std::vector<GLfloat>>& line, float& min, float& max) {
	std::vector<GLfloat> cannon_axis_cent = { 0, 0, 0 };
	float max_sq = 0.0;
	float min_sq = dist_sq_vec(cannon_axis_cent, line[0]);
	std::vector<std::vector<GLfloat>>::iterator v = line.begin();
	while (v != line.end()) {
		float v_dist_sq = dist_sq_vec(cannon_axis_cent, *v);
		if (v_dist_sq > max_sq) { 
			max_sq = v_dist_sq;
		}
		else if (v_dist_sq < min_sq) { 
			min_sq = v_dist_sq;
		}
	}
	min = min_sq;
	max = max_sq;
}
bool check_overlap_along_axis(std::vector<std::vector<GLfloat>>& obj1, 
	std::vector<std::vector<GLfloat>>& obj2, 
	std::vector<GLfloat>& axis) {

	float min_sq1, max_sq1, min_sq2, max_sq2;
	std::vector<std::vector<GLfloat>> proj1;
	std::vector<std::vector<GLfloat>> proj2;

	project_3D_along_axis(obj1, axis, proj1);
	project_3D_along_axis(obj2, axis, proj2);
	find_Min_Max_line(proj1, min_sq1, max_sq1);
	find_Min_Max_line(proj2, min_sq2, max_sq2);

	return min_sq1 <= max_sq2 && max_sq1 >= min_sq2;
}

bool detect_collision_CONVEX_vs_CONVEX_or_SPHERE(std::vector<std::vector<GLfloat>>& conv1,
	std::vector<std::vector<GLfloat>>& conv2) {

	std::vector<std::vector<GLfloat>> f_norm1;
	std::vector<std::vector<GLfloat>> f_norm2;
	f_normals(conv1, f_norm1, TRIANGLE_POL);
	f_normals(conv2, f_norm2, TRIANGLE_POL);

	std::vector<std::vector<GLfloat>> face_norms;
	face_norms.reserve(f_norm1.size() + f_norm2.size());
	face_norms.insert(face_norms.end(), f_norm1.begin(), f_norm1.end());
	face_norms.insert(face_norms.end(), f_norm2.begin(), f_norm2.end());

	std::vector<std::vector<GLfloat>>::iterator norm = face_norms.begin();
	while (norm != face_norms.begin()) {
		if (!check_overlap_along_axis(conv1, conv2, *norm)) { return false; }
	}
	return true;
}

class col_detect {
public:
	bool detect_BOX_BOX(std::vector<std::vector<GLfloat>>& col_mesh1, std::vector<std::vector<GLfloat>>& col_mesh2);
	bool detect_BOX_SPHERE(std::vector<std::vector<GLfloat>>& col_mesh1, std::vector<std::vector<GLfloat>>& col_mesh2);
	bool detect_SPHERE_SPHERE(std::vector<std::vector<GLfloat>>& col_mesh1, std::vector<std::vector<GLfloat>>& col_mesh2);
	bool detect_CONVEX_CONVEX_or_SPHERE(std::vector<std::vector<GLfloat>>& col_mesh1, std::vector<std::vector<GLfloat>>& col_mesh2);
};

bool col_detect::detect_BOX_BOX(std::vector<std::vector<GLfloat>>& col_mesh1, std::vector<std::vector<GLfloat>>& col_mesh2) {
	return detect_collision_BOX_vs_BOX(col_mesh1, col_mesh2);
}

bool col_detect::detect_BOX_SPHERE(std::vector<std::vector<GLfloat>>& col_mesh1, std::vector<std::vector<GLfloat>>& col_mesh2) {
	GLfloat* center_sphere = geometrig_center(col_mesh2);
	return detect_collision_BOX_vs_SPHERE(col_mesh1, col_mesh2, center_sphere);
}

bool col_detect::detect_SPHERE_SPHERE(std::vector<std::vector<GLfloat>>& col_mesh1, std::vector<std::vector<GLfloat>>& col_mesh2) {
	GLfloat* center_sphere1 = geometrig_center(col_mesh1);
	GLfloat* center_sphere2 = geometrig_center(col_mesh2);
	return detect_collision_SPHERE_vs_SPHERE(col_mesh1, col_mesh2, center_sphere1, center_sphere2);
}

bool col_detect::detect_CONVEX_CONVEX_or_SPHERE(std::vector<std::vector<GLfloat>>& col_mesh1, std::vector<std::vector<GLfloat>>& col_mesh2) {
	return detect_collision_CONVEX_vs_CONVEX_or_SPHERE(col_mesh1, col_mesh2);
}

/* ------------------------ define a rigid body ------------------------- */
typedef struct {
	// geometric data
	std::vector<std::vector<GLfloat>> bodyPos;
	std::vector<int> body_polygon_size;
	std::vector<std::vector<GLfloat>> hitBoxPos;
	std::vector<GLfloat> collisionPosition;
	std::vector<std::vector<GLfloat>> rotation_LEGACY_GL;

	// material information
	std::vector<std::string> materialMap;

	// mass distribution
	std::vector<GLfloat> massDistribution;
	GLfloat mass;
	std::vector<GLfloat> centerOfMass;

	// technical possebilities
	bool gravityApplied;
	bool isFullyElasticAndRigid;
	bool collision_allowed;
	int hitBoxType;
	bool collided;

	// force information
	std::vector<GLfloat> gravityForce;
	std::vector<std::vector<GLfloat>> collisionForces;
	std::vector<std::vector<GLfloat>> Force_distrib_radial;
	std::vector<std::vector<GLfloat>> Force_distrib_tangent;
	std::vector<GLfloat> torque;
	std::vector<GLfloat> staticFriction_Force;
	std::vector<GLfloat> kineticFriction_Force;

	// velocities and acceleration information
	std::vector<GLfloat> linearVelocity;
	std::vector<GLfloat> linearAcceleration;
	std::vector<std::vector<GLfloat>> linearVelocityElements;
	std::vector<GLfloat> angularVelocity;
	std::vector<GLfloat> angularAcceleration;
	std::vector<std::vector<GLfloat>> tangentVelocityElements;

	// momentum and inertia information
	std::vector<GLfloat> linearMomentum;
	std::vector<GLfloat> angularMomentum;
	std::vector<std::vector<GLfloat>> inertiaTensor;
	std::vector<std::vector<GLfloat>> inertiaTensorRotation;
}Rigid_body;

/* ------------------------ basic physics functions - simulate a force on a rigid body ---------------------------*/

// structure that defines the physical parameters of a rigid body - at a specific time!
// for continuos simulation - this structure is used to describe a rigid body at a moment, 
// and to parameters change through time

void default_RigidMass_distribution(Rigid_body* rigidBody) {

	std::vector<GLfloat> mass_distrib;
	std::vector<std::vector<GLfloat>> bodyPos = rigidBody->bodyPos;
	std::vector<std::vector<GLfloat>>::iterator v = bodyPos.begin();
	GLuint mesh_size = (GLuint)bodyPos.size();

	while (v != bodyPos.end()) {
		mass_distrib.push_back((GLfloat)(1 / (GLfloat)mesh_size));
		v++;
	}
	rigidBody->massDistribution = mass_distrib;
}

void modify_RigidMass_distribution(Rigid_body* rigidBody, std::vector<GLfloat>& distrib) {
	std::vector<std::vector<GLfloat>> pos_mesh = rigidBody->bodyPos;
	assert(pos_mesh.size() == distrib.size());
	rigidBody->massDistribution = distrib;
}

void calculate_center_mass(Rigid_body* rigidBody) {
	std::vector<GLfloat> mass_distrib = rigidBody->massDistribution;
	std::vector<std::vector<GLfloat>> mesh = rigidBody->bodyPos;
	std::vector<GLfloat> center = { 0.0, 0.0, 0.0 };

	int mesh_size = (int)mesh.size();
	for (int i = 0; i < mesh_size; i++) {
		std::vector<GLfloat> v = mesh[i];
		scale_3Dvectors(v, mass_distrib[i]);
		add_3Dvectors(center, v);
	}
	rigidBody->centerOfMass = center;
}

// inertia calculation

void calc_mass_elem_vec(std::vector<GLfloat>& mass_distrib, GLfloat mass, std::vector<GLfloat>& mass_elems) {

	std::vector<GLfloat> temp_mass_elems;
	std::vector<GLfloat> mass_dist = mass_distrib;

	std::vector<GLfloat>::iterator prob = mass_dist.begin();
	while (prob != mass_dist.end()) {
		GLfloat prob_coeff = *prob;
		temp_mass_elems.push_back(mass * prob_coeff);
	}

	mass_elems = temp_mass_elems;
}

GLfloat calc_I_by_coords(GLfloat mass_elem, std::vector<GLfloat>& v, GLint c[3]) {

	// Ixx
	if (c[0] == 2) {
		return mass_elem * (v[1] * v[1] + v[2] * v[2]);
	}
	// Ixy = Iyx
	else if (c[0] == 1 && c[1] == 1) {
		return -mass_elem * v[0] * v[1];
	}
	// Ixz = Izx
	else if (c[0] == 1 && c[2] == 1) {
		return -mass_elem * v[0] * v[2];
	}
	// Iyy
	else if (c[1] == 2) {
		return mass_elem * (v[0] * v[0] + v[2] * v[2]);
	}
	// Iyz = Izy
	else if (c[1] == 1 && c[2] == 1) {
		return -mass_elem * v[1] * v[2];
	}
	// Izz
	else if (c[2] == 2) {
		return mass_elem * (v[0] * v[0] + v[1] * v[1]);
	}
	return -1;
}

GLfloat calc_I_elem(Rigid_body* rigidBody, GLint coords[3]) {
	std::vector<std::vector<GLfloat>> bodyPos = rigidBody->bodyPos;
	std::vector<GLfloat> mass_distrib = rigidBody->massDistribution;
	std::vector<GLfloat> c_o_m = rigidBody->centerOfMass;
	scale_3Dvectors(c_o_m, -1.0);
	std::vector<GLfloat> mass_elems;
	GLfloat mass = rigidBody->mass;
	GLfloat I = 0.0;

	calc_mass_elem_vec(mass_distrib, mass, mass_elems);
	for (int i = 0; i < (int)bodyPos.size(); i++) {
		std::vector<GLfloat> v = bodyPos[i];
		// relative to center of mass
		add_3Dvectors(v, c_o_m);
		I += calc_I_by_coords(mass_elems[i], v, coords);
	}
	return I;
}

void calc_Inertia_tensor(Rigid_body* rigidBody) {
	std::vector<std::vector<GLfloat>> inerTensor(3, std::vector<GLfloat>(3, 0));

	GLint Ixx[3] =    { 2, 0, 0 };
	GLint Ixy_yx[3] = { 1, 1, 0 };
	GLint Ixz_zx[3] = { 1, 0, 1 };
	GLint Iyy[3] =    { 0, 2, 0 };
	GLint Iyz_zy[3] = { 0, 1, 1 };
	GLint Izz[3] =    { 0, 0, 2 };

	inerTensor[0][0] = calc_I_elem(rigidBody, Ixx);
	inerTensor[0][1] = calc_I_elem(rigidBody, Ixy_yx);
	inerTensor[1][0] = calc_I_elem(rigidBody, Ixy_yx);
	inerTensor[0][2] = calc_I_elem(rigidBody, Ixz_zx);
	inerTensor[2][0] = calc_I_elem(rigidBody, Ixz_zx);
	inerTensor[1][1] = calc_I_elem(rigidBody, Iyy);
	inerTensor[1][2] = calc_I_elem(rigidBody, Iyz_zy);
	inerTensor[2][1] = calc_I_elem(rigidBody, Iyz_zy);
	inerTensor[2][2] = calc_I_elem(rigidBody, Izz);

	rigidBody->inertiaTensor = inerTensor;
}

// for square matrixes
GLfloat* mult_3D_mat_vec(std::vector<std::vector<GLfloat>>& mat, std::vector<GLfloat>& vec) {
	assert(vec.size() == mat[0].size());
	std::vector<GLfloat> res;
	for (int row = 0; row < (int)mat.size(); row++) {
		GLfloat row_res = 0.0;
		for (int col = 0; col < (int)mat[row].size(); col++) {
			row_res += mat[row][col] * vec[col];
		}
		res.push_back(row_res);
	}
	GLfloat _r_res[3];
	std::copy(res.begin(), res.end(), _r_res);
	return _r_res;
}

bool calc_3D_inverse_mat(std::vector<std::vector<GLfloat>>& A, std::vector<std::vector<GLfloat>>& I_A) {
	GLfloat d = 0.0;
	std::vector<std::vector<GLfloat>> temp_I_A;
	// finding determinant
	for (int i = 0; i < 3; i++) {
		d += A[0][i] * (A[1][(i + 1) % 3] * A[2][(i + 2) % 3] - A[1][(i + 2) % 3] * A[2][(i + 1) % 3]);
	}
	if (d == 0.0) {
		return false;
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++)
			temp_I_A[i].push_back(((A[(j + 1) % 3][(i + 1) % 3] * A[(j + 2) % 3][(i + 2) % 3])
				- (A[(j + 1) % 3][(i + 2) % 3] * A[(j + 2) % 3][(i + 1) % 3])) / d);
	}
	I_A = temp_I_A;
	return true;
}

void calc_AngularMomentum_from_InertiaTensor(Rigid_body* rigidBody) {
	std::vector<std::vector<GLfloat>> I = rigidBody->inertiaTensor;
	std::vector<GLfloat> omega = rigidBody->angularVelocity;
	GLfloat* L_wrld_arr = mult_3D_mat_vec(I, omega);
	std::vector<GLfloat> L_wrld(L_wrld_arr, L_wrld_arr + 3);
	rigidBody->angularMomentum = L_wrld;
}

void calc_AngularVelocity_from_InertiaTensor(Rigid_body* rigidBody) {
	std::vector<std::vector<GLfloat>> Inertia_tensor = rigidBody->inertiaTensor;
	std::vector<GLfloat> L_wrld = rigidBody->angularMomentum;
	std::vector<std::vector<GLfloat>> inverse_Inertia_tensor;
	calc_3D_inverse_mat(Inertia_tensor, inverse_Inertia_tensor);
	GLfloat* omega_body = mult_3D_mat_vec(inverse_Inertia_tensor, L_wrld);
	std::vector<GLfloat> omega(omega_body, omega_body + 3);
	rigidBody->angularVelocity = omega;
}

// velocity calculations from pure forces

void init_force(Rigid_body* rigidBody) {
	// size(rad_f) == size(tan_f) == size(bodyPos)
	std::vector<std::vector<GLfloat>> rad_f;
	std::vector<std::vector<GLfloat>> tan_f;
	std::vector<std::vector<GLfloat>> body = rigidBody->bodyPos;
	std::vector<std::vector<GLfloat>>::iterator v_f = body.begin();
	std::vector<GLfloat> init_v = { 0.0, 0.0, 0.0 };
	while (v_f != body.end()) {
		rad_f.push_back(init_v);
		tan_f.push_back(init_v);
		v_f++;
	}
	rigidBody->Force_distrib_radial = rad_f;
	rigidBody->Force_distrib_tangent = tan_f;
}

// analysis and decomposition of a single force applied to the body
// describes the distribution of the force on the elements of the body

void find_poly_interval(std::vector<int> polygon, int poly, int& min_ind, int& max_ind) {
	int polygon_size = polygon[poly];
	int min_body_pos = 0;
	int max_body_pos = 0;
	int i = 0;
	while (i < (int)polygon.size()) {
		if (i == poly) { break; }
		min_body_pos += polygon[i];
		i++;
	}
	max_body_pos = min_body_pos + polygon_size - 1;
	min_ind = min_body_pos;
	max_ind = max_body_pos;
}

void distrib_force_to_polygon(std::vector<GLfloat>& force,
	Rigid_body* rigidBody,
	int min_poly,
	int closest_point,
	std::vector<std::vector<GLfloat>>& force_distrib) {

	std::vector<std::vector<GLfloat>> f_dist;
	std::vector<int> poly = rigidBody->body_polygon_size;
	std::vector<std::vector<GLfloat>> body = rigidBody->bodyPos;
	int min_ind, max_ind;
	find_poly_interval(poly, min_poly, min_ind, max_ind);
	// the colsest point gets 60% of the force distributed to, the other points get equal distribution
	GLfloat cl_scale = 0.6;
	GLfloat surr_scale = 0.4 / (GLfloat)(max_ind - min_ind + 1);
	int i = 0;
	while (i < (int)body.size()) {
		if (i >= min_ind && i <= max_ind) {
			std::vector<GLfloat> v = force;
			if (i == closest_point) { scale_3Dvectors(v, cl_scale); }
			else{ scale_3Dvectors(v, surr_scale); }
			f_dist.push_back(v);
		}
		else { f_dist.push_back({ 0.0, 0.0, 0.0 }); }
	}
	force_distrib = f_dist;

}

void distrib_force_to_mass_elems(Rigid_body* rigidBody,
	std::vector<GLfloat>& init_force_pt,
	std::vector<GLfloat>& force,
	std::vector<std::vector<GLfloat>>& force_distrib) {

	std::vector<std::vector<GLfloat>> body = rigidBody->bodyPos;
	std::vector<int> poly = rigidBody->body_polygon_size;
	int j = 0;
	int lim = poly[0];
	float min_dist_sq = dist_sq_vec(init_force_pt, body[0]);
	int min_poly = 0;
	int closest = 0;

	for (int i = 0; i < (int)poly.size(); i++) {
		while (j < lim) {
			float sq_dist = dist_sq_vec(init_force_pt, body[j]);
			if (sq_dist < min_dist_sq) {
				min_dist_sq = sq_dist;
				min_poly = i;
				closest = j;
			}
			j++;
		}
		if (i < (int)poly.size() - 1) { lim += poly[i + 1]; }
	}
	distrib_force_to_polygon(force, rigidBody, min_poly, closest, force_distrib);
}

// projection of the force onto center mass axis - for calculating the tangent axis
GLfloat* create_CenterMass_axis(Rigid_body* rigidBody, std::vector<GLfloat>& init_force_pt, std::vector<GLfloat>& force) {
	std::vector<GLfloat> c_m_axis = rigidBody->centerOfMass;
	std::vector<GLfloat> init_f = init_force_pt;
	scale_3Dvectors(init_f, -1.0);
	add_3Dvectors(c_m_axis, init_f);
	normalize_vec(c_m_axis);
	
	// project force onto center mass axis
	GLfloat scalar_prod = scalar_mult(force, c_m_axis);
	scale_3Dvectors(c_m_axis, scalar_prod);

	GLfloat cm_axis[3];
	std::copy(c_m_axis.begin(), c_m_axis.end(), cm_axis);
	return cm_axis;
}

// projection of the force onto the tangent axis
GLfloat* create_tangent_axis(GLfloat* cm_axis, std::vector<GLfloat>& force) {
	std::vector<GLfloat> c_m_axis(cm_axis, cm_axis + 3);
	std::vector<GLfloat> tang_axis = force;
	scale_3Dvectors(c_m_axis, -1);
	add_3Dvectors(tang_axis, c_m_axis);

	GLfloat tan_axis[3];
	std::copy(tang_axis.begin(), tang_axis.end(), tan_axis);
	return tan_axis;
}

void create_force_axis(Rigid_body* rigidBody,
	std::vector<GLfloat>& init_force_pt,
	std::vector<GLfloat>& force, 
	std::vector<GLfloat>& center_mass_axis,
	std::vector<GLfloat>& tangent_axis) {

	GLfloat* cm_axis = create_CenterMass_axis(rigidBody, init_force_pt, force);
	GLfloat* tang_axis = create_tangent_axis(cm_axis, force);
	std::vector<GLfloat> c_m_axis(cm_axis, cm_axis + 3);
	std::vector<GLfloat> tan_axis(tang_axis, tang_axis + 3);

	center_mass_axis = c_m_axis;
	tangent_axis = tan_axis;
}

void radial_tangent_decomposition(Rigid_body* rigidBody,
	std::vector<GLfloat>& init_force_pt,
	std::vector<GLfloat>& force,
	std::vector<std::vector<GLfloat>>& force_distrib) {

	// size(force_distribution) == size(rad_f_dist) == size(tan_f_dist) == size(bodyPos)
	std::vector<std::vector<GLfloat>> rad_f = rigidBody->Force_distrib_radial;
	std::vector<std::vector<GLfloat>> tan_f = rigidBody->Force_distrib_tangent;
	std::vector<GLfloat> c_m_axis;
	std::vector<GLfloat> tan_axis;
	int i = 0;
	while (i < (int)force_distrib.size()) {
		create_force_axis(rigidBody, init_force_pt, force_distrib[i], c_m_axis, tan_axis);
		add_3Dvectors(rad_f[i], c_m_axis);
		add_3Dvectors(tan_f[i], tan_axis);
	}
	rigidBody->Force_distrib_radial = rad_f;
	rigidBody->Force_distrib_tangent = tan_f;
}

// force at center mass - average radial force
void average_radial_force(Rigid_body* rigidBody) {
	std::vector<GLfloat> avg_rad_force = { 0, 0, 0 };
	std::vector<std::vector<GLfloat>> avg_rad_dist;
	std::vector<std::vector<GLfloat>> rad_distrib = rigidBody->Force_distrib_radial;
	std::vector<std::vector<GLfloat>>::iterator rad_v = rad_distrib.begin();
	int size = (int)rad_distrib.size();
	while (rad_v != rad_distrib.end()) {
		add_3Dvectors(avg_rad_force, *rad_v);
	}
	scale_3Dvectors(avg_rad_force, (GLfloat)size);
	int i = 0;
	while (i < size) {
		avg_rad_dist.push_back(avg_rad_force);
		i++;
	}
	rigidBody->Force_distrib_radial = avg_rad_dist;
}

// distributes the force, and divides into radial force and tangent force
void apply_force(Rigid_body* rigidBody, std::vector<GLfloat>& init_force_pt, std::vector<GLfloat>& force) {
	std::vector<std::vector<GLfloat>> force_distrib;
	distrib_force_to_mass_elems(rigidBody, init_force_pt, force, force_distrib);
	radial_tangent_decomposition(rigidBody, init_force_pt, force, force_distrib);
	average_radial_force(rigidBody);
}

// convert forces into aacelerations and velocities

// at a specific time stamp
void newton_linear_second_law(Rigid_body* rigid, float time_stamp) {
	// all filled with the average radial force
	std::vector<std::vector<GLfloat>> rad_distrib = rigid->Force_distrib_radial;
	std::vector<GLfloat> F = rad_distrib[0];
	std::vector<GLfloat> v = rigid->linearVelocity;
	GLfloat mass = rigid->mass;
	std::vector<GLfloat> a = F;
	scale_3Dvectors(a, (GLfloat)(1 / mass));
	rigid->linearAcceleration = a;

	std::vector<GLfloat> acc = a;
	scale_3Dvectors(acc, (GLfloat)time_stamp);
	add_3Dvectors(v, acc);
	rigid->linearVelocity = v;
}

void calc_linear_velocity_elems(Rigid_body* rigid, float time_stamp) {
	std::vector<std::vector<GLfloat>> rad_distrib = rigid->Force_distrib_radial;
	std::vector<std::vector<GLfloat>> v_elems = rigid->linearVelocityElements;
	std::vector<GLfloat> mass_d = rigid->massDistribution;
	GLfloat mass = rigid->mass;
	std::vector<GLfloat> mass_elems;
	calc_mass_elem_vec(mass_d, mass, mass_elems);
	int r_s = (int)rad_distrib.size();
	for (int k = 0; k < r_s; k++) {
		std::vector<GLfloat> a = rad_distrib[k];
		scale_3Dvectors(a, (GLfloat)(1 / mass_elems[k]));
		scale_3Dvectors(a, (GLfloat)time_stamp);
		add_3Dvectors(v_elems[k], a);
	}
	rigid->linearVelocityElements = v_elems;
}

void init_torque(Rigid_body* rigid) {
	rigid->torque = { 0, 0, 0 };
}

void calc_torque(Rigid_body* rigid) {
	std::vector<GLfloat> torqueV = rigid->torque;
	std::vector<std::vector<GLfloat>> tangent_d = rigid->Force_distrib_tangent;
	std::vector<std::vector<GLfloat>> body = rigid->bodyPos;
	std::vector<GLfloat> c_o_m = rigid->centerOfMass;
	scale_3Dvectors(c_o_m, -1.0);
	int i = 0;
	while (i < (int)body.size()) {
		std::vector<GLfloat> torque_element;
		std::vector<GLfloat> r = body[i];
		add_3Dvectors(r, c_o_m);
		cross_product(r, tangent_d[i], torque_element);
		add_3Dvectors(torqueV, torque_element);

		i++;
	}
	rigid->torque = torqueV;
}

void calc_angularAcceleration(Rigid_body* rigid) {
	std::vector<GLfloat> torqueVec = rigid->torque;
	std::vector<std::vector<GLfloat>> I = rigid->inertiaTensor;
	std::vector<std::vector<GLfloat>> I_inv;
	calc_3D_inverse_mat(I, I_inv);
	GLfloat* ang_acc = mult_3D_mat_vec(I_inv, torqueVec);
	std::vector<GLfloat> angularAcc(ang_acc, ang_acc + 3);
	rigid->angularAcceleration = angularAcc;
}

void calc_angularVelocity_from_acc(Rigid_body* rigid, float time_stamp) {
	std::vector<GLfloat> omega = rigid->angularVelocity;
	std::vector<GLfloat> alpha = rigid->angularAcceleration;
	scale_3Dvectors(alpha, (GLfloat)time_stamp);
	add_3Dvectors(omega, alpha);
	rigid->angularVelocity = omega;
}

void calc_angular_velocity_elems(Rigid_body* rigid, float time_stamp) {

	std::vector<std::vector<GLfloat>> v_elems = rigid->tangentVelocityElements;
	std::vector<GLfloat> alpha = rigid->angularAcceleration;
	std::vector<std::vector<GLfloat>> body = rigid->bodyPos;
	std::vector<GLfloat> center = rigid->centerOfMass;
	scale_3Dvectors(center, -1.0);
	int size = (int)body.size();
	for (int i = 0; i < size; i++) {
		std::vector<GLfloat> acc_elem;
		std::vector<GLfloat> vec_elem = body[i];
		add_3Dvectors(vec_elem, center);
		cross_product(alpha, vec_elem, acc_elem);
		scale_3Dvectors(acc_elem, (GLfloat)time_stamp);
		add_3Dvectors(v_elems[i], acc_elem);
	}
	rigid->tangentVelocityElements = v_elems;
}

void linear_position_update(Rigid_body* rigid, float time_stamp) {

	std::vector<GLfloat> linear_v = rigid->linearVelocity;
	std::vector<std::vector<GLfloat>> temp_pos;
	std::vector<std::vector<GLfloat>> temp_Hpos; 
	std::vector<std::vector<GLfloat>> position = rigid->bodyPos;
	std::vector<std::vector<GLfloat>> hitBoxPosition = rigid->hitBoxPos;
	std::vector<std::vector<GLfloat>>::iterator pos = position.begin();
	int i = 0;
	while (pos != position.end()) {
		std::vector<GLfloat> v = linear_v;
		std::vector<GLfloat> p = *pos;
		std::vector<GLfloat> h_p = hitBoxPosition[i];
		scale_3Dvectors(v, (GLfloat)time_stamp);
		add_3Dvectors(p, v);
		add_3Dvectors(h_p, v);
		temp_pos.push_back(p);
		temp_Hpos.push_back(h_p);
		pos++; i++;
	}
	rigid->bodyPos = temp_pos;
	rigid->hitBoxPos = temp_Hpos;
}

void rotate_vec_along_axis(std::vector<GLfloat>& v,
	GLfloat rot_angle,
	std::vector<GLfloat>& rot_axis,
	std::vector<GLfloat>& res) {

	std::vector<GLfloat> rot__axis = rot_axis;
	normalize_vec(rot__axis);

	std::vector<GLfloat> temp_res = { 0, 0, 0 };
	std::vector<GLfloat> temp_v1 = v;
	std::vector<GLfloat> temp_v2 = rot__axis;
	std::vector<GLfloat> temp_v3;

	scale_3Dvectors(temp_v1, (GLfloat)cos((long double)rot_angle));
	GLfloat sc_mult = scalar_mult(rot__axis, v);
	sc_mult *= 1.0 - (GLfloat)cos((long double)rot_angle);
	scale_3Dvectors(temp_v2, sc_mult);
	cross_product(rot__axis, v, temp_v3);
	scale_3Dvectors(temp_v3, (GLfloat)sin((long double)rot_angle));

	add_3Dvectors(temp_res, temp_v1);
	add_3Dvectors(temp_res, temp_v2);
	add_3Dvectors(temp_res, temp_v3);
	res = temp_res;
}

float approx_dist_vec(std::vector<GLfloat>& v) {
	return fast_sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void angular_position_update(Rigid_body* rigid, float time_stamp){

	std::vector<GLfloat> angular_v = rigid->angularVelocity;
	GLfloat theta_t = (GLfloat)(approx_dist_vec(angular_v) * time_stamp);
	std::vector<std::vector<GLfloat>> temp_pos;
	std::vector<std::vector<GLfloat>> temp_Hpos;
	std::vector<std::vector<GLfloat>> position = rigid->bodyPos;
	std::vector<std::vector<GLfloat>> hitBoxPosition = rigid->hitBoxPos;
	std::vector<std::vector<GLfloat>>::iterator pos = position.begin();
	int i = 0;
	while (pos != position.end()) {
		std::vector<GLfloat> p = *pos;
		std::vector<GLfloat> h_p = hitBoxPosition[i];
		std::vector<GLfloat> rot_pos;
		std::vector<GLfloat> rot_Hpos;
		rotate_vec_along_axis(p, theta_t, angular_v, rot_pos);
		rotate_vec_along_axis(h_p, theta_t, angular_v, rot_Hpos);
		temp_pos.push_back(rot_pos);
		temp_Hpos.push_back(rot_Hpos);
		pos++; i++;
	}
	rigid->bodyPos = temp_pos;
	rigid->hitBoxPos = temp_Hpos;
}

void apply_force_update_position(Rigid_body* rigid,
	std::vector<GLfloat>& init_force_pt,
	std::vector<GLfloat>& force,
	float time_stamp) {

	apply_force(rigid, init_force_pt, force);
	newton_linear_second_law(rigid, time_stamp);
	calc_torque(rigid);
	calc_angularAcceleration(rigid);
	calc_angularVelocity_from_acc(rigid, time_stamp);
	linear_position_update(rigid, time_stamp);
	angular_position_update(rigid, time_stamp);
}

// TODO PLANS: different body types - elastic body (not rigid), fluid simulation, cloth simulation - seperate library
// TODO PLANS: create forces, collision physics, generate new objects - for rigid body
// gyroscope and top dynamics should be added

/*----------------------------collision physics - simulate a force on a rigid body------------------------------*/

void check_approx_coll(Rigid_body* rigid1, Rigid_body* rigid2, std::vector<GLfloat>& col_pt) {

	std::vector<GLfloat> tmp_colpt = { 0, 0, 0 };
	std::vector<std::vector<GLfloat>> hitMesh1 = rigid1->hitBoxPos;
	std::vector<std::vector<GLfloat>> hitMesh2 = rigid2->hitBoxPos;
	float min_dist1, min_dist2, max_dist1, max_dist2;

	int imp_i, imp_j;

	std::vector<GLfloat> c_wrld = { 0, 0, 0 };
	std::vector<GLfloat> c_m1 = rigid1->centerOfMass;
	std::vector<GLfloat> c_m2 = rigid2->centerOfMass;
	float wrld_dist_sq1 = dist_sq_vec(c_wrld, c_m1);
	float wrld_dist_sq2 = dist_sq_vec(c_wrld, c_m2);

	if (wrld_dist_sq2 >= wrld_dist_sq1) {
		max_dist1 = 0.0;
		min_dist2 = wrld_dist_sq2;
		// check for maximum distance from the origin
		for (int i = 0; i < (int)hitMesh1.size(); i++) {
			if (max_dist1 < dist_sq_vec(c_wrld, hitMesh1[i])) {
				max_dist1 = dist_sq_vec(c_wrld, hitMesh1[i]);
				imp_i = i;
			}
		}
		// check for minimum distance from the origin
		for (int j = 0; j < (int)hitMesh2.size(); j++) {
			if (min_dist2 > dist_sq_vec(c_wrld, hitMesh2[j])) {
				min_dist2 = dist_sq_vec(c_wrld, hitMesh2[j]);
				imp_j = j;
			}
		}
	}
	else if (wrld_dist_sq2 < wrld_dist_sq1) {
		max_dist2 = 0.0;
		min_dist1 = wrld_dist_sq1;
		// check for maximum distance from the origin
		for (int j = 0; j < (int)hitMesh2.size(); j++) {
			if (max_dist2 < dist_sq_vec(c_wrld, hitMesh2[j])) {
				max_dist2 = dist_sq_vec(c_wrld, hitMesh2[j]);
				imp_j = j;
			}
		}
		// check for minimum distance from the origin
		for (int i = 0; i < (int)hitMesh1.size(); i++) {
			if (min_dist1 > dist_sq_vec(c_wrld, hitMesh1[i])) {
				min_dist1 = dist_sq_vec(c_wrld, hitMesh1[i]);
				imp_i = i;
			}
		}
	}
	add_3Dvectors(tmp_colpt, hitMesh1[imp_i]);
	add_3Dvectors(tmp_colpt, hitMesh2[imp_j]);
	scale_3Dvectors(tmp_colpt, 0.5);

	col_pt = tmp_colpt;
}

void approx_collision_point(Rigid_body* rigid1, Rigid_body* rigid2) {

	int hit_box1 = rigid1->hitBoxType;
	int hit_box2 = rigid2->hitBoxType;

	col_detect col_obj;
	std::vector<std::vector<GLfloat>> hitMesh1 = rigid1->hitBoxPos;
	std::vector<std::vector<GLfloat>> hitMesh2 = rigid2->hitBoxPos;

	if (hit_box1 == BOUNDING_BOX && hit_box2 == BOUNDING_BOX) {
		if (col_obj.detect_BOX_BOX(hitMesh1, hitMesh2)) {
			rigid1->collided = true;
			rigid2->collided = true;
			std::vector<GLfloat> col_pt;
			check_approx_coll(rigid1, rigid2, col_pt); 
			rigid1->collisionPosition = col_pt;
			rigid2->collisionPosition = col_pt;
		}
	}
	else if ((hit_box1 == BOUNDING_BOX && hit_box2 == BOUNDING_SPHERE) ||
		(hit_box2 == BOUNDING_BOX && hit_box1 == BOUNDING_SPHERE)) {
		if (col_obj.detect_BOX_SPHERE(hitMesh1, hitMesh2)) {
			rigid1->collided = true;
			rigid2->collided = true;
			std::vector<GLfloat> col_pt;
			check_approx_coll(rigid1, rigid2, col_pt);
			rigid1->collisionPosition = col_pt;
			rigid2->collisionPosition = col_pt;
		}
	}
	else if (hit_box1 == BOUNDING_SPHERE && hit_box2 == BOUNDING_SPHERE) {
		if (col_obj.detect_SPHERE_SPHERE(hitMesh1, hitMesh2)) {
			rigid1->collided = true;
			rigid2->collided = true;
			std::vector<GLfloat> col_pt;
			check_approx_coll(rigid1, rigid2, col_pt);
			rigid1->collisionPosition = col_pt;
			rigid2->collisionPosition = col_pt;
		}
	}
	else if ((hit_box1 == BOUNDING_CONVEX && hit_box2 == BOUNDING_CONVEX) ||
		(hit_box1 == BOUNDING_CONVEX && hit_box2 == BOUNDING_BOX) ||
		(hit_box2 == BOUNDING_CONVEX && hit_box1 == BOUNDING_BOX) || 
		(hit_box1 == BOUNDING_CONVEX && hit_box2 == BOUNDING_SPHERE) || 
		(hit_box2 == BOUNDING_CONVEX && hit_box1 == BOUNDING_SPHERE) || 
		(hit_box1 == SUB_MESH && hit_box2 == SUB_MESH) || 
		(hit_box1 == SUB_MESH && hit_box2 == BOUNDING_CONVEX) || 
		(hit_box2 == SUB_MESH && hit_box1 == BOUNDING_CONVEX)) {
		if (col_obj.detect_CONVEX_CONVEX_or_SPHERE(hitMesh1, hitMesh2)) {
			rigid1->collided = true;
			rigid2->collided = true;
			std::vector<GLfloat> col_pt;
			check_approx_coll(rigid1, rigid2, col_pt);
			rigid1->collisionPosition = col_pt;
			rigid2->collisionPosition = col_pt;
		}
	}
	else if ((hit_box1 == SUB_MESH && hit_box2 == BOUNDING_BOX) || 
		(hit_box2 == SUB_MESH && hit_box1 == BOUNDING_BOX) || 
		(hit_box1 == SUB_MESH && hit_box2 == BOUNDING_SPHERE) || 
		(hit_box2 == SUB_MESH && hit_box1 == BOUNDING_SPHERE)) {
		if (col_obj.detect_CONVEX_CONVEX_or_SPHERE(hitMesh1, hitMesh2)) {
			rigid1->collided = true;
			rigid2->collided = true;
			std::vector<GLfloat> col_pt;
			check_approx_coll(rigid1, rigid2, col_pt);
			rigid1->collisionPosition = col_pt;
			rigid2->collisionPosition = col_pt;
		}
	}
}
