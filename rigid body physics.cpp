#include "glew.h"
#include <windows.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <GL/gl.h>
#include <GL/glu.h>
#include "GL/glut.h"
#include <stdlib.h>
#include <string>
#include <math.h>
#include <random>
#include <assert.h>
#include "model_draw.h"
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

#define START_FORCE 0
#define MID_FORCE 1

#define GJK_EPSILON 1.19209290E-02F
#define IMPACT_CRITICAL_VELOCITY 5.0F
#define G 9.81F
#define DT 1E-03F

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
	   -X + x_0, y_0, Z + z_0,
	   X + x_0, y_0, Z + z_0,
	   -X + x_0, y_0, -Z + z_0,
	   X + x_0, y_0, -Z + z_0,
	   x_0, Z + y_0, X + z_0,
	   x_0, Z + y_0, -X + z_0,
	   x_0, -Z + y_0, X + z_0,
	   x_0, -Z + y_0, -X + z_0,
	   Z + x_0, X + y_0, z_0,
	   -Z + x_0, X + y_0, z_0,
	   Z + x_0, -X + y_0, z_0,
	   -Z + x_0, -X + y_0, z_0
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
	int iters = num > 35 ? 2 : 3;

	int i = 0;
	while (i < iters) {
		x = (x * x * x + 3 * num * x) / (3 * x * x + num);
		i++;
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

	GLfloat vdata[12][3] = {
	   {-X + x_0, y_0, Z + z_0},
	   {X + x_0, y_0, Z + z_0},
	   {-X + x_0, y_0, -Z + z_0},
	   {X + x_0, y_0, -Z + z_0},
	   {x_0, Z + y_0, X + z_0},
	   {x_0, Z + y_0, -X + z_0},
	   {x_0, -Z + y_0, X + z_0},
	   {x_0, -Z + y_0, -X + z_0},
	   {Z + x_0, X + y_0, z_0},
	   {-Z + x_0, X + y_0, z_0},
	   {Z + x_0, -X + y_0, z_0},
	   {-Z + x_0, -X + y_0, z_0}
	};
	GLuint indices[20][3] = {
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

	GLfloat* geomCenter = new GLfloat[3];
	std::copy(center.begin(), center.end(), geomCenter);

	return geomCenter;
}

// detects if the bounding mesh is within the mesh or covering it. if it covers the mesh completely - return true
// else - return false
bool detect_boundries(std::vector<std::vector<GLfloat>>& model_mesh,
	std::vector<std::vector<GLfloat>>& bounding_mesh, GLfloat center[]) {

	std::vector<GLfloat> geomCenter(center, center + 3);
	scale_3Dvectors(geomCenter, -1.0);
	add_3Dvec_to_mesh(model_mesh, geomCenter);
	add_3Dvec_to_mesh(bounding_mesh, geomCenter);

	bool broke = false;
	for (int l1 = 0; l1 < (int)model_mesh.size(); l1++) {
		float d1 = fast_sqrt((float)(model_mesh[l1][0] * model_mesh[l1][0]) +
			(float)(model_mesh[l1][1] * model_mesh[l1][1]) +
			(float)(model_mesh[l1][2] * model_mesh[l1][2]));
		for (int l2 = 0; l2 < (int)bounding_mesh.size(); l2++) {
			float d2 = fast_sqrt((float)(bounding_mesh[l2][0] * bounding_mesh[l2][0]) +
				(float)(bounding_mesh[l2][1] * bounding_mesh[l2][1]) +
				(float)(bounding_mesh[l2][2] * bounding_mesh[l2][2]));
			if (d1 > d2) {
				broke = true;
				break;
			}
		}
		if (broke) {
			break;
		}
	}
	return not broke;
}

// needed to be tested further - so far, no bugs, just oddities...
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
	std::vector<std::vector<GLfloat>>              bodyPos;
	std::vector<int>                               body_polygon_size;
	std::vector<std::vector<GLfloat>>              hitBoxPos;
	std::vector<GLfloat>                           collisionPosition;
	std::vector<std::vector<GLfloat>>              rotation_LEGACY_GL;

	// material information
	std::vector<std::string>                       materialMap;

	// mass distribution
	std::vector<GLfloat>                           massDistribution;
	GLfloat                                        mass;
	std::vector<GLfloat>                           centerOfMass;

	// technical possebilities
	bool                                           gravityApplied;
	bool                                           isFullyElasticAndRigid;
	bool                                           collision_allowed;
	int                                            hitBoxType;
	bool                                           collided;

	// force information
	std::vector<std::vector<GLfloat>>              initPts;
	std::vector<GLfloat>                           CenterGravityForce;
	std::vector<std::vector<GLfloat>>              collisionForces;
	std::vector<std::vector<GLfloat>>              Force_distrib_radial;
	std::vector<std::vector<GLfloat>>              Force_distrib_tangent;
	std::vector<std::vector<std::vector<GLfloat>>> Force_distribContainer;
	std::vector<GLfloat>                           torque;
	std::vector<GLfloat>                           staticFriction_Force;
	std::vector<GLfloat>                           kineticFriction_Force;

	// velocities and acceleration information
	std::vector<GLfloat>                           linearVelocity;
	std::vector<GLfloat>                           linearAcceleration;
	std::vector<std::vector<GLfloat>>              linearVelocityElements;
	std::vector<GLfloat>                           angularVelocity;
	std::vector<GLfloat>                           angularAcceleration;
	std::vector<std::vector<GLfloat>>              tangentVelocityElements;

	// momentum and inertia information
	std::vector<GLfloat>                           linearMomentum;
	std::vector<GLfloat>                           angularMomentum;
	std::vector<std::vector<GLfloat>>              inertiaTensor;
	std::vector<std::vector<GLfloat>>              inertiaTensorRotation;
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
	GLfloat* _r_res = new GLfloat[3];
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

void find_init_pt_onBody(Rigid_body* rigidBody, std::vector<GLfloat>& init_force_pt, int& min_poly, int& closest) {

	std::vector<std::vector<GLfloat>> body = rigidBody->bodyPos;
	std::vector<int> poly = rigidBody->body_polygon_size;
	int j = 0;
	int lim = poly[0];
	float min_dist_sq = dist_sq_vec(init_force_pt, body[0]);

	int m = 0;
	int c = 0;
	for (int i = 0; i < (int)poly.size(); i++) {
		while (j < lim) {
			float sq_dist = dist_sq_vec(init_force_pt, body[j]);
			if (sq_dist < min_dist_sq) {
				min_dist_sq = sq_dist;
				m = i;
				c = j;
			}
			j++;
		}
		if (i < (int)poly.size() - 1) { lim += poly[i + 1]; }
	}

	min_poly = m;
	closest = c;
}

void distrib_force_to_mass_elems(Rigid_body* rigidBody,
	std::vector<GLfloat>& init_force_pt,
	std::vector<GLfloat>& force,
	std::vector<std::vector<GLfloat>>& force_distrib) {

	int min_poly = 0;
	int closest = 0;
	find_init_pt_onBody(rigidBody, init_force_pt, min_poly, closest);
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

	GLfloat* cm_axis = new GLfloat[3];
	std::copy(c_m_axis.begin(), c_m_axis.end(), cm_axis);
	return cm_axis;
}

// projection of the force onto the tangent axis
GLfloat* create_tangent_axis(GLfloat* cm_axis, std::vector<GLfloat>& force) {
	std::vector<GLfloat> c_m_axis(cm_axis, cm_axis + 3);
	std::vector<GLfloat> tang_axis = force;
	scale_3Dvectors(c_m_axis, -1);
	add_3Dvectors(tang_axis, c_m_axis);

	GLfloat* tan_axis = new GLfloat[3];
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
	std::vector<std::vector<GLfloat>>& force_distrib,
	std::vector<std::vector<GLfloat>>& r,
	std::vector<std::vector<GLfloat>>& t) {

	// size(force_distribution) == size(rad_f_dist) == size(tan_f_dist) == size(bodyPos)
	std::vector<std::vector<GLfloat>> body = rigidBody->bodyPos;
	std::vector<std::vector<GLfloat>> rad_f = rigidBody->Force_distrib_radial;
	std::vector<std::vector<GLfloat>> tan_f = rigidBody->Force_distrib_tangent;
	std::vector<GLfloat> c_m_axis;
	std::vector<GLfloat> tan_axis;
	int i = 0;
	while (i < (int)force_distrib.size()) {
		create_force_axis(rigidBody, body[i], force_distrib[i], c_m_axis, tan_axis);
		add_3Dvectors(rad_f[i], c_m_axis);
		add_3Dvectors(tan_f[i], tan_axis);
	}
	r = rad_f;
	t = tan_f;
}

// force at center mass - average radial force
void average_radial_force(Rigid_body* rigidBody, std::vector<std::vector<GLfloat>>& avg_r) {
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
	avg_r = avg_rad_dist;
}

// distributes the force, and divides into radial force and tangent force
void apply_force(Rigid_body* rigidBody, std::vector<GLfloat>& init_force_pt, std::vector<GLfloat>& force) {
	std::vector<std::vector<GLfloat>> force_distrib;
	std::vector<std::vector<GLfloat>> rad;
	std::vector<std::vector<GLfloat>> tang;
	distrib_force_to_mass_elems(rigidBody, init_force_pt, force, force_distrib);
	rigidBody->Force_distribContainer.push_back(force_distrib);
	rigidBody->initPts.push_back(init_force_pt);

	radial_tangent_decomposition(rigidBody, force_distrib, rad, tang);
	rigidBody->Force_distrib_radial = rad;

	average_radial_force(rigidBody, rad);
	rigidBody->Force_distrib_radial = rad; 
	rigidBody->Force_distrib_tangent = tang;

}

// removes the force when the force stoppes being applied to the body
void remove_force(Rigid_body* rigidBody,
	std::vector<GLfloat>& init_force_pt,
	std::vector<GLfloat>& force) {

	std::vector<std::vector<std::vector<GLfloat>>> force_distrib1 = rigidBody->Force_distribContainer;
	std::vector<std::vector<GLfloat>> force_distrib = force_distrib1[(int)force_distrib1.size() - 1];
	std::vector<std::vector<GLfloat>> r = rigidBody->Force_distrib_radial;
	std::vector<std::vector<GLfloat>> t = rigidBody->Force_distrib_tangent;

	std::vector<std::vector<GLfloat>> rad;
	std::vector<GLfloat> avg_r = { 0, 0, 0 };
	std::vector<std::vector<GLfloat>> tang;
	radial_tangent_decomposition(rigidBody, force_distrib, rad, tang);
	
	// recovering average radial
	int i = 0;
	while (i < (int)rad.size()) {
		add_3Dvectors(avg_r, rad[i]);
	}
	scale_3Dvectors(avg_r, (int)rad.size());

	// removing from radial distribution
	int i1 = 0;
	while (i1 < (int)r.size()) {
		subtr_3Dvectors(r[i1], avg_r);
	}

	// removing from the tangent force distribution
	int i2 = 0;
	while (i2 < (int)t.size()) {
		subtr_3Dvectors(t[i2], tang[i2]);
	}

	rigidBody->Force_distrib_radial = r;
	rigidBody->Force_distrib_tangent = t;

	std::vector<std::vector<std::vector<GLfloat>>> f = rigidBody->Force_distribContainer;
	std::vector<std::vector<GLfloat>> init = rigidBody->initPts;
	f.erase(f.end());
	init.erase(init.end());
	rigidBody->Force_distribContainer = f;
	rigidBody->initPts = init;

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

void init_VelocityElems(Rigid_body* rigidBody) {
	
	std::vector<std::vector<GLfloat>> v_e;
	std::vector<std::vector<GLfloat>> r_d = rigidBody->Force_distrib_radial;
	std::vector<std::vector<GLfloat>>::iterator v_f = r_d.begin();
	std::vector<GLfloat> init_v = { 0.0, 0.0, 0.0 };
	while (v_f != r_d.end()) {
		v_e.push_back(init_v);
		v_f++;
	}
	rigidBody->linearVelocityElements = v_e;
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

	std::vector<std::vector<GLfloat>> linear_v = rigid->linearVelocityElements;
	std::vector<std::vector<GLfloat>> temp_pos;
	std::vector<std::vector<GLfloat>> temp_Hpos; 
	std::vector<std::vector<GLfloat>> position = rigid->bodyPos;
	std::vector<std::vector<GLfloat>> hitBoxPosition = rigid->hitBoxPos;
	std::vector<std::vector<GLfloat>>::iterator pos = position.begin();
	int i = 0;
	while (pos != position.end()) {
		std::vector<GLfloat> v = linear_v[i];
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

// for immidiate force - time_stamp = 1;
void apply_force_update_position(Rigid_body* rigid,
	std::vector<GLfloat>& init_force_pt,
	std::vector<GLfloat>& force,
	float time_stamp) {

	apply_force(rigid, init_force_pt, force);
	newton_linear_second_law(rigid, time_stamp);
	calc_linear_velocity_elems(rigid, time_stamp);
	calc_Inertia_tensor(rigid);
	calc_torque(rigid);
	calc_angularAcceleration(rigid);
	calc_angularVelocity_from_acc(rigid, time_stamp);
	linear_position_update(rigid, time_stamp);
	angular_position_update(rigid, time_stamp);
}

// for one constant force working over time
void apply_continous_force(Rigid_body* rigid,
	std::vector<GLfloat>& init_force_pt,
	std::vector<GLfloat>& force_t,
	float time_steps,
	float dt) {

	int i = 0;
	while (i < time_steps) {
		apply_force_update_position(rigid, init_force_pt, force_t, dt);
		remove_force(rigid, init_force_pt, force_t);
		i++;
	}
}

// without forces / continous ones
void update_position(Rigid_body* rigid, float time_step) {

	//remove_force(rigidBo)
	newton_linear_second_law(rigid, time_step);
	calc_linear_velocity_elems(rigid, time_step);
	calc_Inertia_tensor(rigid);
	calc_torque(rigid);
	calc_angularAcceleration(rigid);
	calc_angularVelocity_from_acc(rigid, time_step);
	linear_position_update(rigid, time_step);
	angular_position_update(rigid, time_step);
}

// TODO PLANS: different body types - elastic body (not rigid), fluid simulation, cloth simulation - seperate library
// TODO PLANS: create forces, collision physics, generate new objects - for rigid body
// gyroscope and top dynamics should be added

/*----------------------------collision physics - simulate collision forces on rigid bodies------------------------------*/

// written like that for speed...
void subtr_3Dvectors(std::vector<GLfloat>& v_r, std::vector<GLfloat>& v_a) {
	v_r[0] -= v_a[0];
	v_r[1] -= v_a[1];
	v_r[2] -= v_a[2];
}

void add_toCont3Dvecs(std::vector<GLfloat>& a, std::vector<GLfloat>& b, std::vector<GLfloat>& c) {
	c[0] = 0.5 * (a[0] + b[0]);
	c[1] = 0.5 * (a[1] + b[1]);
	c[2] = 0.5 * (a[2] + b[2]);
}

// GJK algorithm for detecting collisions and finding the point of collision
void GJK_minkowski_diff(Rigid_body* rigid1, Rigid_body* rigid2, std::vector<std::vector<GLfloat>>& res) {

	std::vector<std::vector<GLfloat>> mesh1 = rigid1->hitBoxPos;
	std::vector<std::vector<GLfloat>> mesh2 = rigid2->hitBoxPos;
	std::vector<std::vector<GLfloat>> tmp_res;

	int i = 0;
	int j = 0;
	while (i < (int)mesh1.size()) {
		std::vector<GLfloat> i_r = mesh1[i];
		while (j < (int)mesh2.size()) {
			subtr_3Dvectors(i_r, mesh2[j]);
			tmp_res.push_back(i_r);
			j++;
		}
		i++;
	}
	res = tmp_res;
}

// not optimal at all but working, I hope...
// pt = argmin{ <axis,a> | for a in CSO }
void GJK_shortest_dst_pt(std::vector<std::vector<GLfloat>>& mink_diff,
	std::vector<GLfloat>& axis,
	std::vector<GLfloat>& pt) {

	float min_s_p = scalar_mult(mink_diff[0], axis);
	int min_ind = 0;
	for (int i = 0; i < (int)mink_diff.size(); i++) {
		float s_prod = scalar_mult(mink_diff[i], axis);
		if (min_s_p > s_prod) { min_s_p = s_prod; min_ind = i; }
	}
	pt = mink_diff[min_ind];
}

// the area between the lines is treated as a line itself so the current method suites also for 3D
// filled 2-simplexes and 3-simplexes (triangles and tetrahedrons)
void GJK_MinimumNormLine(std::vector<std::vector<GLfloat>>& simplex, std::vector<GLfloat>& pt) {

	float minNorm = scalar_mult(simplex[0], simplex[0]);
	int min_ind = 0;
	for (int i = 0; i < (int)simplex.size(); i++) {
		float s_prod = scalar_mult(simplex[i], simplex[i]);
		if (minNorm > s_prod) { minNorm = s_prod; min_ind = i; }
	}
	pt = simplex[min_ind];
}

typedef struct {
	std::vector<std::vector<GLfloat>> simplex;
	int simplexSize;
	std::vector<std::vector<std::vector<GLfloat>>> convexHullPoints;
	std::vector<GLfloat> minNormPoint;
	
}convexHull;

void createLine(std::vector<GLfloat>& i_p, std::vector<GLfloat>& e_p, std::vector<std::vector<GLfloat>>& line) {
	GLfloat step = 0.1;
	std::vector<GLfloat> pt = i_p;

	std::vector<GLfloat> v = e_p;
	subtr_3Dvectors(v, i_p);
	scale_3Dvectors(v, step);

	std::vector<std::vector<GLfloat>> tmp_CHP;
	tmp_CHP.push_back(pt);

	float dl = dist_sq_vec(i_p, e_p);
	float dpt = dist_sq_vec(i_p, pt);

	float diff = dl - dpt;
	while (diff > 0) {
		add_3Dvectors(pt, v);
		tmp_CHP.push_back(pt);

		dpt = dist_sq_vec(i_p, pt);
		diff = dl - dpt;
	}
	tmp_CHP.push_back(e_p);
	line = tmp_CHP;
}

// take each point from each line and take the average to fill "the void"
void fillAreaBetweenLines(std::vector<std::vector<GLfloat>>& l1,
	std::vector<std::vector<GLfloat>>& l2,
	std::vector<std::vector<GLfloat>>& l3,
	std::vector<std::vector<GLfloat>>& A) {

	std::vector<std::vector<GLfloat>> tmp;

	for (int i1 = 0; i1 < (int)l1.size(); i1++) {
		for (int i2 = 0; i2 < (int)l2.size(); i2++) {
			for (int i3 = 0; i3 < (int)l3.size(); i3++) {
				std::vector<GLfloat> avg1 = l1[i1];
				std::vector<GLfloat> avg2 = l2[i2];
				std::vector<GLfloat> avg3 = l3[i3];

				std::vector<GLfloat> o1, o2;
				std::vector<GLfloat> o3, o4;
				std::vector<GLfloat> o5, o6;

				add_toCont3Dvecs(avg1, avg2, o1);
				add_toCont3Dvecs(avg1, avg3, o2);
				add_toCont3Dvecs(avg2, avg1, o3);
				add_toCont3Dvecs(avg2, avg3, o4);
				add_toCont3Dvecs(avg3, avg1, o5);
				add_toCont3Dvecs(avg3, avg2, o6);

				tmp.push_back(o1);
				tmp.push_back(o2);
				tmp.push_back(o3);
				tmp.push_back(o4);
				tmp.push_back(o5);
				tmp.push_back(o6);
			}
		}
	}
	A = tmp;
}

void create_ConvexLine(convexHull* convHull) {
	assert(convHull->simplexSize == 2);
	std::vector<GLfloat> init_pt = convHull->simplex[0];
	std::vector<GLfloat> e_p = convHull->simplex[1];

	std::vector<std::vector<GLfloat>> tmp_CHP;
	createLine(init_pt, e_p, tmp_CHP);
	std::vector<std::vector<std::vector<GLfloat>>> CHP;
	CHP.push_back(tmp_CHP);
	convHull->convexHullPoints = CHP;
}

void createTriangle(std::vector<GLfloat>& i1,
	std::vector<GLfloat>& e1,
	std::vector<GLfloat>& i2,
	std::vector<GLfloat>& e2,
	std::vector<GLfloat>& i3,
	std::vector<GLfloat>& e3,
	std::vector<std::vector<std::vector<GLfloat>>>& tmp) {

	std::vector<std::vector<GLfloat>> tmp1;
	std::vector<std::vector<GLfloat>> tmp2;
	std::vector<std::vector<GLfloat>> tmp3;
	std::vector<std::vector<GLfloat>> A;
	createLine(i1, e1, tmp1);
	createLine(i2, e2, tmp2);
	createLine(i3, e3, tmp3);
	fillAreaBetweenLines(tmp1, tmp2, tmp3, A);

	std::vector<std::vector<std::vector<GLfloat>>> tmpTriangle;
	tmpTriangle.push_back(tmp1);
	tmpTriangle.push_back(tmp2);
	tmpTriangle.push_back(tmp3);
	tmpTriangle.push_back(A);

	tmp = tmpTriangle;
}

void create_ConvexTriangle(convexHull* convHull) {
	std::vector<GLfloat> i_p1 = convHull->simplex[0];
	std::vector<GLfloat> e_p1 = convHull->simplex[1];
	std::vector<GLfloat> i_p2 = e_p1;
	std::vector<GLfloat> e_p2 = convHull->simplex[2];
	std::vector<GLfloat> i_p3 = e_p2;
	std::vector<GLfloat> e_p3 = i_p1;

	std::vector<std::vector<std::vector<GLfloat>>> tmp;
	createTriangle(i_p1, e_p1, i_p2, e_p2, i_p3, e_p3, tmp);
	convHull->convexHullPoints = tmp;
}

void create_ConvexTetrahedron(convexHull* convHull) {

	// 4 faces == 4 triangles to create
	// map the points as 1,2,3,4 <-> 0,1,2,3
	// triangle [1] : 1->2->3
	std::vector<GLfloat> i11 = convHull->simplex[0]; // 1->2
	std::vector<GLfloat> e11 = convHull->simplex[1];
	std::vector<GLfloat> i21 = convHull->simplex[1]; // 2->3
	std::vector<GLfloat> e21 = convHull->simplex[2];
	std::vector<GLfloat> i31 = convHull->simplex[2]; // 3->1
	std::vector<GLfloat> e31 = convHull->simplex[0];
	std::vector<std::vector<std::vector<GLfloat>>> tmp1;
	createTriangle(i11, e11, i21, e21, i31, e31, tmp1);

	//triangle [2] : 1->3->4
	std::vector<GLfloat> i12 = convHull->simplex[0]; // 1->3
	std::vector<GLfloat> e12 = convHull->simplex[2]; 
	std::vector<GLfloat> i22 = convHull->simplex[2]; // 3->4
	std::vector<GLfloat> e22 = convHull->simplex[3];
	std::vector<GLfloat> i32 = convHull->simplex[3]; // 4->1
	std::vector<GLfloat> e32 = convHull->simplex[0];
	std::vector<std::vector<std::vector<GLfloat>>> tmp2;
	createTriangle(i12, e12, i22, e22, i32, e32, tmp2);

	// triangle [3] : 1->2->4
	std::vector<GLfloat> i13 = convHull->simplex[0]; // 1->2
	std::vector<GLfloat> e13 = convHull->simplex[1]; 
	std::vector<GLfloat> i23 = convHull->simplex[1]; // 2->4
	std::vector<GLfloat> e23 = convHull->simplex[3]; 
	std::vector<GLfloat> i33 = convHull->simplex[3]; // 4->1
	std::vector<GLfloat> e33 = convHull->simplex[0];
	std::vector<std::vector<std::vector<GLfloat>>> tmp3;
	createTriangle(i13, e13, i23, e23, i33, e33, tmp3);
	
	// triangle [4] : 2->3->4
	std::vector<GLfloat> i14 = convHull->simplex[1]; // 2->3
	std::vector<GLfloat> e14 = convHull->simplex[2];
	std::vector<GLfloat> i24 = convHull->simplex[2]; // 3->4
	std::vector<GLfloat> e24 = convHull->simplex[3];
	std::vector<GLfloat> i34 = convHull->simplex[3]; // 4->2
	std::vector<GLfloat> e34 = convHull->simplex[1];
	std::vector<std::vector<std::vector<GLfloat>>> tmp4;
	createTriangle(i14, e14, i24, e24, i34, e34, tmp4);

	std::vector<std::vector<std::vector<GLfloat>>> tmp;
	tmp.reserve(tmp1.size() + tmp2.size() + tmp3.size() + tmp4.size());
	tmp.insert(tmp.end(), tmp1.begin(), tmp1.end());
	tmp.insert(tmp.end(), tmp2.begin(), tmp2.end());
	tmp.insert(tmp.end(), tmp3.begin(), tmp3.end());
	tmp.insert(tmp.end(), tmp4.begin(), tmp4.end());

	convHull->convexHullPoints = tmp;
}

void createConvexHull(convexHull* convHull) {

	int s_z = convHull->simplexSize;
	if (s_z == 2) {
		// line
		create_ConvexLine(convHull);
	}
	else if (s_z == 3) {
		//triangle
		create_ConvexTriangle(convHull);
	}
	else if (s_z == 4) {
		//tetrahedron
		create_ConvexTetrahedron(convHull);
	}
}

// pt = minimumNorm(ConvexHull(Q U {V})) = argmin{<a,a>|for a in ConvexHull}
// iterates on all lines contained in the convex hull to find from all minimum points
// the global point.
void GJK_MinimumNormConvexHull(convexHull* convHull, std::vector<GLfloat>& p, int& l) {

	std::vector<std::vector<std::vector<GLfloat>>> conv = convHull->convexHullPoints;
	std::vector<GLfloat> tmp = conv[0][0];
	for (int i = 0; i < (int)conv.size(); i++) {
		std::vector<GLfloat> lp;
		GJK_MinimumNormLine(conv[i], lp);
		if (scalar_mult(lp, lp) < scalar_mult(tmp, tmp)) {
			tmp = lp;
			l = i;
		}
	}
	p = tmp;
}

void GJK_add_vec_toSimplex(convexHull* convHull, std::vector<GLfloat>& v) {
	std::vector<std::vector<GLfloat>> tmp = convHull->simplex;
	tmp.push_back(v);
	convHull->simplex = tmp;
	convHull->simplexSize = (int)tmp.size();
	assert(convHull->simplexSize <= 4);
	createConvexHull(convHull);
}

bool cmpr_vecs(std::vector<GLfloat>& v1, std::vector<GLfloat>& v2) {
	if ((v1[0] == v2[0]) && (v1[1] == v2[1]) && (v1[2] == v2[2])) {
		return true;
	}
	return false;
}

bool check_0_simplex(convexHull* convHull,
	std::vector<std::vector<std::vector<GLfloat>>>& conv,
	std::vector<GLfloat>& p,
	int l) {

	int size = (int)conv[l].size();
	int simplex_size = convHull->simplexSize;

	bool r = true;

	std::vector<std::vector<GLfloat>> osimplex;
	if (l > 0 && l < (int)conv.size() - 1) {
		if ((cmpr_vecs(p, conv[l][size - 1]) && cmpr_vecs(p, conv[l + 1][0])) ||
			(cmpr_vecs(p, conv[l][0]) && cmpr_vecs(p, conv[l - 1][size - 1]))) {
			osimplex.push_back(p);
			convHull->simplex = osimplex;
			convHull->simplexSize = 1;
		}
	}
	else if (l == 0) {
		if (cmpr_vecs(p, conv[l][0]) || cmpr_vecs(p, conv[l][size - 1])) {
			osimplex.push_back(p);
			convHull->simplex = osimplex;
			convHull->simplexSize = 1;
		}
	}
	else if (l == (int)conv.size() - 1) {
		if (simplex_size == 2) {
			osimplex.push_back(p);
			convHull->simplex = osimplex;
			convHull->simplexSize = 1;
		}
	}
	else {
		r = false;
	}
	return r;

}

bool check_1_simplex(convexHull* convHull,
	std::vector<std::vector<std::vector<GLfloat>>>& conv,
	std::vector<GLfloat>& p,
	int l) {

	int sz = convHull->simplexSize;
	int size = (int)conv[l].size();
	std::vector<std::vector<GLfloat>> oneSimplex;
	bool r = false;

	if (sz == 2) { r = true; }
	else if (sz == 3) {
		if (l < size - 1) { r = true; }
	}
	else if (sz == 4) {
		if ((l + 1) % 4 != 0) { r = true; }
	}
	else { r = false; }

	if (r) {
		oneSimplex.push_back(conv[l][0]);
		oneSimplex.push_back(conv[l][size - 1]);
		convHull->simplex = oneSimplex;
		convHull->simplexSize = 2;
	}
	return r;
}

bool check_2_simplex(convexHull* convHull,
	std::vector<std::vector<std::vector<GLfloat>>>& conv,
	std::vector<GLfloat>& p,
	int l) {

	int sz = convHull->simplexSize;
	int size = (int)conv[l].size();
	std::vector<std::vector<GLfloat>> twoSimplex;
	bool r = false;

	if (sz < 3) {
		r = false;
	}
	else if (sz == 3) {
		if (l == size - 1) { r = true; }
	}
	else if (sz == 4) {
		if ((l + 1) % 4 == 0) { r = true; }
	}
	else { r = false; }

	if (r) {
		twoSimplex.push_back(conv[l - 3][0]);
		twoSimplex.push_back(conv[l - 2][0]);
		twoSimplex.push_back(conv[l - 1][0]);
		convHull->simplex = twoSimplex;
		convHull->simplexSize = 3;
	}
	return r;
}

// Q <- Q' C Q U {v} && P in Q' && |Q'| = min{|W| | W C Q U {v}}
bool GJK_add_vec_optimizeSimplex(convexHull* convHull, std::vector<GLfloat>& v) {

	// for finding the closest point to origin
	int area;
	std::vector<GLfloat> p;
	std::vector<std::vector<std::vector<GLfloat>>> conv = convHull->convexHullPoints;

	GJK_add_vec_toSimplex(convHull, v);
	GJK_MinimumNormConvexHull(convHull, p, area);

	convHull->minNormPoint = p;
	// a case that p is in the vertices of the convex Hull - on the CSO
	// this is the 0-simplex case because the simplest convex subset of the
	// simplex that containes p is p itself.
	if (check_0_simplex(convHull, conv, p, area)) {
		return true;
	}

	// this is the 1-simplex case - when the minimum convex subset of Q that
	// containes p is a line between two points on the CSO
	else if (check_1_simplex(convHull, conv, p, area)) {
		return true;
	}

	// this is the 2-simplex case - when the minimum convex subset of Q that
	// contains p is a triangle - happens when p is inside the triangle
	// happens probably more with 3-simplexes that can be reduced to 2-simplex
	else if (check_2_simplex(convHull, conv, p, area)) {
		return true;
	}
	return false;
}

void GJK_main(std::vector<std::vector<GLfloat>>& CSO, convexHull* convHull, std::vector<GLfloat>& cp) {

	std::vector<GLfloat> p = CSO[0];

	std::vector<std::vector<GLfloat>> tmp_simplex;
	tmp_simplex.push_back(p);
	convHull->simplex = tmp_simplex;

	std::vector<GLfloat> v;
	GJK_shortest_dst_pt(CSO, p, v);

	while (scalar_mult(p, p) - scalar_mult(p, v) > GJK_EPSILON * GJK_EPSILON) {
		bool opt = GJK_add_vec_optimizeSimplex(convHull, v);
		if(!opt){ throw std::runtime_error(" something went wrong..."); }
		p = convHull->minNormPoint;
		GJK_shortest_dst_pt(CSO, p, v);
	}
	cp = p;
}

void GJK(Rigid_body* rigid1, Rigid_body* rigid2, convexHull* convHull, std::vector<GLfloat>& cp) {

	std::vector<std::vector<GLfloat>> CSO;
	GJK_minkowski_diff(rigid1, rigid2, CSO);
	GJK_main(CSO, convHull, cp);
}

bool recoverPoinTofCollision(Rigid_body* rigid1, Rigid_body* rigid2, std::vector<GLfloat>& p , std::vector<GLfloat>& res) {

	std::vector<std::vector<GLfloat>> mesh1 = rigid1->hitBoxPos;
	std::vector<std::vector<GLfloat>> mesh2 = rigid2->hitBoxPos;
	for (int i1 = 0; i1 < (int)mesh1.size(); i1++) {
		for (int i2 = 0; i2 < (int)mesh2.size(); i2++) {
			std::vector<GLfloat> p1 = mesh1[i1];
			std::vector<GLfloat> p2 = mesh1[i2];
			subtr_3Dvectors(p1, p2);
			if (cmpr_vecs(p1, p)) {
				add_toCont3Dvecs(mesh1[i1], mesh2[i2], res);
				return true;
			}
		}
	}
	return false;
}

bool detectCollisions(int hit_box1,
	int hit_box2,
	std::vector<std::vector<GLfloat>>& hitMesh1,
	std::vector<std::vector<GLfloat>>& hitMesh2) {

	col_detect col_obj;
	bool r = false;

	if (hit_box1 == BOUNDING_BOX && hit_box2 == BOUNDING_BOX) {
		if (col_obj.detect_BOX_BOX(hitMesh1, hitMesh2)) { r = true; }
	}
	else if ((hit_box1 == BOUNDING_BOX && hit_box2 == BOUNDING_SPHERE) ||
		(hit_box2 == BOUNDING_BOX && hit_box1 == BOUNDING_SPHERE)) {
		if (col_obj.detect_BOX_SPHERE(hitMesh1, hitMesh2)) { r = true; }
	}
	else if (hit_box1 == BOUNDING_SPHERE && hit_box2 == BOUNDING_SPHERE) {
		if (col_obj.detect_SPHERE_SPHERE(hitMesh1, hitMesh2)) { r = true; }
	}
	else if ((hit_box1 == BOUNDING_CONVEX && hit_box2 == BOUNDING_CONVEX) ||
		(hit_box1 == BOUNDING_CONVEX && hit_box2 == BOUNDING_BOX) ||
		(hit_box2 == BOUNDING_CONVEX && hit_box1 == BOUNDING_BOX) ||
		(hit_box1 == BOUNDING_CONVEX && hit_box2 == BOUNDING_SPHERE) ||
		(hit_box2 == BOUNDING_CONVEX && hit_box1 == BOUNDING_SPHERE) ||
		(hit_box1 == SUB_MESH && hit_box2 == SUB_MESH) ||
		(hit_box1 == SUB_MESH && hit_box2 == BOUNDING_CONVEX) ||
		(hit_box2 == SUB_MESH && hit_box1 == BOUNDING_CONVEX)) {
		if (col_obj.detect_CONVEX_CONVEX_or_SPHERE(hitMesh1, hitMesh2)) { r = true; }
	}
	else if ((hit_box1 == SUB_MESH && hit_box2 == BOUNDING_BOX) ||
		(hit_box2 == SUB_MESH && hit_box1 == BOUNDING_BOX) ||
		(hit_box1 == SUB_MESH && hit_box2 == BOUNDING_SPHERE) ||
		(hit_box2 == SUB_MESH && hit_box1 == BOUNDING_SPHERE)) {
		if (col_obj.detect_CONVEX_CONVEX_or_SPHERE(hitMesh1, hitMesh2)) { r = true; }
	}
	return r;
}

void detectAndFindCollisionPoint(Rigid_body* rigid1, Rigid_body* rigid2, convexHull* convHull, bool detected) {

	int hit_box1 = rigid1->hitBoxType;
	int hit_box2 = rigid2->hitBoxType;

	std::vector<std::vector<GLfloat>> hitMesh1 = rigid1->hitBoxPos;
	std::vector<std::vector<GLfloat>> hitMesh2 = rigid2->hitBoxPos;
	
	if (detected) {
		rigid1->collided = true;
		rigid2->collided = true;
		std::vector<GLfloat> CSO_pt;
		std::vector<GLfloat> col_pt;
		GJK(rigid1, rigid2, convHull, CSO_pt);
		recoverPoinTofCollision(rigid1, rigid2, CSO_pt, col_pt);
		rigid1->collisionPosition = col_pt;
		rigid2->collisionPosition = col_pt;
	}
}

void createPolygonNormal(Rigid_body* rigid, std::vector<GLfloat>& init_pt, std::vector<GLfloat>& n, int& c) {

	std::vector<std::vector<GLfloat>> body = rigid->bodyPos;
	std::vector<int> polpos = rigid->body_polygon_size;

	// project collision point onto the rigid body
	int minpoly = 0;
	int closest = 0;
	find_init_pt_onBody(rigid, init_pt, minpoly, closest);

	// find its polygon
	int i = 0;
	int j = 0;
	while (i < minpoly) {
		j += polpos[i];
		i++;
	}

	// create the normal
	std::vector<GLfloat> i_vec = body[j];
	std::vector<GLfloat> e_vec = body[j + polpos[minpoly] - 1];
	std::vector<GLfloat> mid_vec = body[closest];

	std::vector<GLfloat> p_v1 = e_vec;
	std::vector<GLfloat> p_v2 = mid_vec;
	subtr_3Dvectors(p_v1, i_vec);
	subtr_3Dvectors(p_v2, i_vec);

	std::vector<GLfloat> normal;
	cross_product(p_v1, p_v2, normal);
	normalize_vec(normal);
	n = normal;
	c = closest;
}

// based on a loose approximation from the book "contact mechanics" by K L thornton (1987)
float approx_E(Rigid_body* rigid1, Rigid_body* rigid2) {

	float e;
	float v1_sq = scalar_mult(rigid1->linearVelocity, rigid1->linearVelocity);
	float v2_sq = scalar_mult(rigid2->linearVelocity, rigid2->linearVelocity);
	float maxV_sq = IMPACT_CRITICAL_VELOCITY * IMPACT_CRITICAL_VELOCITY;
	if (rigid1->linearVelocity > rigid2->linearVelocity) {
		if (v1_sq < maxV_sq) {
			e = 1;
		}
		else {
			e = pow((long double)v1_sq, -1 / 8);
		}
	}
	else {
		if (v2_sq < maxV_sq) {
			e = 1;
		}
		else {
			e = pow((long double)v2_sq, -1 / 8);
		}
	}
	return e;
}

// the output is the force vector, but clculating it directly is difficult, so
// the vector calculated is the impulse infact, and not the actual force.
// in moment of collision, because the time window between 
// the initial state of the system to final state is very small, 
// the force can be approximated with the impulse, as shown:
/*
for clearence : (a=) <=> approximately equall, FF' = F
J = integral[F]_{ti->te}, te = ti + dt , dt -> 0.
=> J (a=) FF(te)-FF(ti) = FF(ti + dt) - FF(ti) = [(FF(ti + dt) - FF(ti))/dt]dt = f(ti)dt
																			 dt->0
fro riemann sum of the inegral we can see the same:
J (a=) sum_{k = 0 to n} f(t_bk)dt_k , t_bk in [t_k, t_k+1] = f(te)dt/ f(ti)dt

the impulse itself is the difference between final and initial momentum,
and for short events such as collisions, if we look at the equations of motion:

v(te) = v(ti) + a(te - ti) = v(ti) + F(te - ti)/m = F(te)dt/m = J/m => v(te) = v(ti) + J/m.

so, from the impulse we can derive the force:
F = J/dt = J/(collision_time).

for calculating J, the formula is:
e = (v_bf - v_af)/(v_bi - v_ai)
J = -(1+e)*{[(v_ai - v_bi)*n + (ra x n)*wa - (rb x n)*wb]/[1/ma + 1/mb + (ra x n)*(Ia^-1(ra x n)) + (rb x n)*(Ib^-1(rb x n))]}
*/
std::vector<GLfloat> calc_collision_force(Rigid_body* rigid1, Rigid_body* rigid2) {

	// body 1
	int init_ind1;
	std::vector<std::vector<GLfloat>> body1 = rigid1->bodyPos;
	std::vector<GLfloat> col1 = rigid1->collisionPosition;
	std::vector<GLfloat> n1;
	createPolygonNormal(rigid1, col1, n1, init_ind1);

	std::vector<GLfloat> r1 = body1[init_ind1];
	std::vector<GLfloat> c_m1 = rigid1->centerOfMass;
	subtr_3Dvectors(r1, c_m1);

	GLfloat m1 = rigid1->mass;

	std::vector<GLfloat> v1 = rigid1->linearVelocity;
	std::vector<GLfloat> w1 = rigid1->angularVelocity;
	std::vector<std::vector<GLfloat>> I1 = rigid1->inertiaTensor;


	// body 2
	int init_ind2;
	std::vector<std::vector<GLfloat>> body2 = rigid2->bodyPos;
	std::vector<GLfloat> col2 = rigid2->collisionPosition;
	std::vector<GLfloat> n2;
	createPolygonNormal(rigid2, col2, n2, init_ind2);

	std::vector<GLfloat> r2 = body2[init_ind2];
	std::vector<GLfloat> c_m2 = rigid2->centerOfMass;
	subtr_3Dvectors(r2, c_m2);

	GLfloat m2 = rigid2->mass;

	std::vector<GLfloat> v2 = rigid2->linearVelocity;
	std::vector<GLfloat> w2 = rigid2->angularVelocity;
	std::vector<std::vector<GLfloat>> I2 = rigid2->inertiaTensor;


	// global 
	float e = approx_E(rigid1, rigid2);
	float Jtop, Jbottom, J_factor;
	
	// calculation - J coeff from v1 - the same force works on both bodies but in opposite directions - 
	// 3rd law of newton.
	std::vector<GLfloat> diff = v1;
	std::vector<GLfloat> c1;
	std::vector<GLfloat> c2;
	std::vector<std::vector<GLfloat>> invI1;
	std::vector<std::vector<GLfloat>> invI2;
	subtr_3Dvectors(diff, v2);
	cross_product(r1, n1, c1);
	cross_product(r2, n1, c2);

	calc_3D_inverse_mat(I1, invI1);
	calc_3D_inverse_mat(I2, invI2);

	GLfloat* a1 = mult_3D_mat_vec(invI1, c1);
	GLfloat* a2 = mult_3D_mat_vec(invI2, c2);

	std::vector<GLfloat> ac1(a1, a1 + 3);
	std::vector<GLfloat> ac2(a2, a2 + 3);

	Jtop = scalar_mult(diff, n1) + scalar_mult(c1, w1) + scalar_mult(c2, w2);
	Jbottom = 1 / m1 + 1 / m2 + scalar_mult(c1, ac1) + scalar_mult(c2, ac2);
	J_factor = -(1 + e) * (Jtop / Jbottom);

	std::vector<GLfloat> J1 = n1;
	scale_3Dvectors(J1, J_factor);
	// F = J/dt = J/(te - ti) - for the moment of collision - it is the force.
	return J1;
}

// in order to calculate the force on the second body - use the 3rd law of newton
std::vector<GLfloat> newton_3rd_law(std::vector<GLfloat>& F1) {
	std::vector<GLfloat> F2 = F1;
	scale_3Dvectors(F2, -1.0);
	return F2;
}

void gravityElem(Rigid_body* rigid, GLfloat m_elem, std::vector<GLfloat>& G_elem) {
	std::vector<GLfloat> centerMass = rigid->centerOfMass;
	std::vector<GLfloat> f = { 0, 0, -1 };
	scale_3Dvectors(f, m_elem * G);
	G_elem = f;
}

// if G is uniform and constant - the gravity center is the center of mass
void setGravity(Rigid_body* rigid, std::vector<std::vector<GLfloat>>& G_distrib) {
	std::vector<GLfloat> m_d = rigid->massDistribution;
	std::vector<std::vector<GLfloat>> G_d;
	int i = 0;
	while (i < (int)m_d.size()) {
		std::vector<GLfloat> G_elem;
		gravityElem(rigid, m_d[i], G_elem);
		G_d.push_back(G_elem);
	}
	G_distrib = G_d;
}

// apply the gravity into the force distribution of the body
void distributeGravity(Rigid_body* rigid) {

	std::vector<std::vector<GLfloat>> G_distrib;
	setGravity(rigid, G_distrib);

	std::vector<std::vector<GLfloat>> r;
	std::vector<std::vector<GLfloat>> t;
	radial_tangent_decomposition(rigid, G_distrib, r, t);
	rigid->Force_distribContainer.push_back(G_distrib);
	rigid->Force_distrib_radial = r;

	average_radial_force(rigid, r);
	rigid->Force_distrib_radial = r;
	rigid->Force_distrib_tangent = t;
}

void apply_gravity(Rigid_body* rigid, float time_stamp) {

	if (rigid->gravityApplied) { distributeGravity(rigid); }
	update_position(rigid, time_stamp);
}

void remove_gravity(Rigid_body* rigid) {
	std::vector<std::vector<std::vector<GLfloat>>> fd = rigid->Force_distribContainer;
	std::vector<std::vector<GLfloat>> f_df = fd[(int)fd.size() - 1];
	std::vector<std::vector<GLfloat>> r = rigid->Force_distrib_radial;
	std::vector<std::vector<GLfloat>> t = rigid->Force_distrib_tangent;
	std::vector<std::vector<GLfloat>> rad_g;
	std::vector<std::vector<GLfloat>> tang_g;
	radial_tangent_decomposition(rigid, f_df, rad_g, tang_g);

	// find radial average gravity
	std::vector<GLfloat> avg_r = { 0, 0, 0 };
	std::vector<std::vector<GLfloat>>::iterator v = rad_g.begin();
	while (v != rad_g.end()) {
		add_3Dvectors(avg_r, *v);
		v++;
	}
	GLfloat scale = (GLfloat)(1 / ((int)rad_g.size()));
	scale_3Dvectors(avg_r, scale);

	// removing from radial distribution
	int i1 = 0;
	while (i1 < (int)r.size()) {
		subtr_3Dvectors(r[i1], avg_r);
	}

	// removing from the tangent force distribution
	int i2 = 0;
	while (i2 < (int)t.size()) {
		subtr_3Dvectors(t[i2], tang_g[i2]);
	}

	rigid->Force_distrib_radial = r;
	rigid->Force_distrib_tangent = t;

	std::vector<std::vector<std::vector<GLfloat>>> f = rigid->Force_distribContainer;
	f.erase(f.end());
	rigid->Force_distribContainer = f;

}

// for one constant force working over time
void apply_continous_gravity(Rigid_body* rigid, float time_steps, float dt) {

	int i = 0;
	while (i < time_steps) {
		apply_gravity(rigid, dt);
		remove_gravity(rigid);
		i++;
	}
}

/*----------------------full time pipeline - putting all the peices together through time---------------------*/
// the pipeline:
/*
initial state <- 0,0,0...
while time is running:
	if bodies are added:
		initial state <- bodies
	end
	body1,2,...,n <- gravity
	if n > 1 and collided(bodym , bodyk) m, k in {1,2,...,n}:
		bodym <- force, boyk <- -force
	end
	if added force:
		calculate position change
		update position on screen
	end
end
*/

// for loading the body - first time
void loadBody(Rigid_body* rigid, std::vector<std::vector<std::vector<GLfloat>>>& body, int meshType) {

	std::vector<std::vector<GLfloat>> flatten_b;
	std::vector<std::vector<GLfloat>> bounds;
	
	translate_vertices_LEGACY_GL(body, flatten_b);
	fitMesh(flatten_b, bounds, meshType);

	rigid->bodyPos = flatten_b;
	rigid->hitBoxPos = bounds;
	rigid->hitBoxType = meshType;
}

void initiate_physics(Rigid_body* rigid,
	std::vector<std::vector<std::vector<GLfloat>>>& body,
	std::vector<int> polygons,
	bool modifyMassDistribution,
	std::vector<GLfloat> mass_distribution,
	GLfloat M,
	int meshType) {

	loadBody(rigid, body, meshType);
	rigid->body_polygon_size = polygons;
	rigid->collisionPosition = { 0, 0, 0 };
	default_RigidMass_distribution(rigid);
	if (modifyMassDistribution) { modify_RigidMass_distribution(rigid, mass_distribution); }
	rigid->mass = M;
	calculate_center_mass(rigid);
	rigid->gravityApplied = true;
	rigid->isFullyElasticAndRigid = true;
	rigid->collision_allowed = true;
	rigid->collided = false;
	init_force(rigid);
	init_torque(rigid);
	rigid->linearVelocity = { 0, 0, 0 };
	rigid->linearAcceleration = { 0, 0, 0 };
	rigid->linearVelocityElements = { {} };
	rigid->angularVelocity = { 0, 0, 0 };
	rigid->angularAcceleration = { 0, 0, 0 };
	init_VelocityElems(rigid);
	calc_Inertia_tensor(rigid);
}

// default force - will be added custom force making
void addForce(Rigid_body* rigid) {

	std::vector<std::vector<GLfloat>> body = rigid->bodyPos;
	std::vector<GLfloat> init = body[0];
	std::vector<GLfloat> force = { 1,1,1 };

	apply_continous_force(rigid, init, force, 1, DT);
}

// general structure , what is written NOW is for a test
void singleRigidBodyPhysics(Rigid_body* currBody,
	std::vector<Rigid_body>& bodyList,
	bool applyLinearForce) {

	// apply gravity
	if(currBody->gravityApplied){ apply_continous_gravity(currBody, 1, DT); }
	else { update_position(currBody, 1); }

	// checking for random force that applies
	if (applyLinearForce) { addForce(currBody); }
	
	// checking for collision - bodyList is without the current body
	std::vector<Rigid_body>::iterator b = bodyList.begin();
	int hCtype = currBody->hitBoxType;
	std::vector<std::vector<GLfloat>> hCmesh = currBody->hitBoxPos;
	while (b != bodyList.end()) {
		Rigid_body external_body = *b;
		convexHull convHull;
		std::vector<std::vector<GLfloat>> hmesh = external_body.hitBoxPos;
		int htype = external_body.hitBoxType;
		if (detectCollisions(hCtype, htype, hCmesh, hmesh)) {
			detectAndFindCollisionPoint(currBody, &external_body, &convHull, true);
			std::vector<GLfloat> f1 = calc_collision_force(currBody, &external_body);
			std::vector<GLfloat> init1 = currBody->collisionPosition;
			apply_continous_force(currBody, init1, f1, 1, DT);
		}
	}
}

// for the graphics rendering
void draw_multipleFlatRigidBodies_LEGACY_GL(std::vector<Rigid_body>& bodies, GLenum render_type) {

	for (auto obj = bodies.begin(); obj != bodies.end(); ++obj) {
		std::vector<std::vector<GLfloat>> v = obj->bodyPos;
		draw_flat_obj_LEGACY_GL(v, render_type);
	}
}

void generate_icosahedron_rigidBody_array(std::vector<Rigid_body>& bodyList) {

	// for icosahedrons
	std::vector<int> p_z;
	for (int j = 0; j < 20; j++) { p_z.push_back(3); }
	std::vector<GLfloat> massD;
	for (int k = 0; k < 12; k++) { massD.push_back(1 / 12); }

	std::vector<Rigid_body> bodyL;
	int i = 0;
	while (i < 5) {
		// calculate mesh
		std::vector<std::vector<GLfloat>> mesh;
		std::vector<std::vector<std::vector<GLfloat>>> meshObj;
		int rand_x = generate_random_number_INT(0, 10);
		int rand_y = generate_random_number_INT(0, 10);
		int rand_z = generate_random_number_INT(0, 10);
		GLfloat center[3] = { rand_x, rand_y, rand_z };
		generate_tinyCONVEX_mesh(mesh, center, 1);
		meshObj.push_back(mesh);
		

		Rigid_body body;
		initiate_physics(&body, meshObj, p_z, false, massD, 50, BOUNDING_BOX);
		bodyL.push_back(body);
	}
	bodyList = bodyL;
}

// for a demo
void systemPhysicsLoop(int val) {
	// display bodies & scene - from model_draw. TODO in *MODEL_DRAW* - to create a function that draws multiple bodies
	//loop over all to check for collisions and update physics

	bool gravityApplied = true;
	std::vector<bool> applyLinearForce = { false, false, true, true, false };
	std::vector<Rigid_body> bodyList;
	generate_icosahedron_rigidBody_array(bodyList);

	draw_multipleFlatRigidBodies_LEGACY_GL(bodyList, GL_TRIANGLES);

	int i = 0;
	for (auto b = bodyList.begin(); b != bodyList.end(); ++b) {
		Rigid_body body = *b;
		singleRigidBodyPhysics(&body, bodyList, applyLinearForce[i]);
		i++; b++;
	}

	glutTimerFunc(1, systemPhysicsLoop, 0);
}