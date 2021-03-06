#pragma once
#ifndef COL_DET_f_H
#define COL_DET_f_H

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

// creating collision box
void translate_vertices_LEGACY_GL(std::vector<std::vector<std::vector<GLfloat>>>& vertices_per_object, std::vector<std::vector<GLfloat>>& vertex_vec);
void sortByIndex(GLfloat vertex_data[], GLushort index[], int ISIZE, std::vector<std::vector<GLfloat>>& vertex_vec);
void generate_tinyBOX_mesh(std::vector<std::vector<GLfloat>>& vertex_vec, GLfloat center[], GLfloat epsilon);
void generate_tinyCONVEX_mesh(std::vector<std::vector<GLfloat>>& mesh, GLfloat center[], GLfloat epsilon);
float fast_sqrt(float num);
void normalize(float v[3]);
void PolygonSubdivision(float* v1, float* v2, float* v3, long depth, std::vector<std::vector<GLfloat>>& mesh);
void generate_tinySPHERE_mesh(std::vector<std::vector<GLfloat>>& mesh, GLfloat center[3], GLfloat epsilon, int depth);
auto generate_random_number_INT(int min, int max);
void generate_SUBSAMPLE_mesh(std::vector<std::vector<GLfloat>>& mesh, std::vector<std::vector<GLfloat>>& orig_mesh, int sample_mode);
void init_3Dvec(std::vector<GLfloat>& v);
void add_3Dvectors(std::vector<GLfloat>& v_r, std::vector<GLfloat>& v_a);
void scale_3Dvectors(std::vector<GLfloat>& v, GLfloat scale);
void add_3Dvec_to_mesh(std::vector<std::vector<GLfloat>>& mesh, std::vector<GLfloat>& v);
void scale_3Dmesh(std::vector<std::vector<GLfloat>>& mesh, GLfloat scale);
GLfloat* geometrig_center(std::vector<std::vector<GLfloat>>& model_mesh);
bool detect_boundries(std::vector<std::vector<GLfloat>>& model_mesh, std::vector<std::vector<GLfloat>>& bounding_mesh, GLfloat center[]);
void inflate_mesh(std::vector<std::vector<GLfloat>>& model_mesh, std::vector<std::vector<GLfloat>>& bounding_mesh, GLfloat center[], GLfloat init_scale, GLfloat step);
void fitMesh(std::vector<std::vector<GLfloat>>& model_mesh, std::vector<std::vector<GLfloat>>& bounding_mesh, int mesh_type);
void fitMesh_from_orig_Model_LEGACY_GL(std::vector<std::vector<std::vector<GLfloat>>>& vertices, std::vector<std::vector<GLfloat>>& bounding_mesh, int mesh_type);

// detecting collisions
void find_max_min_Coords_vals(std::vector<std::vector<GLfloat>>& box, GLfloat& x, GLfloat& y, GLfloat& z, GLfloat& mx, GLfloat& my, GLfloat& mz);
bool detect_collision_BOX_vs_BOX(std::vector<std::vector<GLfloat>>& box1, std::vector<std::vector<GLfloat>>& box2);
float radius_squared(std::vector<GLfloat>& point, GLfloat center[]);
float approx_radius(std::vector<GLfloat>& point, GLfloat center[]);
float approx_distance(GLfloat point1[], GLfloat point2[]);
bool detect_collision_SPHERE_vs_SPHERE(std::vector<std::vector<GLfloat>>& sphere1, std::vector<std::vector<GLfloat>>& sphere2, GLfloat center1[], GLfloat center2[]);
bool detect_collision_BOX_vs_SPHERE(std::vector<std::vector<GLfloat>>& box, std::vector<std::vector<GLfloat>>& sphere, GLfloat center[]);
void cross_product(std::vector<GLfloat>& v1, std::vector<GLfloat>& v2, std::vector<GLfloat>& res);
void normalize_vec(std::vector<GLfloat>& v);
void f_normal_per_polygon(std::vector<std::vector<GLfloat>>& polygon, std::vector<GLfloat>& normal);
void f_normals(std::vector<std::vector<GLfloat>>& mesh, std::vector<std::vector<GLfloat>>& f_normals, int pol_type);
GLfloat scalar_mult(std::vector<GLfloat>& v1, std::vector<GLfloat>& v2);
void project_3D_along_axis(std::vector<std::vector<GLfloat>>& obj, std::vector<GLfloat>& axis, std::vector<std::vector<GLfloat>>& proj);
float dist_sq_vec(std::vector<GLfloat>& p1, std::vector<GLfloat>& p2);
void find_Min_Max_line(std::vector<std::vector<GLfloat>>& line, float& min, float& max);
bool check_overlap_along_axis(std::vector<std::vector<GLfloat>>& obj1, std::vector<std::vector<GLfloat>>& obj2, std::vector<GLfloat>& axis);
bool detect_collision_CONVEX_vs_CONVEX_or_SPHERE(std::vector<std::vector<GLfloat>>& conv1, std::vector<std::vector<GLfloat>>& conv2);
class col_detect;

// simple physics -- for a given time t_0
typedef struct Rigid_body;
void default_RigidMass_distribution(Rigid_body* rigidBody);
void modify_RigidMass_distribution(Rigid_body* rigidBody, std::vector<GLfloat>& distrib);
void calculate_center_mass(Rigid_body* rigidBody);
void calc_mass_elem_vec(std::vector<GLfloat>& mass_distrib, GLfloat mass, std::vector<GLfloat>& mass_elems);
GLfloat calc_I_by_coords(GLfloat mass_elem, std::vector<GLfloat>& v, GLint c[3]);
GLfloat calc_I_elem(Rigid_body* rigidBody, GLint coords[3]);
void calc_Inertia_tensor(Rigid_body* rigidBody);
GLfloat* mult_3D_mat_vec(std::vector<std::vector<GLfloat>>& mat, std::vector<GLfloat>& vec);
bool calc_3D_inverse_mat(std::vector<std::vector<GLfloat>>& A, std::vector<std::vector<GLfloat>>& I_A);
void calc_AngularMomentum_from_InertiaTensor(Rigid_body* rigidBody);
void calc_AngularVelocity_from_InertiaTensor(Rigid_body* rigidBody);
void init_force(Rigid_body* rigidBody);
void find_poly_interval(std::vector<int> polygon, int poly, int& min_ind, int& max_ind);
void distrib_force_to_polygon(std::vector<GLfloat>& force, Rigid_body* rigidBody, int min_poly, int closest_point, std::vector<std::vector<GLfloat>>& force_distrib);
void distrib_force_to_mass_elems(Rigid_body* rigidBody, std::vector<GLfloat>& init_force_pt, std::vector<GLfloat>& force, std::vector<std::vector<GLfloat>>& force_distrib);
GLfloat* create_CenterMass_axis(Rigid_body* rigidBody, std::vector<GLfloat>& init_force_pt, std::vector<GLfloat>& force);
GLfloat* create_tangent_axis(GLfloat* cm_axis, std::vector<GLfloat>& force);
void create_force_axis(Rigid_body* rigidBody, std::vector<GLfloat>& init_force_pt, std::vector<GLfloat>& force, std::vector<GLfloat>& center_mass_axis, std::vector<GLfloat>& tangent_axis);
void radial_tangent_decomposition(Rigid_body* rigidBody, std::vector<GLfloat>& init_force_pt, std::vector<GLfloat>& force, std::vector<std::vector<GLfloat>>& force_distrib);
void average_radial_force(Rigid_body* rigidBody);
void apply_force(Rigid_body* rigidBody, std::vector<GLfloat>& init_force_pt, std::vector<GLfloat>& force);
void newton_linear_second_law(Rigid_body* rigid, float time_stamp);
void calc_linear_velocity_elems(Rigid_body* rigid, float time_stamp);
void init_torque(Rigid_body* rigid);
void calc_torque(Rigid_body* rigid);
void calc_angularAcceleration(Rigid_body* rigid);
void calc_angularVelocity_from_acc(Rigid_body* rigid, float time_stamp);
void calc_angular_velocity_elems(Rigid_body* rigid, float time_stamp);
void linear_position_update(Rigid_body* rigid, float time_stamp);
void rotate_vec_along_axis(std::vector<GLfloat>& v, GLfloat rot_angle, std::vector<GLfloat>& rot_axis, std::vector<GLfloat>& res);
float approx_dist_vec(std::vector<GLfloat>& v);
void angular_position_update(Rigid_body* rigid, float time_stamp);
void apply_force_update_position(Rigid_body* rigid, std::vector<GLfloat>& init_force_pt, std::vector<GLfloat>& force, float time_stamp);
void check_approx_coll(Rigid_body* rigid1, Rigid_body* rigid2, std::vector<GLfloat>& col_pt);
void approx_collision_point(Rigid_body* rigid1, Rigid_body* rigid2);
#endif