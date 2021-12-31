#pragma once
#ifndef MODEL_DRAWER_H
#define MODEL_DRAWER_H

#include "glew.h"
#include <windows.h>
#include <iostream>
#include <vector>
//#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
//#include "glext.h"
#include <GL/glu.h>
#include "GL/glut.h"
#include <stdlib.h>
#include <string>
#include "image_path_loader.h"
#include "obj_model_read_write.h"

// defines
/*
#pragma warning( push )
#pragma warning( disable : 6262)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma warning( pop )
*/

/* functions for textured .obj model drawing*/

void defineColorsAndMaterials_LEGACY_GL(unsigned int numObj, std::vector<std::vector<std::vector<GLfloat>>>& material_Matrix);
GLenum encode_GL_polygon_type(int polygonSize);
void draw_OBJ_Object_Polygons_LEGACY_GL(std::vector<std::vector<GLfloat>>& vertices, std::vector<std::vector<GLfloat>>& UVs, std::vector<std::vector<GLfloat>>& normals, std::vector<int>& polygon_size, bool isTextured, bool TEXTURED);
void loadTexMaps(std::vector<std::string>& texBuffer, std::vector<std::string>& textureMapVec, std::vector<int>& TexWidth, std::vector<int>& TexHeight, std::vector<int>& TexNrChannels);
void textureDefPerObject(std::vector<std::string>& textureMapVec, std::vector<std::string>& texBuffer, unsigned int numTex, GLuint textureID, int& TexWidth, int& TexHeight, int& TexNrChannels);
void init_model(const char* OBJpath, const char* MTLpath, std::vector<std::vector<std::vector<GLfloat>>>& materialModel, std::vector<std::string>& texMaps, std::vector<std::vector<std::vector<GLfloat>>>& vetrices, std::vector<std::vector<std::vector<GLfloat>>>& UVs, std::vector<std::vector<std::vector<GLfloat>>>& normals,std::vector<int>& polygon_size, bool TEXTURED, std::vector<bool>& isTextured, std::vector<std::string>& texBuffer, std::vector<int>& TexWidth, std::vector<int>& TexHeight, std::vector<int>& TexNrChannels);
void draw_textured_elements_LEGACY_GL(std::vector<std::vector<std::vector<GLfloat>>>& material_Matrix, std::vector<std::string>& texMaps, std::vector<std::string>& texBuffer, std::vector<int>& TexWidth, std::vector<int>& TexHeight, std::vector<int>& TexNrChannels, std::vector<std::vector<std::vector<GLfloat>>>& vetrices, std::vector<std::vector<std::vector<GLfloat>>>& UVs, std::vector<std::vector<std::vector<GLfloat>>>& normals, std::vector<int>& polygon_size, bool TEXTURED, std::vector<bool>& isTextured);
void init_scene_params_LEGACY_GL(GLint& fogMode, GLenum shade_mode);
void display_scene_light_LEGACY_GL(std::vector<GLfloat>& init_cam_pos, std::vector<GLfloat>& cam_trans, std::vector<GLfloat>& cam_rot, std::vector<GLfloat>& init_light_pos, std::vector<GLfloat>& light_trans, std::vector<GLfloat>& light_rot);
void draw_flat_obj_LEGACY_GL(std::vector<std::vector<GLfloat>>& indexed_vertices, GLenum render_type);
void draw_wireframe_LEGACY_GL(std::vector<std::vector<GLfloat>>& indexed_vertices);
void drawCheckerBoard_LEGACY_GL(int Cwidth, int Cdepth, GLfloat COLOR1[], GLfloat COLOR2[]);
void display_scene_obj(std::vector<GLfloat>& init_cam_pos);
#endif