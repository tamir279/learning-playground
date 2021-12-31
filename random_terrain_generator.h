#pragma once
#ifndef RANDOM_TERRAIN_H
#define RANDOM_TERRAIN_H

#include <windows.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <GL/gl.h>
#include <GL/glu.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <algorithm>
#include <random>


// generate

typedef struct vertex2D {
	GLfloat x = 0;
	GLfloat y = 0;
}vector2D;

typedef struct vertex {
	GLfloat x = 0;
	GLfloat y = 0;
	GLfloat z = 0;
}vector3D;

typedef struct cont2D {
	std::vector<vector2D> grid;
}vector2DContainer;

typedef struct cont {
	std::vector<vector3D> grid;
}vector3DContainer;

typedef struct Container {
	vector3DContainer vertexGrid;
	vector3DContainer normalGrid;
	vector3DContainer GridIndices;
}renderDataContainer;

vector3DContainer getGridIndices(int x_s, int y_s);
renderDataContainer generateFlatSurface(GLfloat min_x, GLfloat min_y, GLfloat max_x, GLfloat max_y, GLfloat z_level);
float euclideanDistance(vector3D pt1, vector3D pt2);
vector3D noise(double mean, double s_d);
vector3D cross_product(vector3D v1, vector3D v2);
vector3D normalize(vector3D v);
vector3DContainer getClosestPointInGrid(vector3DContainer vertexGrid, vector3D refPt);
vector3D calculateNormal(vector3DContainer vertexGrid, vector3D refPt);
vector3DContainer add_noiseToVertexGrid(renderDataContainer grid);
vector3DContainer calculate_noisyNormals(renderDataContainer grid);
renderDataContainer generateNoisySurface(GLfloat min_x, GLfloat min_y, GLfloat max_x, GLfloat max_y, GLfloat z_level);

// draw

void drawNoisySurface_LEGACY_GL(renderDataContainer surface, GLenum type);

#endif
