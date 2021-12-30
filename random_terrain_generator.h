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
}renderDataContainer;

renderDataContainer generateFlatSurface(GLfloat min_x, GLfloat min_y, GLfloat max_x, GLfloat max_y, GLfloat z_level);
vector3D noise(GLfloat amplitude);
float euclideanDistance(vector3D pt1, vector3D pt2);
vector3D cross_product(vector3D v1, vector3D v2);
vector3D normalize(vector3D v);
vector3DContainer getClosestPointInGrid(vector3DContainer vertexGrid, vector3D refPt);
vector3D calculateNormal(vector3DContainer vertexGrid, vector3D refPt);
vector3DContainer add_noiseToVertexGrid(renderDataContainer grid, GLfloat amplitude);
vector3DContainer calculate_noisyNormals(renderDataContainer grid);
renderDataContainer generateNoisySurface(GLfloat min_x, GLfloat min_y, GLfloat max_x, GLfloat max_y, GLfloat z_level, GLfloat amplitude);

// draw

void drawNoisySurface_LEGACY_GL(renderDataContainer surface);

#endif