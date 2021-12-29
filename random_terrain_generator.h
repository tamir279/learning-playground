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
#include <random>


// generate

typedef struct vertex {
	GLfloat x = 0;
	GLfloat y = 0;
	GLfloat z = 0;
}vector3D;

typedef struct cont {
	std::vector<vector3D> grid;
}vector3DContainer;

typedef struct Container {
	vector3DContainer vertexGrid;
	vector3DContainer normalGrid;
}renderDataContainer;

renderDataContainer generateFlatSurface(GLfloat min_x, GLfloat min_y, GLfloat max_x, GLfloat max_y, GLfloat z_level);
vector3D noise(GLfloat amplitude);
renderDataContainer add_noiseToGrid(renderDataContainer grid, GLfloat amplitude);
// draw

#endif