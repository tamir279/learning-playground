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
#include "random_terrain_generator.h"

#define DX 1E-02F;
#define DY 1E-02F;

// generate noise induced surface from flat surface
renderDataContainer generateFlatSurface(GLfloat min_x, GLfloat min_y, GLfloat max_x, GLfloat max_y, GLfloat z_level) {
	
	vector3DContainer flatMesh;
	vector3DContainer normals;

	int i = 0, j = 0;
	GLfloat x_pt = min_x;
	GLfloat y_pt = min_y;
	vector3D point;
	vector3D normal;

	while (x_pt < max_x) {
		y_pt = min_y;
		while (y_pt < max_y) {

			point.x = x_pt;
			point.y = y_pt;
			point.z = z_level;

			normal.x = 0.0;
			normal.y = 0.0;
			normal.z = 1.0;

			flatMesh.grid.push_back(point);
			normals.grid.push_back(normal);

			y_pt += DY;
		}
		x_pt += DX;
	}
	
	renderDataContainer flatSurface;
	flatSurface.vertexGrid = flatMesh;
	flatSurface.normalGrid = normals;

	return flatSurface;
}

// generate noise vector
vector3D noise(GLfloat amplitude) {

	double mean = 5.0;
	double s_d = 2.0;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(mean, s_d);

	vector3D noise;

	noise.x = (GLfloat)distribution(generator);
	noise.y = (GLfloat)distribution(generator);
	noise.z = (GLfloat)distribution(generator);

	return noise;
}


// TODO: how to change normals?
renderDataContainer add_noiseToGrid(renderDataContainer grid, GLfloat amplitude) {

	vector3DContainer flatVertexGrid = grid.vertexGrid;
	vector3DContainer flatNormals = grid.normalGrid;

	vector3DContainer noisyVertexGrid;

	GLfloat alpha1 = 2.4, alpha2 = 0.8;

	for (auto vector3 = flatVertexGrid.grid.begin(); vector3 != flatVertexGrid.grid.end(); ++vector3) {
		vector3D v = *vector3;
		// randomly sample noise vector 
 		vector3D n = noise(amplitude);
		vector3D linearCombination;

		// it can be better if the combination would be with more axis (y, z)
		linearCombination.x = alpha1 * n.x + alpha2 * v.x;
		linearCombination.y = alpha1 * n.y + alpha2 * v.y;
		linearCombination.z = alpha1 * n.z + alpha2 * v.z;

		noisyVertexGrid.grid.push_back(linearCombination);
	}
}