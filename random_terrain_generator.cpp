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
#include <algorithm>
#include "random_terrain_generator.h"
#include "rigid_body_physics.h"

#define DX 0.5F;
#define DY 0.5F;

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


float euclideanDistance(vector3D pt1, vector3D pt2) {

	float dist_sq = (pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y) + (pt1.z - pt2.z) * (pt1.z - pt2.z);
	return fast_sqrt(dist_sq);
}

vector3D cross_product(vector3D v1, vector3D v2) {
	GLfloat c_x = v1.y * v2.z - v1.z * v2.y;
	GLfloat c_y = v1.z * v2.x - v1.x * v2.z;
	GLfloat c_z = v1.x * v2.y - v1.y * v2.x;

	vector3D res;
	res.x = c_x;
	res.y = c_y;
	res.z = c_z;

	return res;
}

vector3D normalize(vector3D v) {
	vector3D zeroV;
	vector3D res;
	GLfloat norm = (GLfloat)euclideanDistance(v, zeroV);

	res.x = v.x / norm;
	res.y = v.y / norm;
	res.z = v.z / norm;

	return res;
}

vector3DContainer getClosestPointInGrid(vector3DContainer vertexGrid, vector3D refPt) {
	vector2DContainer distanceArr;
	int vertexInd = 0;
	// store indices and distances into a 2D vector container
	for (auto vertex = vertexGrid.grid.begin(); vertex != vertexGrid.grid.end(); ++vertex) {
		vector3D v = *vertex;
		float d = euclideanDistance(v, refPt);

		vector2D id;
		id.x = d;
		id.y = (GLfloat)vertexInd;
		distanceArr.grid.push_back(id); vertexInd++;
	}
	std::sort(distanceArr.grid.begin(), distanceArr.grid.end(), [](vector2D a1, vector2D a2) {
		return a1.x < a2.x;
		});

	int closestPtIndex = (int)distanceArr.grid[0].y;
	int secondClosestPtIndex = (int)distanceArr.grid[1].y;

	vector3DContainer triangle;
	triangle.grid.push_back(vertexGrid.grid[closestPtIndex]);
	triangle.grid.push_back(vertexGrid.grid[secondClosestPtIndex]);
	
	return triangle;
}


vector3D calculateNormal(vector3DContainer vertexGrid, vector3D refPt) {
	vector3DContainer triangleVertices = getClosestPointInGrid(vertexGrid, refPt);
	vector3D vec1;
	vec1.x = triangleVertices.grid[0].x - refPt.x;
	vec1.y = triangleVertices.grid[0].y - refPt.y;
	vec1.z = triangleVertices.grid[0].z - refPt.z;

	vector3D vec2;
	vec2.x = triangleVertices.grid[1].x - refPt.x;
	vec2.y = triangleVertices.grid[1].y - refPt.y;
	vec2.z = triangleVertices.grid[1].z - refPt.z;

	vector3D res = cross_product(vec1, vec2);
	res = normalize(res);
	return res;
}


vector3DContainer add_noiseToVertexGrid(renderDataContainer grid, GLfloat amplitude) {

	vector3DContainer flatVertexGrid = grid.vertexGrid;

	vector3DContainer noisyVertexGrid;

	GLfloat alpha1 = 2.4f, alpha2 = 0.8f;

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
	return noisyVertexGrid;
}

vector3DContainer calculate_noisyNormals(renderDataContainer grid) {

	vector3DContainer noisyVertexGrid = grid.vertexGrid;

	vector3DContainer noisyNormalGrid;
	for (auto ver3 = noisyVertexGrid.grid.begin(); ver3 != noisyVertexGrid.grid.end(); ++ver3) {
		vector3D noisyVertex = *ver3;
		vector3D noisyNormal = calculateNormal(noisyVertexGrid, noisyVertex);
		noisyNormalGrid.grid.push_back(noisyNormal);
	}

	return noisyNormalGrid;
}

renderDataContainer generateNoisySurface(GLfloat min_x,
	GLfloat min_y,
	GLfloat max_x,
	GLfloat max_y,
	GLfloat z_level,
	GLfloat amplitude) {

	renderDataContainer flatSurface = generateFlatSurface(min_x, min_y, max_x, max_y, z_level);
	vector3DContainer NoisyVertexGrid = add_noiseToVertexGrid(flatSurface, amplitude);
	vector3DContainer NoisyNormalGrid = calculate_noisyNormals(flatSurface);

	renderDataContainer noisySurface;
	noisySurface.normalGrid = NoisyNormalGrid;
	noisySurface.vertexGrid = NoisyVertexGrid;

	return noisySurface;
}

// draw the surface
// TODO : create directional rendering
void drawNoisySurface_LEGACY_GL(renderDataContainer surface) {

	vector3DContainer vertexField = surface.vertexGrid;
	vector3DContainer normalContainer = surface.normalGrid;

	int v_size = (int)vertexField.grid.size();
	int n_size = (int)normalContainer.grid.size();

	glBegin(GL_TRIANGLES);
	for (int i1 = 0, i2 = 0; (i1 < v_size) && (i2 < n_size); i1++, i2++) {
		vector3D vertex = vertexField.grid[i1];
		vector3D normal = normalContainer.grid[i2];
		glNormal3f(normal.x, normal.y, normal.z);
		glVertex3f(vertex.x, vertex.y, vertex.z);
	}
	glEnd();
}
