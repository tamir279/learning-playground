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


vector3DContainer getGridIndices(int x_s, int y_s) {

	vector3DContainer polygonIndices;
	int M_SIZE = x_s * y_s;

	for (int i = 0; i < M_SIZE - y_s; i++) {

		if ((i + 1) % y_s == 0 && i > 0) {
			continue;
		}
		else {
			// first triange
			vector3D triangleIndices;

			triangleIndices.x = (GLfloat)i;
			triangleIndices.y = (GLfloat)i + (GLfloat)y_s;
			triangleIndices.z = (GLfloat)i + (GLfloat)y_s + 1;

			polygonIndices.grid.push_back(triangleIndices);

			// second triangle
			triangleIndices.x = (GLfloat)i;
			triangleIndices.y = (GLfloat)i + 1;
			triangleIndices.z = (GLfloat)i + (GLfloat)y_s + 1;

			polygonIndices.grid.push_back(triangleIndices);
		}
	}
	return polygonIndices;
}

renderDataContainer generateFlatSurface(GLfloat min_x, GLfloat min_y, GLfloat max_x, GLfloat max_y, GLfloat z_level) {

	vector3DContainer flatMesh;
	vector3DContainer normals;

	int x_size = 0, y_size = 0;
	GLfloat x_pt = min_x;
	GLfloat y_pt = min_y;
	vector3D point;
	vector3D normal;

	while (x_pt < max_x) {
		y_pt = min_y; y_size = 0;
		while (y_pt < max_y) {

			point.x = x_pt;
			point.y = y_pt;
			point.z = z_level;

			normal.x = 0.0;
			normal.y = 0.0;
			normal.z = 1.0;

			flatMesh.grid.push_back(point);
			normals.grid.push_back(normal);

			y_pt += DY; y_size++;
		}
		x_pt += DX; x_size++;
	}

	renderDataContainer flatSurface;
	flatSurface.vertexGrid = flatMesh;
	flatSurface.normalGrid = normals;
	flatSurface.GridIndices = getGridIndices(x_size, y_size);

	return flatSurface;
}

float euclideanDistance(vector3D pt1, vector3D pt2) {

	float dist_sq = (pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y) + (pt1.z - pt2.z) * (pt1.z - pt2.z);
	return fast_sqrt(dist_sq);
}

// generate noise vector
vector3D noise(double mean, double s_d) {

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::normal_distribution<double> distribution(mean, s_d);

	vector3D noise;
	noise.z = (GLfloat)distribution(generator);

	return noise;
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


vector3DContainer add_noiseToVertexGrid(renderDataContainer grid) {

	vector3DContainer flatVertexGrid = grid.vertexGrid;

	vector3DContainer noisyVertexGrid;

	GLfloat alpha1 = 2.4f, alpha2 = 0.8f, alpha3 = 1.4f, alpha4 = 0.2f;

	for (auto vector3 = flatVertexGrid.grid.begin(); vector3 != flatVertexGrid.grid.end(); ++vector3) {
		vector3D v = *vector3;
		// randomly sample noise vector 
		vector3D n1 = noise(2.0, 0.5);
		vector3D n2 = noise(0.6, 0.2);
		vector3D n3 = noise(5.0, 2.5);

		vector3D linearCombination;

		// it can be better if the combination would be with more axis (y, z)
		linearCombination.x = v.x;
		linearCombination.y = v.y;
		linearCombination.z = alpha1 * n2.z - alpha2 * n1.z + alpha3 * v.z - alpha4 * n3.z;

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
	GLfloat z_level) {

	renderDataContainer flatSurface = generateFlatSurface(min_x, min_y, max_x, max_y, z_level);
	vector3DContainer NoisyVertexGrid = add_noiseToVertexGrid(flatSurface);
	vector3DContainer NoisyNormalGrid = calculate_noisyNormals(flatSurface);

	renderDataContainer noisySurface;
	noisySurface.normalGrid = NoisyNormalGrid;
	noisySurface.vertexGrid = NoisyVertexGrid;
	noisySurface.GridIndices = flatSurface.GridIndices;

	return noisySurface;
}

// n is the number of tiles on each axis (odd), the function creates a squared tiling
surfaceTileContainer generateSurfaceTiles(int n, GLfloat z, bool surfaceType) {
	surfaceTileContainer tiles;
	// run on x axis
	for (int i = 0; i < n; i++) {
		// run on y axis
		for (int j = 0; j < n; j++) {
			int min_x = -n + 2 * i, min_y = -n + 2 * j;
			int max_x = -n + 2 * (i + 1), max_y = -n + 2 * (j + 1);

			GLfloat mx = (GLfloat)min_x - (GLfloat)0.25, my = (GLfloat)min_y - (GLfloat)0.25;
			GLfloat mxx = (GLfloat)max_x + (GLfloat)0.25, mxy = (GLfloat)max_y + (GLfloat)0.25;
			renderDataContainer surface;
			if(surfaceType){ surface = generateNoisySurface(mx, my, mxx, mxy, z); }
			else{ surface = generateFlatSurface(mx, my, mxx, mxy, z); }
			tiles.tileContainer.push_back(surface);
		}
	}
	return tiles;
}

// draw the surface
// TODO : create directional rendering
void drawNoisySurface_LEGACY_GL(renderDataContainer surface, GLenum type) {

	vector3DContainer vertexField = surface.vertexGrid;
	vector3DContainer normalContainer = surface.normalGrid;
	vector3DContainer indices = surface.GridIndices;

	int indexArraySize = (int)indices.grid.size();

	glBegin(type);
	for (int i = 0; i < indexArraySize; i++) {
		vector3D ind = indices.grid[i];

		vector3D normal1 = normalContainer.grid[(int)ind.x];
		vector3D vertex1 = vertexField.grid[(int)ind.x];

		vector3D normal2 = normalContainer.grid[(int)ind.y];
		vector3D vertex2 = vertexField.grid[(int)ind.y];

		vector3D normal3 = normalContainer.grid[(int)ind.z];
		vector3D vertex3 = vertexField.grid[(int)ind.z];

		glNormal3f(normal1.x, normal1.y, normal1.z);
		glVertex3f(vertex1.x, vertex1.y, vertex1.z);

		glNormal3f(normal2.x, normal2.y, normal2.z);
		glVertex3f(vertex2.x, vertex2.y, vertex2.z);

		glNormal3f(normal3.x, normal3.y, normal3.z);
		glVertex3f(vertex3.x, vertex3.y, vertex3.z);
	}
	glEnd();
}

void draw_surfaceTiles(surfaceTileContainer surface_tiles, GLenum type) {
	surfaceTileContainer tmp = surface_tiles;
	for (auto tile = tmp.tileContainer.begin(); tile != tmp.tileContainer.end(); ++tile) {
		renderDataContainer surface_tile = *tile;
		drawNoisySurface_LEGACY_GL(surface_tile, type);
	}
}
