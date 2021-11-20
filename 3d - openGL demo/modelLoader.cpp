#include <windows.h>
#include <iostream>
#include <vector>
#include <GL/gl.h>
#include <GL/glu.h>
#include <stdlib.h>
#include <string>
#include <errno.h>

#include "objectParser.h"

bool loadData(const char* OBJpath, const char* MTLpath, std::vector<std::vector<std::vector<GLfloat>>>& materialModel, 
	std::vector<std::string>& texMaps, std::vector<std::vector<std::vector<GLfloat>>>& vetrices,
	std::vector<std::vector<std::vector<GLfloat>>>& UVs, std::vector<std::vector<std::vector<GLfloat>>>& normals, bool TEXTURED,
	std::vector<bool>& isTextured) {

	// temps for later adjusting values to correct indices - these are 2d vectors - one for each "miniature" object
	std::vector<std::vector<unsigned int>> vertexIndices, UVsIndices, normalIndices;
	std::vector<std::vector<std::vector<GLfloat>>> temp_vetrices;
	std::vector<std::vector<std::vector<GLfloat>>> temp_UVs;
	std::vector<std::vector<std::vector<GLfloat>>> temp_normals;

	// for textures
	std::vector<std::vector<std::vector<GLfloat>>> tempMaterials;
	std::vector<std::string> tempTextureMaps;
	std::vector<bool> temp_texStatus;

	// opening the obj. file
	FILE* file;
	errno_t err = fopen_s(&file, OBJpath, "r");
	if (err) {
		printf("Impossible to open the file !\n");
		return false;
	}

	// opening the .mtl file
	FILE* MTL;
	errno_t mtlErr = fopen_s(&MTL, MTLpath, "r");
	if (mtlErr) {
		printf("Impossible to open the file !\n");
		return false;
	}

	// reading to temporary vectors
	int objectNum = 0;
	std::vector<unsigned int> tempVertexIndices, tempUVsIndices, tempNormalIndices;
	std::vector<std::vector<GLfloat>> tempVetrexBuffer;
	std::vector<std::vector<GLfloat>> tempUVsBuffer;
	std::vector<std::vector<GLfloat>> tempNormalsBuffer;
	std::vector<std::vector<GLfloat>> tempMaterialBuffer(6, std::vector<GLfloat>(3,0));
	std::string textureMapPic;
	bool texStatus = false;

	// loop over the files
	while (1) {
		char lineHeader[128];
		lineHeader[127] = '\0';

		int firstWord = fscanf_s(file, "%s", lineHeader, 128);
		if (firstWord == EOF) {
			// store all accumulated vertices, faces and normals 
			temp_vetrices.push_back(tempVetrexBuffer);
			temp_normals.push_back(tempNormalsBuffer);
			vertexIndices.push_back(tempVertexIndices);
			normalIndices.push_back(tempNormalIndices);

			if (TEXTURED) {
				temp_UVs.push_back(tempUVsBuffer);
				UVsIndices.push_back(tempUVsIndices);
				tempMaterials.push_back(tempMaterialBuffer);
				tempTextureMaps.push_back(textureMapPic);
				temp_texStatus.push_back(texStatus);
			}
			break;
		}

		// initialize temporary containers and store previous data into data containers.
		// transition to a new object
		if (!strcmp(lineHeader, "g") | !strcmp(lineHeader, "o")) {
			if (objectNum > 0) {
				// store all accumulated vertices, faces and normals 
				temp_vetrices.push_back(tempVetrexBuffer);
				temp_normals.push_back(tempNormalsBuffer);
				vertexIndices.push_back(tempVertexIndices);
				normalIndices.push_back(tempNormalIndices);

				// initialize the buffer vectors
				tempVetrexBuffer.clear();
				tempNormalsBuffer.clear();
				tempVertexIndices.clear();
				tempNormalIndices.clear();

				// if textures are implied, store all accumulated UV vectors, and clear buffers
				if (TEXTURED) {
					temp_UVs.push_back(tempUVsBuffer);
					UVsIndices.push_back(tempUVsIndices);
					tempMaterials.push_back(tempMaterialBuffer);
					tempTextureMaps.push_back(textureMapPic);
					tempUVsBuffer.clear();
					tempUVsIndices.clear();
					tempMaterialBuffer.clear();
				}
			}

			// move to next object
			objectNum++;
		}
		// if textures enabled - there is a reference to an mtl file 
		else if (!strcmp(lineHeader, "usemtl")) {
			// reading the correct material map to read from the corresponding MTL file
			char materialMap[128];
			materialMap[127] = '\0';
			fscanf_s(file, "%s\n", materialMap, 128);

			// read first word in lines of the MTL file
			// materialId - for finding the right material for the corresponding object
			bool readData = false;
			while (1) {
				char MTLine[128];
				MTLine[127] = '\0';

				int first = fscanf_s(MTL, "%s", MTLine, 128);
				// if the file is ended or we already found our material - break
				if (first == EOF ) {
					break;
				}

				// read data if needed
				if (readData) {
					if (!strcmp(MTLine, "Ka")) {
						GLfloat Ka[3];
						fscanf_s(MTL, "%f %f %f\n", &Ka[0], &Ka[1], &Ka[2]);
						tempMaterialBuffer[0][0] = Ka[0];
						tempMaterialBuffer[0][1] = Ka[1];
						tempMaterialBuffer[0][2] = Ka[2];
					}
					else if (!strcmp(MTLine, "Kd")) {
						GLfloat Kd[3];
						fscanf_s(MTL, "%f %f %f\n", &Kd[0], &Kd[1], &Kd[2]);
						tempMaterialBuffer[1][0] = Kd[0];
						tempMaterialBuffer[1][1] = Kd[1];
						tempMaterialBuffer[1][2] = Kd[2];
					}
					else if (!strcmp(MTLine, "Ks")) {
						GLfloat Ks[3];
						fscanf_s(MTL, "%f %f %f\n", &Ks[0], &Ks[1], &Ks[2]);
						tempMaterialBuffer[2][0] = Ks[0];
						tempMaterialBuffer[2][1] = Ks[1];
						tempMaterialBuffer[2][2] = Ks[2];
					}
					else if (!strcmp(MTLine, "Ns")) {
						GLfloat Ns;
						fscanf_s(MTL, "%f\n", &Ns);
						tempMaterialBuffer[3][0] = Ns;
						tempMaterialBuffer[3][1] = 0.0;
						tempMaterialBuffer[3][2] = 0.0;
					}
					else if (!strcmp(MTLine, "Ni")) {
						GLfloat Ni;
						fscanf_s(MTL, "%f\n", &Ni);
						tempMaterialBuffer[4][0] = Ni;
						tempMaterialBuffer[4][1] = 0.0;
						tempMaterialBuffer[4][2] = 0.0;
					}
					else if (!strcmp(MTLine, "d")) {
						GLfloat d;
						fscanf_s(MTL, "%f\n", &d);
						tempMaterialBuffer[5][0] = d;
						tempMaterialBuffer[5][1] = 0.0;
						tempMaterialBuffer[5][2] = 0.0;
					}
					else if (!strcmp(MTLine, "map_Kd")) {
						char mapPic[128];
						mapPic[127] = '\0';
						fscanf_s(MTL, "%s\n", mapPic, 128);
						textureMapPic = mapPic;
						texStatus = true;
					}
				}
				// scaning the mtl data
				else if (!strcmp(MTLine, "newmtl")) {
					char materialType[128];
					materialType[127] = '\0';
					fscanf_s(MTL, "%s\n", materialType, 128);
					// if we have found the correct identifier - read the data
					if (!strcmp(materialType, materialMap)) {
						readData = true;
					}
				}
			}
		}
		// read vertex data
		else if (!strcmp(lineHeader, "v")) {
			std::vector<GLfloat> vertex(3);
			fscanf_s(file, "%f %f %f\n", &vertex[0], &vertex[1], &vertex[2]);
			tempVetrexBuffer.push_back(vertex);
		}
		// read UV texture data if textures exist
		else if (!strcmp(lineHeader, "vt")) {
			std::vector<GLfloat> UV(2);
			GLfloat fillerUV;
			fscanf_s(file, "%f %f %f\n", &UV[0], &UV[1], &fillerUV);
			tempUVsBuffer.push_back(UV);
		}
		// read normal data
		else if (!strcmp(lineHeader, "vn")) {
			std::vector<GLfloat> normal(3);
			fscanf_s(file, "%f %f %f\n", &normal[0], &normal[1], &normal[2]);
			tempNormalsBuffer.push_back(normal);
		}
		// reader for 3 coordinates **triangles only!!!**
		else if (!strcmp(lineHeader, "f")) {
			// vertex UVs and normal arrays
			unsigned int vertexIndex3[3], UVsIndex3[3], normalIndex3[3];
			int matchesTriangular;

			// reading the indices
			if (TEXTURED) {
				matchesTriangular = fscanf_s(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex3[0], &UVsIndex3[0],
					&normalIndex3[0], &vertexIndex3[1], &UVsIndex3[1], &normalIndex3[1], &vertexIndex3[2],
					&UVsIndex3[2], &normalIndex3[2]);
			}
			else {
				matchesTriangular = fscanf_s(file, "%d//%d %d//%d %d//%d\n", &vertexIndex3[0],
					&normalIndex3[0], &vertexIndex3[1], &normalIndex3[1], &vertexIndex3[2], &normalIndex3[2]);
			}

			// checking for the right amount of assignments
			if (matchesTriangular == 6 && !TEXTURED) {
				for (int i = 0; i < sizeof(vertexIndex3) / sizeof(*vertexIndex3); i++) {
					tempVertexIndices.push_back(vertexIndex3[i]);
					tempNormalIndices.push_back(normalIndex3[i]);
				}
			}
			else if (matchesTriangular == 9 && TEXTURED) {
				for (int i = 0; i < sizeof(vertexIndex3) / sizeof(*vertexIndex3); i++) {
					tempVertexIndices.push_back(vertexIndex3[i]);
					tempUVsIndices.push_back(UVsIndex3[i]);
					tempNormalIndices.push_back(normalIndex3[i]);
				}
			}
			else {
				printf("File can't be read by this simple loader ):");
				std::cout << matchesTriangular;
				return false;
			}
		}
	}

	// moving the temporal texture data to inputs
	if (TEXTURED) {
		materialModel = tempMaterials;
		texMaps = tempTextureMaps;
		isTextured = temp_texStatus;
	}

	// calculating the offset of the face indices for each loop iteration
	int vertexOffset = 0;
	int UVsOffset = 0;
	int normalOffset = 0;

	// looping over all objects in the model
	for (unsigned int j = 0; j < vertexIndices.size(); j++) {
		/* vertex data */
		// vertex vector that contains the vertices of each object
		std::vector<std::vector<GLfloat>> verticesPerObject;
		// vertex index vector for each object
		std::vector<unsigned int> objectVertexIndices = vertexIndices[j];

		/* UV texture position data */
		// for definition
		// UV vector that containes the texture coordinates of each object
		std::vector<std::vector<GLfloat>> UVcoordsPerObject;
		std::vector<unsigned int> objectUVcoordsIndices;

		if (TEXTURED) {
			// UV index vector for each object
			objectUVcoordsIndices = UVsIndices[j];
		}

		/* normal data */
		// normal vector that contains the normals corresponding to each vertex vector
		std::vector<std::vector<GLfloat>> normalsPerObject;
		// normal index vector for each object
		std::vector<unsigned int> objectNormalIndices = normalIndices[j];

		//looping over all vertices on the same object
		for (unsigned int k = 0; k < objectVertexIndices.size(); k++) {
			// vertex index
			unsigned int vertexInd = objectVertexIndices[k];
			verticesPerObject.push_back(temp_vetrices[j][vertexInd - vertexOffset - 1]);

			if(TEXTURED) {
				// UV index
				unsigned int UVsInd = objectUVcoordsIndices[k];
				UVcoordsPerObject.push_back(temp_UVs[j][UVsInd - UVsOffset - 1]);
			}

			// normal index
			unsigned int normalInd = objectNormalIndices[k];
			normalsPerObject.push_back(temp_normals[j][normalInd - normalOffset - 1]);
		}
		// outputing indexed vertices, UVs and normals per object
		vetrices.push_back(verticesPerObject);
		normals.push_back(normalsPerObject);

		// update the offset
		vertexOffset += temp_vetrices[j].size();
		normalOffset += temp_normals[j].size();

		if (TEXTURED) {
			UVs.push_back(UVcoordsPerObject);
			UVsOffset += temp_UVs[j].size();
		}
	}
	printf("succeded loading data!! (: \n");
	std::cout << "loaded model:" << "" << "\n";
	std::cout << "accepted polygons:" << "GL_TRIANGLES" << "\n";
	std::cout << "is textured:" << TEXTURED << "\n";
	return true;
}

// NOTE: the normals from blender are usually the FACE normals and not vertex normals. 
// therefore, one vertex can be assosiated with many different normals - depending on the number of polygons 
// containing that particular vertex.
// FIX: map all normals of one vertex for each of the vertices, for each vertex calculate the normalized 
// vector that is created from linear combination of the assosiated normals, i.e:
// k - number of assosiated normals to vertex v
// {f_normal_1, f_normal_2, ..., f_normal_k} - set of all normals assosiated to vertex v
// v_tot = f_normal_1 + f_normal_2 +...+ f_normal_k - linear combination of the set elements
// ||v_tot||_2 = sqrt(v_tot.x^2 + v_tot.y^2 + v_tot.z^2) - euclidean norm
// v_normal = v_tot/||v_tot||_2 - the normalized vertex normal