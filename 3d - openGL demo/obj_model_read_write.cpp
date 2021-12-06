#include <windows.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <GL/gl.h>
#include <GL/glu.h>
#include <stdlib.h>
#include <string>
#include "image_path_loader.h"

template<typename T>
std::vector<T> slice(std::vector<T> const& v, int m, int n)
{
	auto first = v.cbegin() + m;
	auto last = v.cbegin() + n + 1;

	std::vector<T> vec(first, last);
	return (n <= m) ? throw std::length_error(" vec_length < 0") : vec;
}

void eraseSubStr(std::string& mainStr, std::string& toErase)
{
	// Search for the substring in string
	size_t pos = mainStr.find(toErase);
	if (pos != std::string::npos)
	{
		// If found then erase it from string
		mainStr.erase(pos, toErase.length());
	}
}

// for info only...
void getModelName(const char* mainStr, std::string& objName) {

	// project path
	std::string basePath = "C:/Users/תמיר/source/repos/openGL textures and collision detection/textured objects/";

	// working with temporary strings - convert to string
	std::string tempName(mainStr);

	// remove the file path from the input string
	eraseSubStr(tempName, basePath);

	objName = tempName;
}

void printModelData(const char* OBJpath, bool TEXTURED) {

	// show the file opened
	std::string objectFile;
	getModelName(OBJpath, objectFile);

	printf("succeded loading data!! (: \n");
	std::cout << "loaded model: " << objectFile << "\n";
	std::cout << "accepted polygons:" << "GL_TRIANGLES or GL_QUADS or GL_TRIANGLE_STRIP or GL_QUAD_STRIP" << "\n";
	std::cout << "is textured:" << TEXTURED << "\n";
}

void clearDataStr(const char* dataID, std::string& dataStr) {

	// convert the ID to a string
	std::string ID(dataID);

	// data IDs
	std::string dID = ID + " ";

	// remove the ID from the input data string
	eraseSubStr(dataStr, dID);

	// check if there is another space - and delete it.
	if (dataStr[0] == ' ') {
		std::string space = " ";
		eraseSubStr(dataStr, space);
	}

}

bool getStrMatch(const char* dataID, const char* dataID2, std::string& dataStr1, std::string& dataStr2) {

	// work with temporary strings
	std::string tempDataStr1 = dataStr1;
	std::string tempDataStr2 = dataStr2;

	// convert the ID to a string
	std::string ID1(dataID);
	std::string ID2(dataID2);

	// data IDs
	std::string dID1 = ID1 + " ";
	std::string dID2 = ID2 + " ";

	// erase the ID to compare data strings
	eraseSubStr(tempDataStr1, dID1);
	eraseSubStr(tempDataStr2, dID2);

	// comparing
	if (!strcmp(tempDataStr1.c_str(), tempDataStr2.c_str())) {
		return true;
	}
	return false;
}

void getStrData(const char* dataID, std::string& lineStr, std::vector<GLfloat>& dataVec) {

	// work with temporary strings and vectors
	std::string tempDataStr = lineStr + " ";
	std::vector<GLfloat> tempData;

	// clean beginning fro IDs and spaces
	clearDataStr(dataID, tempDataStr);

	// now we should have a clean data string to read from
	// read the data until there is a space
	const char* tempStrArr = tempDataStr.c_str();
	int i = 0;
	int offset = 0;
	while (tempStrArr[i] != '\0') {
		if (tempStrArr[i] == ' ') {
			std::string subStr = tempDataStr.substr(offset, i - offset + 1);
			GLfloat tempNum = (GLfloat)std::stof(subStr);
			tempData.push_back(tempNum);

			offset += i - offset + 1;
		}
		i++;
	}

	// copy the temporary data to target vector
	dataVec = tempData;
}

void getStrAsData(const char* dataID, std::string& lineStr, std::string& dataStr) {

	// work with temporary strings
	std::string tempStr = lineStr;

	// clean beginning fro IDs and spaces
	clearDataStr(dataID, tempStr);

	// copy to target string
	dataStr = tempStr;
}

void getFitDataStr(const char* dataID, std::string& lineStr,
	std::vector<int>& Vindex, std::vector<int>& TexIndex, std::vector<int>& Nindex) {

	// work with temporary strings
	std::string tempStr = lineStr + " ";
	std::vector<int> tempV;
	std::vector<int> tempTex;
	std::vector<int> tempN;

	// clean beginning fro IDs and spaces
	clearDataStr(dataID, tempStr);

	// search for a substring of type "v/vt/vn"
	const char* tempStrArr = tempStr.c_str();
	int i = 0;
	int offset = 0;
	while (tempStrArr[i] != '\0') {
		if (tempStrArr[i] == ' ') {
			int subOffset = 0;
			int type = 0;
			std::string subStr = tempStr.substr(offset, i - offset);
			subStr = subStr + "/";
			for (std::string::size_type j = 0; j < subStr.size(); j++) {
				if (subStr[j] == '/') {
					std::string numStr = subStr.substr(subOffset, j - subOffset);
					int ind = std::stoi(numStr);
					if (type == 0) {
						tempV.push_back(ind);
					}
					else if (type == 1) {
						tempTex.push_back(ind);
					}
					else if (type == 2) {
						tempN.push_back(ind);
					}

					// changing the data to be stored
					type++;
					subOffset += j - subOffset + 1;
				}
			}

			offset += i - offset + 1;
		}
		i++;
	}

	// moving the data to containers
	Vindex = tempV;
	TexIndex = tempTex;
	Nindex = tempN;
}

void get_index_data_polygon_mapping(const char* dataID, std::string& lineStr,
	std::vector<int>& Vindex, std::vector<int>& TexIndex, std::vector<int>& Nindex, std::vector<int>& polygon_size) {

	// each line is a polygon - the length of each of the vectors - Vindex && TexIndex && Nindex
	// gives away the number of vertices of the polygon
	getFitDataStr(dataID, lineStr, Vindex, TexIndex, Nindex);
	polygon_size.push_back((int)Vindex.size());
}

void dataSort(std::vector<std::vector<std::vector<GLfloat>>>& tempMaterials,
	std::vector<std::string>& tempTextureMaps,
	std::vector<bool>& temp_texStatus,
	std::vector<std::vector<unsigned int>>& vertexIndices,
	std::vector<std::vector<unsigned int>>& UVsIndices,
	std::vector<std::vector<unsigned int>>& normalIndices,
	std::vector<std::vector<std::vector<GLfloat>>>& temp_vetrices,
	std::vector<std::vector<std::vector<GLfloat>>>& temp_UVs,
	std::vector<std::vector<std::vector<GLfloat>>>& temp_normals,
	std::vector<std::vector<std::vector<GLfloat>>>& materialModel,
	std::vector<std::string>& texMaps,
	std::vector<bool>& isTextured,
	std::vector<std::vector<std::vector<GLfloat>>>& vetrices,
	std::vector<std::vector<std::vector<GLfloat>>>& UVs,
	std::vector<std::vector<std::vector<GLfloat>>>& normals,
	bool TEXTURED) {

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

			if (TEXTURED) {
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
}


bool read_OBJ_models_LEGACY_GL(const char* OBJpath,
	const char* MTLpath,
	std::vector<std::vector<std::vector<GLfloat>>>& materialModel,
	std::vector<std::string>& texMaps,
	std::vector<std::vector<std::vector<GLfloat>>>& vetrices,
	std::vector<std::vector<std::vector<GLfloat>>>& UVs,
	std::vector<std::vector<std::vector<GLfloat>>>& normals,
	std::vector<int>& polygon_size,
	bool TEXTURED,
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

	// reading to temporary vectors
	int objectNum = 0;
	std::vector<unsigned int> tempVertexIndices, tempUVsIndices, tempNormalIndices;
	std::vector<std::vector<GLfloat>> tempVetrexBuffer;
	std::vector<std::vector<GLfloat>> tempUVsBuffer;
	std::vector<std::vector<GLfloat>> tempNormalsBuffer;
	std::vector<std::vector<GLfloat>> tempMaterialBuffer(6, std::vector<GLfloat>(3, 0));
	std::string textureMapPic;
	bool texStatus = false;

	// reading the files
	std::ifstream OBJfile(OBJpath, std::ios::in);

	if (OBJfile.is_open()) {
		std::cout << "OBJECT file opened (.obj)" << "\n";
		// checking if the data is represented as groups/objects of (v, vt, vn, f) for each object.
		// if not, the data is parsed in a different way from usual.

		std::string line;
		while (std::getline(OBJfile, line)) {
			// starting to read the main .OBJ file
			if (line.find("vertex positions") != std::string::npos) {
				// wrong format - different mathod
				// function - fix fromat
				// if fixed - just read regularly
				std::cout << "error in formatting" << "\n";
				return false;
			}
			else {
				// starting to read from "good" format - division of (v, vt, vn, f) for each group
					// right format (probably...)

					// reaing from .OBJ file
				if (line.find("g ") != std::string::npos || line.find("o ") != std::string::npos) {
					// found a group/object - if there are many (> 1) objects - push back data
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
							temp_texStatus.push_back(texStatus);
							tempUVsBuffer.clear();
							tempUVsIndices.clear();
							tempMaterialBuffer.clear();
						}
					}

					// move to next object
					objectNum++;
				}

				// if textures enabled - there is a reference to an mtl file 
				// reading the .MTL file
				else if (line.find("usemtl") != std::string::npos) {
					texStatus = false;
					std::ifstream MTLfile(MTLpath, std::ios::in);

					if (MTLfile.fail()) {
						printf("Impossible to open the file !\n");
						return false;
					}

					if (MTLfile.is_open()) {
						std::cout << "MATERIAL file opened (.mtl)" << "\n";

						std::string MTLline;
						bool readData = false;
						while (std::getline(MTLfile, MTLline)) {
							if (readData) {
								if (MTLline.find("Ka") != std::string::npos) {
									std::vector<GLfloat> Ka;
									getStrData("Ka", MTLline, Ka);
									tempMaterialBuffer[0][0] = Ka[0];
									tempMaterialBuffer[0][1] = Ka[1];
									tempMaterialBuffer[0][2] = Ka[2];
								}
								else if (MTLline.find("Kd") != std::string::npos &&
									MTLline.find("map_Kd") == std::string::npos) {
									std::vector<GLfloat> Kd;
									getStrData("Kd", MTLline, Kd);
									tempMaterialBuffer[1][0] = Kd[0];
									tempMaterialBuffer[1][1] = Kd[1];
									tempMaterialBuffer[1][2] = Kd[2];
								}
								else if (MTLline.find("Ks") != std::string::npos) {
									std::vector<GLfloat> Ks;
									getStrData("Ks", MTLline, Ks);
									tempMaterialBuffer[2][0] = Ks[0];
									tempMaterialBuffer[2][1] = Ks[1];
									tempMaterialBuffer[2][2] = Ks[2];
								}
								else if (MTLline.find("Ns") != std::string::npos) {
									std::vector<GLfloat> Ns;
									getStrData("Ns", MTLline, Ns);
									tempMaterialBuffer[3][0] = Ns[0];
									tempMaterialBuffer[3][1] = 0.0;
									tempMaterialBuffer[3][2] = 0.0;
								}
								else if (MTLline.find("Ni") != std::string::npos) {
									std::vector<GLfloat> Ni;
									getStrData("Ni", MTLline, Ni);
									tempMaterialBuffer[4][0] = Ni[0];
									tempMaterialBuffer[4][1] = 0.0;
									tempMaterialBuffer[4][2] = 0.0;
								}
								else if (MTLline.find("d") != std::string::npos &&
									MTLline.find("map_Kd") == std::string::npos &&
									MTLline.find("Kd") == std::string::npos) {
									std::vector<GLfloat> d;
									getStrData("d", MTLline, d);
									tempMaterialBuffer[5][0] = d[0];
									tempMaterialBuffer[5][1] = 0.0;
									tempMaterialBuffer[5][2] = 0.0;
								}
								else if (MTLline.find("map_Kd") != std::string::npos) {
									getStrAsData("map_Kd", MTLline, textureMapPic);
									texStatus = true;
								}
							}
							if (MTLline.find("newmtl") != std::string::npos) {
								if (getStrMatch("usemtl", "newmtl", line, MTLline)) {
									readData = true;
								}
								else {
									readData = false;
								}
							}
						}
					}
					MTLfile.close();
				}
				else if (line.find("v ") != std::string::npos) {
					std::vector<GLfloat> vertex(3);
					getStrData("v", line, vertex);
					tempVetrexBuffer.push_back(vertex);
				}
				else if (line.find("vt ") != std::string::npos) {
					std::vector<GLfloat> UV;
					getStrData("vt", line, UV);
					std::vector<GLfloat> UVsClean;
					UVsClean = slice(UV, 0, 1);
					tempUVsBuffer.push_back(UVsClean);
				}
				else if (line.find("vn ") != std::string::npos) {
					std::vector<GLfloat> normal(3);
					getStrData("vn", line, normal);
					tempNormalsBuffer.push_back(normal);
				}
				else if (line.find("f ") != std::string::npos) {
					std::vector<int> Vind;
					std::vector<int> UVind;
					std::vector<int> Nind;
					get_index_data_polygon_mapping("f", line, Vind, UVind, Nind, polygon_size);

					// extrating the data
					for (unsigned int i = 0; i < Vind.size(); i++) {
						tempVertexIndices.push_back(Vind[i]);
						if (texStatus) { tempUVsIndices.push_back(UVind[i]); }
						else { tempUVsIndices.push_back(0); }
						tempNormalIndices.push_back(Nind[i]);
					}
				}
				// end of "good" file format readig
			}
			// end if .OBJ file reading
		}
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
	}

	else if (OBJfile.fail()) {
		printf("Impossible to open the file !\n");
		return false;
	}
	OBJfile.close();

	// data loading in the right order (according to indices)
	dataSort(tempMaterials, tempTextureMaps, temp_texStatus,
		vertexIndices, UVsIndices, normalIndices, temp_vetrices,
		temp_UVs, temp_normals, materialModel, texMaps, isTextured,
		vetrices, UVs, normals, TEXTURED);

	printModelData(OBJpath, TEXTURED);
	return true;
}

// correct assets that are convertet from .blend to .obj
// the function reads the raw data and rewrites it into another file in a correct format.
bool write_asset_data_LEGACY_GL(const char* OBJpath, std::string& new_OBJ_path) {

	std::vector<std::string> vertex_lines;
	std::vector<std::string> texture_lines;
	std::vector<std::string> normal_lines;
	std::vector<std::vector<std::string>> index_lines;

	std::vector<std::string> obj_lines;
	std::vector<std::string> mtl_lines;

	std::vector<std::vector<int>> Vindex;
	std::vector<std::vector<int>> TexIndex;
	std::vector<std::vector<int>> Nindex;

	unsigned int obj = 0;

	// reading the main .obj file
	std::ifstream OBJfile(OBJpath, std::ios::in);

	if (OBJfile.is_open()) {
		std::cout << "reading from OBJECT file (.obj)" << "\n";
		std::string line;
		while (std::getline(OBJfile, line)) {
			if (line.find("v ") != std::string::npos) {
				vertex_lines.push_back(line);
			}
			else if (line.find("vt ") != std::string::npos) {
				texture_lines.push_back(line);
			}
			else if (line.find("vn ") != std::string::npos) {
				normal_lines.push_back(line);
			}
			else if (line.find("g ") != std::string::npos || line.find("o ") != std::string::npos) {
				obj_lines.push_back(line);
				obj++;
			}
			else if (line.find("usemtl") != std::string::npos) {
				mtl_lines.push_back(line);
			}
			else if (line.find("f ") != std::string::npos) {
				index_lines[obj - 1].push_back(line);

				std::vector<int> Vind;
				std::vector<int> UVind;
				std::vector<int> Nind;
				getFitDataStr("f", line, Vind, UVind, Nind);

				for (unsigned int i = 0; i < Vind.size(); i++) {
					Vindex[obj - 1].push_back(Vind[i]);
					if (!UVind.empty()) { TexIndex[obj - 1].push_back(UVind[i]); }
					else { TexIndex[obj - 1].push_back(0); }
					Nindex[obj - 1].push_back(Nind[i]);
				}
			}
		}
	}
	else if (OBJfile.fail()) {
		printf("Impossible to open the file !\n");
		return false;
	}
	OBJfile.close();

	// write to a second file
	std::string write_file_name("correct_file_format.obj");
	buildImagePath(write_file_name, new_OBJ_path);
	std::ofstream newFile(new_OBJ_path.c_str(), std::ios::out);

	if (newFile.is_open()) {
		std::cout << "writing to OBJECT file (.obj)" << "\n";
		for (unsigned int m = 0; m < obj; m++) {
			newFile << obj_lines[m] << "\n";
			newFile << mtl_lines[m] << "\n";
			for (unsigned int k1 = 0; k1 < Vindex[m].size(); k1++) {
				int index = Vindex[m][k1];
				newFile << vertex_lines[index - 1] << "\n";
			}
			for (unsigned int k2 = 0; k2 < TexIndex[m].size(); k2++) {
				int tex_index = TexIndex[m][k2];
				if (tex_index != 0) { newFile << texture_lines[tex_index - 1] << "\n"; }
			}
			for (unsigned int k3 = 0; k3 < Nindex[m].size(); k3++) {
				int n_index = Nindex[m][k3];
				newFile << normal_lines[n_index - 1] << "\n";
			}
			for (unsigned int k4 = 0; k4 < index_lines[m].size(); k4++) {
				newFile << index_lines[m][k4] << "\n";
			}
		}
	}
	else if (newFile.fail()) {
		printf("Impossible to open the file !\n");
		return false;
	}
	newFile.close();
	return true;
}

bool load_model_data_LEGACY_GL(const char* OBJpath,
	const char* MTLpath,
	std::vector<std::vector<std::vector<GLfloat>>>& materialModel,
	std::vector<std::string>& texMaps,
	std::vector<std::vector<std::vector<GLfloat>>>& vetrices,
	std::vector<std::vector<std::vector<GLfloat>>>& UVs,
	std::vector<std::vector<std::vector<GLfloat>>>& normals,
	std::vector<int>& polygon_size,
	bool TEXTURED,
	std::vector<bool>& isTextured) {

	bool CORRECT_FORMAT = true;

	// reading the main .obj file
	std::ifstream file(OBJpath, std::ios::in);
	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {
			if (line.find("vertex positions") != std::string::npos) {
				CORRECT_FORMAT = false;
				break;
			}
			else if (line.find("g ") != std::string::npos || line.find("o ") != std::string::npos) {
				CORRECT_FORMAT = true;
				break;
			}
		}
	}
	else if (file.fail()) {
		printf("Impossible to open the file !\n");
		return false;
	}
	file.close();

	if (CORRECT_FORMAT) {
		return read_OBJ_models_LEGACY_GL(OBJpath, MTLpath, materialModel, texMaps, vetrices, UVs, normals, polygon_size, TEXTURED, isTextured);
	}
	else {
		std::string new_path;
		if (write_asset_data_LEGACY_GL(OBJpath, new_path)) {
			const char* newP = new_path.c_str();
			return read_OBJ_models_LEGACY_GL(newP, MTLpath, materialModel, texMaps, vetrices, UVs, normals, polygon_size, TEXTURED, isTextured);
		}
	}
	return false;
}
