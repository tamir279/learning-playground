#pragma once
#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

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
std::vector<T> slice(std::vector<T> const& v, int m, int n);
void eraseSubStr(std::string& mainStr, std::string& toErase);
void getModelName(const char* mainStr, std::string& objName);
void printModelData(const char* OBJpath, bool TEXTURED);
void clearDataStr(const char* dataID, std::string& dataStr);
bool getStrMatch(const char* dataID, const char* dataID2, std::string& dataStr1, std::string& dataStr2);
void getStrData(const char* dataID, std::string& lineStr, std::vector<GLfloat>& dataVec);
void getStrAsData(const char* dataID, std::string& lineStr, std::string& dataStr);
void getFitDataStr(const char* dataID, std::string& lineStr, std::vector<int>& Vindex, std::vector<int>& TexIndex, std::vector<int>& Nindex);
void get_index_data_polygon_mapping(const char* dataID, std::string& lineStr, std::vector<int>& Vindex, std::vector<int>& TexIndex, std::vector<int>& Nindex, std::vector<int>& polygon_size);
void dataSort(std::vector<std::vector<std::vector<GLfloat>>>& tempMaterials, std::vector<std::string>& tempTextureMaps, std::vector<bool>& temp_texStatus, std::vector<std::vector<unsigned int>>& vertexIndices, std::vector<std::vector<unsigned int>>& UVsIndices, std::vector<std::vector<unsigned int>>& normalIndices, std::vector<std::vector<std::vector<GLfloat>>>& temp_vetrices, std::vector<std::vector<std::vector<GLfloat>>>& temp_UVs, std::vector<std::vector<std::vector<GLfloat>>>& temp_normals, std::vector<std::vector<std::vector<GLfloat>>>& materialModel, std::vector<std::string>& texMaps, std::vector<bool>& isTextured, std::vector<std::vector<std::vector<GLfloat>>>& vetrices, std::vector<std::vector<std::vector<GLfloat>>>& UVs, std::vector<std::vector<std::vector<GLfloat>>>& normals, bool TEXTURED);
bool read_OBJ_models_LEGACY_GL(const char* OBJpath, const char* MTLpath, std::vector<std::vector<std::vector<GLfloat>>>& materialModel, std::vector<std::string>& texMaps, std::vector<std::vector<std::vector<GLfloat>>>& vetrices, std::vector<std::vector<std::vector<GLfloat>>>& UVs, std::vector<std::vector<std::vector<GLfloat>>>& normals, std::vector<int>& polygon_size, bool TEXTURED, std::vector<bool>& isTextured);
bool write_asset_data_LEGACY_GL(const char* OBJpath, std::string& new_OBJ_path);
bool load_model_data_LEGACY_GL(const char* OBJpath, const char* MTLpath, std::vector<std::vector<std::vector<GLfloat>>>& materialModel, std::vector<std::string>& texMaps, std::vector<std::vector<std::vector<GLfloat>>>& vetrices, std::vector<std::vector<std::vector<GLfloat>>>& UVs, std::vector<std::vector<std::vector<GLfloat>>>& normals, std::vector<int>& polygon_size, bool TEXTURED, std::vector<bool>& isTextured);

#endif