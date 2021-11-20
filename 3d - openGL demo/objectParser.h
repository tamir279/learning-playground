#pragma once
#ifndef OBJECT_PARSER_H  
#define OBJECT_PARSER_H

#include <windows.h>
#include <iostream>
#include <vector>
#include <GL/gl.h>
#include <GL/glu.h>
#include <stdlib.h>
#include <errno.h>
#include <string>

bool loadData(const char* OBJpath,
	const char* MTLpath,
	std::vector<std::vector<std::vector<GLfloat>>>& materialModel,
	std::vector<std::string>& texMaps,
	std::vector<std::vector<std::vector<GLfloat>>>& vetrices,
	std::vector<std::vector<std::vector<GLfloat>>>& UVs,
	std::vector<std::vector<std::vector<GLfloat>>>& normals,
	bool TEXTURED,
	std::vector<bool>& isTextured);

bool loadDataV2(const char* OBJpath,
	const char* MTLpath,
	std::vector<std::vector<std::vector<GLfloat>>>& materialModel,
	std::vector<std::string>& texMaps,
	std::vector<std::vector<std::vector<GLfloat>>>& vetrices,
	std::vector<std::vector<std::vector<GLfloat>>>& UVs,
	std::vector<std::vector<std::vector<GLfloat>>>& normals,
	bool TEXTURED,
	std::vector<bool>& isTextured);

#endif