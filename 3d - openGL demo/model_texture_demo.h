#pragma once
#ifndef MODEL_TEXTURE_DEMO_H
#define MODEL_TEXTURE_DEMO_H

#include <windows.h>
#include <iostream>
#include <vector>
//#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
//#include "glext.h"
#include <GL/glu.h>
#include "GL/glut.h"
#include <stdlib.h>
#include <string>

#include "objectParser.h"
#include "image_path_loader.h"


void drawObjectDots(std::vector<std::vector<std::vector<GLfloat>>>& Fvertices,
	std::vector<std::vector<std::vector<GLfloat>>>& FUVs, std::vector<std::vector<std::vector<GLfloat>>>& Fnormals,
	bool FTEXTURED, unsigned int numObj, std::vector<std::vector<std::vector<GLfloat>>>& materialM);
void loadTexMaps(std::vector<std::string>& texBuffer, std::vector<std::string>& textureMapVec, int& TexWidth,
	int& TexHeight, int& TexNrChannels);
void textureDefPerObject(std::vector<std::string>& textureMapVec, std::vector<std::string>& texBuffer, unsigned int numTex,
	GLuint textureID, int& TexWidth, int& TexHeight, int& TexNrChannels);
class MODEL;
class PuffMODEL;
void init();
void display();
void sceneReshape(int w, int h);
void spinDirection();
void keyboard(unsigned char key, int x, int y);
void mainPipeLine(int argc, char** argv);

#endif