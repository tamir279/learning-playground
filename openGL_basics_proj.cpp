// openGL_basics_proj.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#pragma comment(lib, "glew32.lib")
#include "glew.h"
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
#include "image_path_loader.h"
#include "obj_model_read_write.h"
#include "rigid_body_physics.h"
#include "model_draw.h"

static GLint fogMode;
GLenum shade_mode = GL_SMOOTH;

// initialize glew
void initGlew(void) {
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		std::cout << "oh no!!" << "\n";
		std::cout << glewGetErrorString(err);
		// GLEW failed!
		exit(1);
	}
}

std::vector<Rigid_body> bodyList;

// for a demo
void systemPhysicsLoop(int val) {
	// display bodies & scene - from model_draw. TODO in *MODEL_DRAW* - to create a function that draws multiple bodies
	//loop over all to check for collisions and update physics

	bool gravityApplied = true;
	std::vector<bool> applyLinearForce = { false, false, true, true, false };


	draw_multipleFlatRigidBodies_LEGACY_GL(bodyList, GL_TRIANGLES);

	int i = 0;
	for (auto b = bodyList.begin(); b != bodyList.end(); ++b) {
		Rigid_body body = *b;
		singleRigidBodyPhysics(&body, bodyList, applyLinearForce[i]);
		i++; b++;
	}

	glutTimerFunc(1, systemPhysicsLoop, 0);
}


void init(void) {
	generate_icosahedron_rigidBody_array(bodyList);
	init_scene_params_LEGACY_GL(fogMode, shade_mode);
}

void display(void) {

	display_scene_obj();
	draw_multipleFlatRigidBodies_LEGACY_GL(bodyList, GL_TRIANGLES);
}

void sceneReshape(int w, int h) {
	// viewport transformation
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	// projection
	glFrustum(-1.0, 1.0, -1.0, 1.0, 1.5, 50.0);
	glMatrixMode(GL_MODELVIEW);
	// modelview matrix
	gluLookAt(3.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
}

int main(int argc, char** argv)
{
    // unit tests of the physics code written.
	glutInit(&argc, argv);
	//glutSetOption(GLUT_MULTISAMPLE, 2);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(800, 800);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("3D textures and lighted model");
	initGlew();
	init();
	glutDisplayFunc(display);
	glutReshapeFunc(sceneReshape);
	systemPhysicsLoop(0);
	glutMainLoop();
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
