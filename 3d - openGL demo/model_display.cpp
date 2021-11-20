#pragma comment(lib, "glew32.lib")
#include "glew.h"
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
#include "model_texture_demo.h"

// defines
#pragma warning( push )
#pragma warning( disable : 6262)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma warning( pop )

// file paths -----------------------------------------------------------------------------------------------------------------
const char* OBJpath = "C:/Users/תמיר/source/repos/openGL textures and collision detection/textured objects/sofa/sofa4.obj";
const char* MTLpath = "C:/Users/תמיר/source/repos/openGL textures and collision detection/textured objects/sofa/sofa4.mtl";

// needed data ----------------------------------------------------------------------------------------------------------------
std::vector<std::vector<std::vector<GLfloat>>> materialModel;
std::vector<std::string> texMaps;
std::vector<bool> isTextured;
std::vector<std::vector<std::vector<GLfloat>>> vetrices;
std::vector<std::vector<std::vector<GLfloat>>> UVs;
std::vector<std::vector<std::vector<GLfloat>>> normals;
static bool TEXTURED = true;

// needed data - texture images -----------------------------------------------------------------------------------------------
std::vector<std::string> texMapsBuffer;
int width, height, nrChannels;

// global variables - for effects ---------------------------------------------------------------------------------------------
static GLint fogMode;

// global variables - for camera movement, colors -----------------------------------------------------------------------------
static GLfloat x_translation = 0.0;
static GLfloat y_translation = 0.0;
static GLfloat z_translation = 0.0;

static GLfloat RED[] = { 1, 0, 0 };
static GLfloat WHITE[] = { 1, 1, 1 };
int CheckerWidth = 16;
int CheckerDepth = 16;

// global variables - for light rotation --------------------------------------------------------------------------------------
static int spin = 0;
// ----------------------------------------------------------------------------------------------------------------------------

// initialize glew
void initGlew() {
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		std::cout << "oh no!!" << "\n";
		std::cout << glewGetErrorString(err);
		// GLEW failed!
		exit(1);
	}
}

// draws a single object
void drawObjectDots(std::vector<std::vector<std::vector<GLfloat>>>& Fvertices,
	std::vector<std::vector<std::vector<GLfloat>>>& FUVs, std::vector<std::vector<std::vector<GLfloat>>>& Fnormals,
	bool FTEXTURED, unsigned int numObj, std::vector<std::vector<std::vector<GLfloat>>>& materialM) {

	// define material colors
	std::vector<GLfloat> Ka_String = materialM[numObj][0];
	std::vector<GLfloat> Kd_String = materialM[numObj][1];
	std::vector<GLfloat> Ks_String = materialM[numObj][2];

	// convert to arrays
	GLfloat* Ka = &Ka_String[0];
	GLfloat* Kd = &Kd_String[0];
	GLfloat* Ks = &Ks_String[0];

	//glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, Ka);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, Kd);
	glMaterialfv(GL_FRONT, GL_SPECULAR, Ks);

	// draw each object
	unsigned int numVerticesAndNormals = Fvertices[numObj].size();
	glBegin(GL_TRIANGLES);
	for (unsigned int j = 0; j < numVerticesAndNormals; j++) {
		// define displayed vertices and normals + convert to array
		std::vector<GLfloat> vertexVector = Fvertices[numObj][j];
		std::vector<GLfloat> UVsVector;
		std::vector<GLfloat> normalVector = Fnormals[numObj][j];

		if (FTEXTURED)UVsVector = FUVs[numObj][j];

		GLfloat* vertex = &vertexVector[0];
		GLfloat* texture = &UVsVector[0];
		GLfloat* normal = &normalVector[0];

		// draw
		glNormal3fv(normal);
		if (FTEXTURED) glTexCoord2fv(texture);
		glVertex3fv(vertex);
	}
	glEnd();
}

void drawCheckerBoard(int Cwidth, int Cdepth) {
	glEnable(GL_COLOR_MATERIAL);
	glBegin(GL_QUADS);
	for (GLint x = 0; x < Cwidth; x++) {
		for (GLint z = 0; z < Cdepth; z++) {
			glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (x + z) % 2 == 0 ? RED : WHITE);
			if ((x + z) % 2 == 0) {
				glColor3f(1.0, 0.0, 0.0);
				glVertex3i(x, 0, z);
				glColor3f(1.0, 0.0, 0.0);
				glVertex3i(x + 1, 0, z);
				glColor3f(1.0, 0.0, 0.0);
				glVertex3i(x + 1, 0, z + 1);
				glColor3f(1.0, 0.0, 0.0);
				glVertex3i(x, 0, z + 1);
			}
			else {
				glColor3f(1.0, 1.0, 1.0);
				glVertex3i(x, 0, z);
				glColor3f(1.0, 1.0, 1.0);
				glVertex3i(x + 1, 0, z);
				glColor3f(1.0, 1.0, 1.0);
				glVertex3i(x + 1, 0, z + 1);
				glColor3f(1.0, 1.0, 1.0);
				glVertex3i(x, 0, z + 1);
			}
		}
	}
	glEnd();
	glDisable(GL_COLOR_MATERIAL);
}

void loadTexMaps(std::vector<std::string>& texBuffer, std::vector<std::string>& textureMapVec, int& TexWidth, 
	int& TexHeight, int& TexNrChannels) {

	const unsigned int numTextures = textureMapVec.size();

	// configure maximum image size for dynamic allocation and loading of a texture
	unsigned char* data;

	for (unsigned int i = 0; i < numTextures; i++) {
		// generate the file path to texture maps
		std::string textureMapPath;
		buildImagePath(textureMapVec[i], textureMapPath);
		// load the data from the texture map
		data = stbi_load(textureMapPath.c_str(), &TexWidth, &TexHeight, &TexNrChannels, 0);

		// convert: unsigned char* -> std::string
		std::string dataString(reinterpret_cast<char const*>(data));
		texBuffer.push_back(dataString);

		stbi_image_free(data);
	}
}

// configure and load the texture images ("maps") per object
void textureDefPerObject(std::vector<std::string>& textureMapVec, std::vector<std::string>& texBuffer, unsigned int numTex,
	GLuint textureID, int& TexWidth, int& TexHeight, int& TexNrChannels) {

	// configure maximum image size for dynamic allocation and loading of a texture
	std::string dataString = texBuffer[numTex];
	unsigned char* data = (unsigned char*)dataString.c_str();

	if (data)
	{
		// determine the type of image based on channel number
		GLenum format = 0;
		if (TexNrChannels == 1)
			format = GL_RED;
		else if (TexNrChannels == 3)
			format = GL_RGB;
		else if (TexNrChannels == 4)
			format = GL_RGBA;

		glBindTexture(GL_TEXTURE_2D, textureID);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, TexWidth, TexHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);

		// set the texture wrapping/filtering options (on the currently bound texture object)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 1);
	}
	else
	{
		std::cout << "Failed to load texture" << std::endl;
	}
}


class MODEL {
	bool TEXTURES_ENABLED = TEXTURED;
public:
	MODEL(bool TEXTURES_ENABLED) {};
	void createModel() {
		loadDataV2(OBJpath, MTLpath, materialModel, texMaps, vetrices, UVs, normals, TEXTURED, isTextured);
	}
	void drawModelAndEnableTextures() {
		const unsigned int numObjects = vetrices.size();
		const unsigned int numTextures = texMaps.size();

		// save to the texture ID array
		GLuint* texture = new GLuint[numTextures];
		if(TEXTURED){
			glGenTextures(numTextures, texture);
		}

		unsigned int texId = 0;
		for (unsigned int obj = 0; obj < numObjects; obj++) {
			if (isTextured[obj] && TEXTURED) {
				GLuint currentTexture = texture[texId];
				textureDefPerObject(texMaps, texMapsBuffer, obj, currentTexture, width, height, nrChannels);
				texId++;
			}
			drawObjectDots(vetrices, UVs, normals, TEXTURED, obj, materialModel);
		}
		delete[] texture;
	}
};

// global variable - model class
MODEL model1(TEXTURED);

void init() {

	GLfloat light_position[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat mat_ambient[] = { 0.9, 0.91, 0.91, 0.1 };

	// create model
	model1.createModel();
	loadTexMaps(texMapsBuffer, texMaps, width, height, nrChannels);

	// clear scene and define shading
	glClearColor(0.0, 0.0, 0.0, 0.0);
	//glClearColor(0.2, 0.1, 0.6, 0.0);
	glShadeModel(GL_SMOOTH);

	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	// enabeling lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_DITHER);

	// defining fog levels
	glEnable(GL_FOG);
	{
		GLfloat fogColor[4] = { 0.25, 0.25, 0.25, 1.0 };

		fogMode = GL_EXP;
		glFogi(GL_FOG_MODE, fogMode);
		glFogfv(GL_FOG_COLOR, fogColor);
		glFogf(GL_FOG_DENSITY, 0.02);
		glHint(GL_FOG_HINT, GL_DONT_CARE);
		glFogf(GL_FOG_START, 1.0);
		glFogf(GL_FOG_END, 5.0);
	}
}

void display() {
	// clear colors, enable textures
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	// light position
	GLfloat light_position[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat mat_ambient[] = { 0.9, 0.91, 0.91, 0.1 };
	// viewing transformation
	// looking at the origin from point (1,2,5) as the "camera" - should be replaced by gluPerspective
	glPushMatrix();
	GLfloat x_pos = 3.0 + x_translation;
	GLfloat y_pos = 6.0 + y_translation;
	GLfloat z_pos = 7.0 + z_translation;
	//       (eyex eyey eyez)    (cenx ceny cenz) (upx  upy  upz)
	gluLookAt(x_pos, y_pos, z_pos, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

	// rotating the light the light
	glPushMatrix();
	glRotatef((GLfloat)spin, 0.0, 1.0, 0.0);
	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glPopMatrix();

	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);
	glPushMatrix();
	glTranslatef(3.0, 0.0, 0.0);
	glRotatef(90.0, 1.0, 0.0, 0.0);
	glRotatef(180.0, 0.0, 1.0, 0.0);
	glScalef(0.03, 0.03, 0.03);
	model1.drawModelAndEnableTextures();
	glPopMatrix();
	glDisable(GL_TEXTURE_2D);

	// draw checkerboard
	glPushMatrix();
	glTranslatef(-(GLfloat)CheckerWidth / 2, (GLfloat)CheckerDepth / 2, 0.0);
	glRotatef(90, 1.0, 0.0, 0.0);
	drawCheckerBoard(CheckerWidth, CheckerDepth);
	glPopMatrix();

	//glPushMatrix();
	//glutSolidSphere(2, 16, 10);
	//glPopMatrix();

	glPopMatrix();
	glFlush();
	glutSwapBuffers();
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

// spin in idle fassion
void spinDirection() {
	spin = (spin + 1) % 360;
	glutPostRedisplay();
}

// respond to keboard commands
void keyboard(unsigned char key, int x, int y) {
	switch (key) {
	case 'a':
		x_translation += 0.15;
		glutPostRedisplay();
		break;
	case 'd':
		x_translation -= 0.15;
		glutPostRedisplay();
		break;
	case 's':
		z_translation -= 0.15;
		glutPostRedisplay();
		break;
	case 'w':
		z_translation += 0.15;
		glutPostRedisplay();
	case 'e':
		y_translation += 0.15;
		glutPostRedisplay();
		break;
	case 'q':
		y_translation -= 0.15;
		glutPostRedisplay();
		break;
	default:
		break;
	}
}

void mainPipeLine(int argc, char** argv) {
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
	glutIdleFunc(spinDirection);
	glutKeyboardFunc(keyboard);
	glutMainLoop();
}