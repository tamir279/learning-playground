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
#include "image_path_loader.h"
#include "obj_model_read_write.h"

// defines
#pragma warning( push )
#pragma warning( disable : 6262)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma warning( pop )

/* functions for textured .obj model drawing*/

void defineColorsAndMaterials_LEGACY_GL(unsigned int numObj, std::vector<std::vector<std::vector<GLfloat>>>& material_Matrix) {

	// define material colors
	std::vector<GLfloat> Ka_Vec = material_Matrix[numObj][0];
	std::vector<GLfloat> Kd_Vec = material_Matrix[numObj][1];
	std::vector<GLfloat> Ks_Vec = material_Matrix[numObj][2];

	// convert to arrays
	GLfloat* Ka = !Ka_Vec.empty() ? &Ka_Vec[0] : NULL;
	GLfloat* Kd = !Kd_Vec.empty() ? &Kd_Vec[0] : NULL;
	GLfloat* Ks = !Ks_Vec.empty() ? &Ks_Vec[0] : NULL;

	if (!Ka_Vec.empty()) { glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, Ka); }
	if (!Kd_Vec.empty()) { glMaterialfv(GL_FRONT, GL_DIFFUSE, Kd); }
	if (!Ks_Vec.empty()) { glMaterialfv(GL_FRONT, GL_SPECULAR, Ks); }
}

GLenum encode_GL_polygon_type(int polygonSize) {
	GLenum render_type = 0;
	if (polygonSize < 3) { throw std::length_error("error in reading vertex data : num_vertices < 3"); }
	else if (polygonSize == 3) { render_type = GL_TRIANGLES; }
	else if (polygonSize == 4) { render_type = GL_QUADS; }
	else if (polygonSize > 4 && polygonSize % 3 == 0) { render_type = GL_TRIANGLE_STRIP; }
	else if (polygonSize > 4 && polygonSize % 4 == 0) { render_type = GL_QUAD_STRIP; }
	else { // has to split between triangles and quads - TODO
	}

	return render_type;
}

void draw_OBJ_Object_Polygons_LEGACY_GL(std::vector<std::vector<GLfloat>>& vertices,
	std::vector<std::vector<GLfloat>>& UVs,
	std::vector<std::vector<GLfloat>>& normals,
	std::vector<int>& polygon_size,
	bool isTextured,
	bool TEXTURED) {

	// check for size match
	if (vertices.size() != normals.size()) {
		throw std::length_error("error in reading vertex and normal data / unexpected .OBJ format");
	}

	int offset = 0;
	for (unsigned int i = 0; i < polygon_size.size(); i++) {
		GLenum render_type = encode_GL_polygon_type(polygon_size[i]);

		glBegin(render_type);
		for (int j = 0; j < polygon_size[i]; j++) {
			std::vector<GLfloat> vertexVector = vertices[offset + j];
			std::vector<GLfloat> UVsVector;
			std::vector<GLfloat> normalVector = normals[offset + j];

			GLfloat* vertex = &vertexVector[0];
			GLfloat* UV = NULL;
			GLfloat* normal = &normalVector[0];

			if (isTextured && TEXTURED) {
				UVsVector = UVs[i];
				UV = &UVsVector[0];
			}

			glNormal3fv(normal);
			if (isTextured && TEXTURED)glTexCoord2fv(UV);
			glVertex3fv(vertex);
		}
		glEnd();
		offset += polygon_size[i];
	}
}

void loadTexMaps(std::vector<std::string>& texBuffer, std::vector<std::string>& textureMapVec, std::vector<int>& TexWidth,
	std::vector<int>& TexHeight, std::vector<int>& TexNrChannels) {

	const unsigned int numTextures = textureMapVec.size();

	// configure maximum image size for dynamic allocation and loading of a texture
	unsigned char* data;

	for (unsigned int i = 0; i < numTextures; i++) {
		int width, height, NrChannels;
		// generate the file path to texture maps
		std::string textureMapPath;
		buildImagePath(textureMapVec[i], textureMapPath);
		// load the data from the texture map
		data = stbi_load(textureMapPath.c_str(), &width, &height, &NrChannels, 0);

		// convert: unsigned char* -> std::string
		std::string dataString(reinterpret_cast<char const*>(data));
		texBuffer.push_back(dataString);
		TexWidth.push_back(width);
		TexHeight.push_back(height);
		TexNrChannels.push_back(NrChannels);

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
		if (TexNrChannels == 1)format = GL_RED;
		else if (TexNrChannels == 3)format = GL_RGB;
		else if (TexNrChannels == 4)format = GL_RGBA;

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

void init_model(const char* OBJpath,
	const char* MTLpath,
	std::vector<std::vector<std::vector<GLfloat>>>& materialModel,
	std::vector<std::string>& texMaps,
	std::vector<std::vector<std::vector<GLfloat>>>& vetrices,
	std::vector<std::vector<std::vector<GLfloat>>>& UVs,
	std::vector<std::vector<std::vector<GLfloat>>>& normals,
	std::vector<int>& polygon_size,
	bool TEXTURED,
	std::vector<bool>& isTextured,
	std::vector<std::string>& texBuffer,
	std::vector<int>& TexWidth,
	std::vector<int>& TexHeight,
	std::vector<int>& TexNrChannels) {

	read_OBJ_models_LEGACY_GL(OBJpath, MTLpath, materialModel, texMaps, vetrices, UVs, normals, polygon_size, TEXTURED, isTextured);
	loadTexMaps(texBuffer, texMaps, TexWidth, TexHeight, TexNrChannels);
}

void draw_textured_elements_LEGACY_GL(std::vector<std::vector<std::vector<GLfloat>>>& material_Matrix,
	std::vector<std::string>& texMaps,
	std::vector<std::string>& texBuffer,
	std::vector<int>& TexWidth,
	std::vector<int>& TexHeight,
	std::vector<int>& TexNrChannels,
	std::vector<std::vector<std::vector<GLfloat>>>& vetrices,
	std::vector<std::vector<std::vector<GLfloat>>>& UVs,
	std::vector<std::vector<std::vector<GLfloat>>>& normals,
	std::vector<int>& polygon_size,
	bool TEXTURED,
	std::vector<bool>& isTextured) {

	const unsigned int numObjects = vetrices.size();
	const unsigned int numTextures = texMaps.size();

	// save to the texture ID array
	GLuint* texture = new GLuint[numTextures];
	if (TEXTURED) {
		glGenTextures(numTextures, texture);
	}

	unsigned int texId = 0;
	for (unsigned int obj = 0; obj < numObjects; obj++) {
		defineColorsAndMaterials_LEGACY_GL(obj, material_Matrix);
		if (isTextured[obj] && TEXTURED) {
			GLuint currentTexture = texture[texId];
			textureDefPerObject(texMaps, texBuffer, obj, currentTexture, TexWidth[obj], TexHeight[obj], TexNrChannels[obj]);
			texId++;
		}
		draw_OBJ_Object_Polygons_LEGACY_GL(vetrices[obj], UVs[obj], normals[obj], polygon_size, isTextured[obj], TEXTURED);
	}
	delete[] texture;
}

void init_scene_params_LEGACY_GL(GLint& fogMode, GLenum shade_mode) {
	GLfloat light_position[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat mat_ambient[] = { 0.9, 0.91, 0.91, 0.1 };

	// clear scene and define shading
	glClearColor(0.0, 0.0, 0.0, 0.0);
	//glClearColor(0.2, 0.1, 0.6, 0.0);
	glShadeModel(shade_mode);

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

void display_scene_light_LEGACY_GL(
	std::vector<GLfloat>& init_cam_pos,
	std::vector<GLfloat>& cam_trans,
	std::vector<GLfloat>& cam_rot,
	std::vector<GLfloat>& init_light_pos,
	std::vector<GLfloat>& light_trans,
	std::vector<GLfloat>& light_rot) {

	// clear colors, enable textures
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	GLfloat mat_ambient[] = { 0.9, 0.91, 0.91, 0.1 };

	glPushMatrix();
	// starting to operate on vectors
	GLfloat x = init_cam_pos[0];
	GLfloat y = init_cam_pos[1];
	GLfloat z = init_cam_pos[2];
	// (eyex eyey eyez)(cenx ceny cenz)(upx  upy  upz)
	gluLookAt(x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

	glRotatef(cam_rot[0], 1.0, 0.0, 0.0);
	glRotatef(cam_rot[1], 0.0, 1.0, 0.0);
	glRotatef(cam_rot[2], 0.0, 0.0, 1.0);
	glTranslatef(cam_trans[0], cam_trans[1], cam_trans[2]);

	// rotating the light the light
	const GLfloat* init_pos = &init_light_pos[0];

	glPushMatrix();
	glRotatef(light_rot[0], 1.0, 0.0, 0.0);
	glRotatef(light_rot[1], 0.0, 1.0, 0.0);
	glRotatef(light_rot[2], 0.0, 0.0, 1.0);
	glTranslatef(light_trans[0], light_trans[1], light_trans[2]);
	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glLightfv(GL_LIGHT0, GL_POSITION, init_pos);
	glPopMatrix();
}

/* functions for more specific purposes - drawing only flat polygons of specific polygon shape/
specific wireframes for collision detection visualization/ flat surface */

void draw_flat_obj_LEGACY_GL(std::vector<std::vector<GLfloat>>& indexed_vertices, GLenum render_type) {
	glBegin(render_type);
	for (int i = 0; i < (int)indexed_vertices.size(); i++) {
		std::vector<GLfloat> vertex_vec = indexed_vertices[i];
		GLfloat* vertex = &vertex_vec[0];
		glVertex3fv(vertex);
	}
	glEnd();
}

void draw_wireframe_LEGACY_GL(std::vector<std::vector<GLfloat>>& indexed_vertices) {
	draw_flat_obj_LEGACY_GL(indexed_vertices, GL_LINES);
}

// draw flat checkerBoard surface
void drawCheckerBoard_LEGACY_GL(int Cwidth, int Cdepth, GLfloat COLOR1[], GLfloat COLOR2[]) {
	glEnable(GL_COLOR_MATERIAL);
	glBegin(GL_QUADS);
	for (GLint x = 0; x < Cwidth; x++) {
		for (GLint z = 0; z < Cdepth; z++) {
			glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (x + z) % 2 == 0 ? COLOR1 : COLOR2);
			if ((x + z) % 2 == 0) {
				glColor3fv(COLOR1);
				glVertex3i(x, 0, z);
				glColor3fv(COLOR1);
				glVertex3i(x + 1, 0, z);
				glColor3fv(COLOR1);
				glVertex3i(x + 1, 0, z + 1);
				glColor3fv(COLOR1);
				glVertex3i(x, 0, z + 1);
			}
			else {
				glColor3fv(COLOR2);
				glVertex3i(x, 0, z);
				glColor3fv(COLOR2);
				glVertex3i(x + 1, 0, z);
				glColor3fv(COLOR2);
				glVertex3i(x + 1, 0, z + 1);
				glColor3fv(COLOR2);
				glVertex3i(x, 0, z + 1);
			}
		}
	}
	glEnd();
	glDisable(GL_COLOR_MATERIAL);
}
