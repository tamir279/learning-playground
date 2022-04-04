#include "glew.h"
#include <windows.h>
#include <iostream>
#include <vector>
#include <GL/gl.h>
#include <GL/glu.h>

#pragma warning(push)
#pragma warning(disable : 26812)
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#pragma warning(pop)

#include "renderer.h"

namespace MLE::RENDERER {

	// window opening and closing

	void WindoW::initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		window = glfwCreateWindow(width, height, windowName.c_str(), nullptr, nullptr);
		if (window == NULL)
		{
			std::cout << "Failed to create GLFW window" << std::endl;
			glfwTerminate();
		}
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	WindoW::WindoW(int w, int h, std::string name) : width{ w }, height{ h }, windowName{ name }{
		initWindow();
	}

	WindoW::~WindoW() {
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	void WindoW::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<WindoW*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
		app->width = width;
		app->height = height;
	}

	// mesh setup and drawing

	void Mesh::setMesh() {
		// create buffers/arrays
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);
		// bind the VAO generatoed
		glBindVertexArray(VAO);
		// bind the VBO with the data itself
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_DYNAMIC_DRAW);
		// bind the EBO with index data itself
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_DYNAMIC_DRAW);
		// bind VBO with vertex shader	
		// vertex Positions
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
		// vertex normals
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
		// vertex texture coords
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoord));
	}

	void Mesh::draw(shader &shader) {
		// for multiple texture layers - suited for PBR
		unsigned int diffuseNr = 1;
		unsigned int specularNr = 1;
		unsigned int normalNr = 1;
		unsigned int heightNr = 1;

		// bind textures
		for (unsigned int i = 0; i < textures.size(); i++) {
			std::string number;
			std::string name = textures[i].type;
			// activate texture
			glActiveTexture(GL_TEXTURE0 + i);
			// transfer number to string
			if (name == "texture_diffuse") number = std::to_string(diffuseNr++);
			else if (name == "texture_specular") number = std::to_string(specularNr++);
			else if (name == "texture_normal") number = std::to_string(normalNr++);
			else if (name == "texture_height") number = std::to_string(heightNr++);

			// now set the sampler to the correct texture unit
			glUniform1i(glGetUniformLocation(shader.ID, (name + number).c_str()), i);
			// and finally bind the texture
			glBindTexture(GL_TEXTURE_2D, textures[i].id);
		}
		// draw mesh
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
	}
}