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

#include <unordered_map>

#include "renderer.h"

namespace MLE::RENDERER {

	// window opening and closing

	void WindoW::initWindow() {
		if (!glfwInit()) { std::cout << "GLFW has failed to initialize!"; }
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
	Mesh::~Mesh() {
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
		glDeleteBuffers(1, &EBO);
	}

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

	auto setTextureLayerMap() {
		std::unordered_map<std::string, uint32_t> textureLayers({
			{ "texture_diffuse" , 1 },
			{ "texture_specular" , 1 },
			{ "texture_normal" , 1 },
			{ "texture_normal" , 1 },
			{ "texture_height" , 1 } });

		return textureLayers;
	}

	void Mesh::draw(shader &shader) {
		// map of texture layers - suited for PBR
		auto textureMap = setTextureLayerMap();
		// bind textures
		std::string number;
		for (uint32_t i = 0; i < textures.size(); i++) {
			std::string name = textures[i].type;
			// activate texture
			glActiveTexture(GL_TEXTURE0 + i);
			// transfer number to string
			number = std::to_string((textureMap)[name]++);
			// now set the sampler to the correct texture unit
			glUniform1i(glGetUniformLocation(shader.ID, (name + number).c_str()), i);
			// and finally bind the texture
			glBindTexture(GL_TEXTURE_2D, textures[i].id);
		}

		// draw mesh
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, static_cast<uint32_t>(indices.size()), GL_UNSIGNED_INT, 0);
	}

}