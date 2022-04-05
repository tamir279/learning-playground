#pragma once
#include "glew.h"
#include <windows.h>
#include <iostream>
#include <vector>
#include <GL/gl.h>
#include <GL/glu.h>

#include <string>
#include <fstream>
#include <sstream>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>

#pragma warning(push)
#pragma warning(disable : 26812)
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#pragma warning(pop)
/*
******* OPENGL RENDERER *******
*/

namespace MLE {
	namespace RENDERER {

		// important structs
		struct Vertex {
			glm::vec3 position;
			glm::vec3 normal;
			glm::vec3 TexCoord;
		};

		struct Texture {
			unsigned int id;
			std::string type;
			std::string path;
		};

		// rendering classes
		class WindoW {
		public:
			WindoW(int w, int h, std::string name);
			~WindoW();

			// delete constructor
			WindoW(const WindoW&) = delete;
			WindoW& operator=(const WindoW&) = delete;

			// window opening and closing
			bool shouldClose() { return glfwWindowShouldClose(window); }
			bool wasWindowResized() { return framebufferResized; }
			void resetWindowResizedFlag() { framebufferResized = false; }
			GLFWwindow* getGLFWwindow() const { return window; }

		private:
			// callback for framebuffer - to initialize the window
			static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
			void initWindow();

			// window parameters
			int width;
			int height;
			bool framebufferResized = false;

			std::string windowName;
			GLFWwindow* window;
		};

		class camera {

		};

		class shader {
		public:
			unsigned int ID;
			// shader constructor - for now using only vertex and fragment shaders
			shader(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr);
			// use the compiled shader program recognized by ID
			void use();
			// set uniform value of certain type
			template<typename... Args>
			void setUniformValue(const std::string& name, Args&... values)const;
		private:
			void checkCompileErrors(GLuint shader, std::string type);
		};

		class Mesh {
		public:
			// data - vertices, indices and textures - all unpacked in
			// the model class
			std::vector<Vertex> vertices;
			std::vector<unsigned int> indices;
			std::vector<Texture> textures;
			unsigned int VAO;

			Mesh(
				std::vector<Vertex> v,
				std::vector<unsigned int> i,
				std::vector<Texture> t) {

				this->vertices = v;
				this->indices = i;
				this->textures = t;
				// set mesh to rendering
				setMesh();
			}

			void draw(shader &shader);
		private:
			unsigned int EBO, VBO;
			void setMesh();
		};

		class model {
		public:
			std::vector<Texture> loadedTextures;
			std::vector<Mesh> meshes;
			std::string directory;
			bool gammaCorrection;

			// constructor, expects a filepath to a 3D model.
			model(std::string const& path, bool gamma = false) : gammaCorrection{ gamma } {
				loadModel(path);
			}

			// draws the model, and thus all its meshes
			void draw(shader& shader);

		private:
			void loadModel(std::string const& path);
			//void processNode(aiNode* node, const aiScene* scene);
			//Mesh processMesh(aiMesh* mesh, const aiScene* scene);
			//std::vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, string typeName);
			unsigned int TextureFromFile(const char* path, const std::string& directory, bool gamma);
		};
	}
}