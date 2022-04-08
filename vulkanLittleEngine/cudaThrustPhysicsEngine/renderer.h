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
#include <tuple>

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
	// TODO : add resource namespace for managing all non-model data
	// quaternion class - general
	class quaternion {
	public:

		// rotation + vector
		float s;
		glm::vec3 vector;

		// build quaternion
		quaternion(float rotation, glm::vec3 D3Dvector) : s{ rotation }, vector{ D3Dvector } {}

		// overload - optional for defining a quaternion without inputs
		quaternion() : s{ 0 }, vector{ glm::vec3(0) } {}

		// destructor
		~quaternion() {}

		// basic operations
		void operator+=(const quaternion& q);
		quaternion operator+(const quaternion& q);
		void operator-=(const quaternion& q);
		quaternion operator-(const quaternion& q);
		void operator*=(const quaternion& q);
		quaternion operator*(const quaternion& q);
		void operator*=(const float scale);
		quaternion operator*(const float scale);

		// specified functions to use for 3d vector rotations

		/*
		L2 norm ||q||_2
		*/
		float Norm();
		void Normalize();

		/*
		q* = [s, -vector]
		*/
		void conjugate();
		quaternion Conjugate();

		/*
		q^-1 = q* /||q||^2
		*/
		quaternion inverse();

		// convert to rotation in 3d

		/*
		v' = v/||v||
		q_unit = [cos(o/2), sin(o/2)v']
		*/
		void ConvertToRotationQuaternionRepresentation();

	private:
		float fastSquareRoot(float num);
		float DegreesToRadians(float angle);
	};

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
		public:
			// building blocks for a lookAt transformation
			glm::vec3 eye;
			glm::vec3 center;
			glm::vec3 up;

			// camera response speed
			float delta;
			float mouseSensitivity;
			float FieldOfView;

			camera(glm::vec3 initEye, glm::vec3 initCenter, glm::vec3 initUp, float _speed, float ms) {
				this->eye = initEye;
				this->center = initCenter;
				this->up = initUp;
				this->delta = _speed;
				this->mouseSensitivity = ms;
				this->FieldOfView = 1.0f;
			}

			// overload for default camera positioning
			camera() : eye{ glm::vec3(1, 0, 0) }, center{ glm::vec3(0) }, up{ glm::vec3(0, 0, 1) }, delta{ 1 } {
				this->mouseSensitivity = 0.1;
				this->FieldOfView = 1.0f;
			}

			// get transformed view matrix for shader use
			auto getViewMatrix();
			// camera keyboard inputs
			void processKeyboardInput(GLFWwindow* window);
			// camera mouse inputs
			void processMouseMovement(float Xoffset, float Yoffset);
			void processMouseScrolling(float Yoffset);
			void mousePositionCallBack(GLFWwindow* window, double x, double y);
			void mouseScrollCallBack(GLFWwindow* window, double Xoffset, double Yoffset);
			// set from global values
			void setXYvalues(float X, float Y) { lastX = X; lastY = Y; }
			auto getXYvalues() { return std::make_tuple(lastX, lastY); }

			// set and get mouse movement flag
			void setMovementFlag(bool _Move) { firstMove = _Move; }
			bool getMovementFlag() { return firstMove; }

		private:
			static float lastX, lastY;
			static bool firstMove;
			// process inputs
			void processKeyboardInput_W_();
			void processKeyboardInput_A_();
			void processKeyboardInput_S_();
			void processKeyboardInput_D_();
			// rotation
			void RotateVector(glm::vec3& v, quaternion rotation);

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
			void setUniformValue(const std::string& name, Args&... values);
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

			~Mesh();

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