#include "glew.h"
#include <windows.h>
#include <iostream>
#include <vector>
#include <GL/gl.h>
#include <GL/glu.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <string>
#include "renderer.h"

namespace MLE::RENDERER {

	GLenum getFormat(int channels) {
		return !(channels - 1) ? GL_RED : !(channels - 3) ? GL_RGB : !(channels - 4) ? GL_RGBA : GL_GREEN;
	}

	unsigned int model::TextureFromFile(const char* path, const std::string& directory, bool gamma) {
		std::string filename = std::string(path);
		filename = directory + '/' + filename;
		// generate texture ID to be bound to 
		unsigned int textureID;
		glGenTextures(1, &textureID);
		// image dimensions : [w, h] x nrComponents "=" [w,h,channels]
		int width, height, channels;
		unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
		if (data) {
			auto format = getFormat(channels);
			glBindTexture(GL_TEXTURE_2D, textureID);
			glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
			glGenerateMipmap(GL_TEXTURE_2D);
			
			// texture parameters - define textures to repeat to the edges of a model
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			// define smoothing filters and mipmap (LOD)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			stbi_image_free(data);
		}
		else {
			std::cout << "Texture failed to load at path: " << path << std::endl;
			stbi_image_free(data);
		}
		return textureID;
	}
}