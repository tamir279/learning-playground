#pragma once
#ifndef IMAGE_PATH_LOADER_H
#define IMAGE_PATH_LOADER_H

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <string>

void buildImagePath(std::string& textureImageName, std::string& textureImagePath) {
	std::string basePath = "C:/Users/תמיר/source/repos/openGL textures and collision detection/textured objects/";
	if (!strcmp(textureImageName.c_str(), "mufiber03.png")) {
		textureImagePath = basePath + "sofa/" + textureImageName;
	}
	else if (!strcmp(textureImageName.c_str(), "singleChair1.jpg") | !strcmp(textureImageName.c_str(), "singleChair2.jpg")) {
		textureImagePath = basePath + "single chair/" + textureImageName;
	}
	else if (!strcmp(textureImageName.c_str(), "clockface2.jpg") | !strcmp(textureImageName.c_str(), "wood_dark.jpg")) {
		textureImagePath = basePath + "longcase clock/" + textureImageName;
	}
	else if (!strcmp(textureImageName.c_str(), "carpet.jpg")) {
		textureImagePath = basePath + "carpet/" + textureImageName;
	}
	else if (!strcmp(textureImageName.c_str(), "puff_texture.jpg")) {
		textureImagePath = basePath + "puff/" + textureImageName;
	}
}

#endif