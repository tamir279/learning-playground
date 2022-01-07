#define GLFW_INCLUDE_VULKAN
#pragma warning(push)
#pragma warning(disable : 26812)
#include <GLFW/glfw3.h>
#pragma warning(pop)
#include <vector>
#include <string>

#include "vulkanMainPipeLine.h"
#include "validationLayersDebug.h"

int main() {

	app application;
    try {
        application.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;

	return 0;
}