#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#pragma warning(push)
#pragma warning(disable : 26812)
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#pragma warning(pop)

#include "glfw_window.h"
#include <stdexcept>

namespace MLE {

	MyLittleWindow::MyLittleWindow(int w, int h, std::string name) : width{ w }, height{ h }, windowName{ name } {
		initWindow();
	}

	MyLittleWindow::~MyLittleWindow() {
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	void MyLittleWindow::initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(width, height, windowName.c_str(), nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	void MyLittleWindow::createWindowSurface(VkInstance instance, VkSurfaceKHR* surface) {
		if (glfwCreateWindowSurface(instance, window, nullptr, surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	void MyLittleWindow::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto Ewindow = reinterpret_cast<MyLittleWindow*>(glfwGetWindowUserPointer(window));
		Ewindow->framebufferResized = true;
		Ewindow->width = width;
		Ewindow->height = height;
	}
}