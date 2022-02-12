#pragma once

#ifndef GLFW_WINDOW_H
#define GLFW_WINDOW_H

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#pragma warning(push)
#pragma warning(disable : 26812)
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#pragma warning(pop)

#include <string>

namespace MLE {

	class MyLittleWindow {
	public:
		MyLittleWindow(int w, int h, std::string name);
		~MyLittleWindow();

		// delete constructor
		MyLittleWindow(const MyLittleWindow&) = delete;
		MyLittleWindow& operator=(const MyLittleWindow&) = delete;

		// window opening and closing
		bool shouldClose(){ return glfwWindowShouldClose(window); }
		VkExtent2D getExtent() { return { static_cast<uint32_t>(width), static_cast<uint32_t>(height) }; }
		bool wasWindowResized() { return framebufferResized; }
		void resetWindowResizedFlag() { framebufferResized = false; }
		GLFWwindow* getGLFWwindow() const { return window; }

		void createWindowSurface(VkInstance instance, VkSurfaceKHR* surface);

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
}


#endif