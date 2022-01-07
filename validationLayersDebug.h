#pragma once

#ifndef VALIDATION_LAYERS_DEBUG
#define VALIDATION_LAYERS_DEBUG

#define GLFW_INCLUDE_VULKAN
#pragma warning(push)
#pragma warning(disable : 26812)
#include <GLFW/glfw3.h>
#pragma warning(pop)

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);

#endif