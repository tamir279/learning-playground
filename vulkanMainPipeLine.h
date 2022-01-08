#pragma once
#ifndef VULKAN_MAIN_PIPELINE_H
#define VULKAN_MAIN_PIPELINE_H

#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <optional>
#include "dataLoadingAndGraphics.h"

struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() {
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

class app {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanUp();
	}

	void set_framebufferResized(bool _framebufferResized) { framebufferResized = _framebufferResized; }
private:
	void initWindow();
	void initVulkan();
	void mainLoop();
	void cleanUp();

	GLFWwindow* window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;
	VkSurfaceKHR surface;
	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	VkRenderPass renderPass;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;
	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	std::vector<VkFence> imagesInFlight;
	size_t currentFrame = 0;
	bool framebufferResized = false;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;
	VkDescriptorSetLayout descriptorSetLayout;
	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;
	VkImage textureImage;
	VkDeviceMemory textureImageMemory;
	VkImageView textureImageView;
	VkSampler textureSampler;
	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indexArr;
};

void framebufferResizeCallback(GLFWwindow* window, int width, int height);
void createInstance(VkInstance& instance);
void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
void setupDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT& debugMessenger);
std::vector<const char*> getRequiredExtensions();
bool checkValidationLayerSupport();
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
void pickPhysicalDevice(VkInstance instance, VkPhysicalDevice& physicalDevice, VkSurfaceKHR surface);
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface);
bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface);
void createLogicalDevice(VkSurfaceKHR surface, VkPhysicalDevice physicalDevice, VkDevice& device, VkQueue& graphicsQueue, VkQueue& presentQueue);
void createSurface(VkInstance instance, GLFWwindow* window, VkSurfaceKHR& surface);
bool checkDeviceExtensionSupport(VkPhysicalDevice device);
SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);
VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window);
void createSwapChain(VkDevice device, VkPhysicalDevice physicalDevice, GLFWwindow* window, VkSurfaceKHR surface, VkSwapchainKHR& swapChain, std::vector<VkImage>& swapChainImages, VkFormat& swapChainImageFormat, VkExtent2D& swapChainExtent);
void recreateSwapChain(GLFWwindow*& window, VkDevice device, std::vector<VkFence>& imagesInFlight, std::vector<VkImage>& swapChainImages,
	VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, VkSwapchainKHR& swapChain, VkFormat& swapChainImageFormat, VkExtent2D& swapChainExtent,
	std::vector<VkImageView>& swapChainImageViews, VkRenderPass& renderPass, std::vector<VkFramebuffer>& swapChainFramebuffers, VkCommandPool& commandPool,
	std::vector<VkCommandBuffer>& commandBuffers, VkPipeline& graphicsPipeline, VkPipelineLayout& pipelineLayout, VkBuffer& vertexBuffer, VkBuffer& indexBuffer,
	VkDescriptorSetLayout& descriptorSetLayout, std::vector<VkBuffer>& uniformBuffers, std::vector<VkDeviceMemory>& uniformBuffersMemory, VkDescriptorPool& descriptorPool, std::vector<VkDescriptorSet>& descriptorSets,
	VkImageView textureImageView, VkSampler textureSampler, VkImageView& depthImageView, VkImage& depthImage, VkDeviceMemory& depthImageMemory, VkQueue graphicsQueue, std::vector<uint32_t> indexArr);
void createImageViews(VkDevice device, std::vector<VkImage> swapChainImages, VkFormat swapChainImageFormat, std::vector<VkImageView>& swapChainImageViews);
void cleanupSwapChain(VkDevice device, std::vector<VkFramebuffer> swapChainFramebuffers, VkCommandPool commandPool, std::vector<VkCommandBuffer> commandBuffers, VkPipeline graphicsPipeline,VkPipelineLayout pipelineLayout, VkRenderPass renderPass, std::vector<VkImageView> swapChainImageViews, VkSwapchainKHR swapChain,
	std::vector<VkImage> swapChainImages, std::vector<VkBuffer> uniformBuffers, std::vector<VkDeviceMemory> uniformBuffersMemory, VkDescriptorPool descriptorPool, VkImage depthImage, VkDeviceMemory depthImageMemory, VkImageView depthImageView);
static std::vector<char> readFile(const std::string& filename);
VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);
void createGraphicsPipeLine(VkDevice device, VkExtent2D swapChainExtent, VkPipelineLayout& pipelineLayout, VkRenderPass renderPass, VkPipeline& graphicsPipeline, VkDescriptorSetLayout descriptorSetLayout);
void createRenderPass(VkDevice device, VkPhysicalDevice physicalDevice, VkFormat swapChainImageFormat, VkRenderPass& renderPass);
void createFrameBuffers(std::vector<VkFramebuffer>& swapChainFramebuffers, std::vector<VkImageView> swapChainImageViews, VkRenderPass renderPass, VkExtent2D swapChainExtent, VkDevice device, VkImageView depthImageView);
void createCommandPool(VkDevice device, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, VkCommandPool& commandPool);
void createCommandBuffers(std::vector<VkCommandBuffer>& commandBuffers, std::vector<VkFramebuffer> swapChainFramebuffers, VkCommandPool commandPool, VkDevice device, VkRenderPass renderPass, VkExtent2D swapChainExtent, VkPipeline graphicsPipeline, VkBuffer vertexBuffer, VkBuffer indexBuffer,
	VkPipelineLayout pipelineLayout, std::vector<VkDescriptorSet> descriptorSets , std::vector<uint32_t> indexArr);
void createSemaphores(VkDevice device, std::vector<VkImage> swapChainImages, std::vector<VkSemaphore>& imageAvailableSemaphores, std::vector<VkSemaphore>& renderFinishedSemaphores, std::vector<VkFence>& inFlightFences, std::vector<VkFence>& imagesInFlight);
void drawFrame(GLFWwindow* window, VkDevice device, VkQueue graphicsQueue, VkQueue presentQueue, VkSwapchainKHR swapChain, std::vector<VkSemaphore> imageAvailableSemaphores, std::vector<VkSemaphore> renderFinishedSemaphores, std::vector<VkCommandBuffer>& commandBuffers, size_t& currentFrame, std::vector<VkFence> inFlightFences,
	std::vector<VkFence> imagesInFlight, std::vector<VkImage>& swapChainImages, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, VkFormat& swapChainImageFormat, VkExtent2D& swapChainExtent, std::vector<VkImageView>& swapChainImageViews, VkRenderPass& renderPass, std::vector<VkFramebuffer>& swapChainFramebuffers,
	VkCommandPool& commandPool, VkPipeline& graphicsPipeline, VkPipelineLayout& pipelineLayout, VkBuffer& vertexBuffer, VkBuffer& indexBuffer, bool& framebufferResized, VkDescriptorSetLayout& descriptorSetLayout, std::vector<VkBuffer>& uniformBuffers, std::vector<VkDeviceMemory>& uniformBuffersMemory,
	VkDescriptorPool& descriptorPool, std::vector<VkDescriptorSet>& descriptorSets, VkImageView textureImageView, VkSampler textureSampler, VkImageView depthImageView, VkImage& depthImage, VkDeviceMemory& depthImageMemory, std::vector<uint32_t> indexArr);
void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
void createVertexBuffer(VkDevice device, VkBuffer& vertexBuffer, VkDeviceMemory& vertexBufferMemory, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, std::vector<Vertex> vertices);
uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkPhysicalDevice physicalDevice);
void copyBuffer(VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue, VkBuffer srcBuffer, VkBuffer& dstBuffer, VkDeviceSize size);
void createIndexBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkBuffer& indexBuffer, VkDeviceMemory& indexBufferMemory, VkCommandPool commandPool, VkQueue graphicsQueue, std::vector<uint32_t> indexArr);
void createDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout& descriptorSetLayout);
void createUniformBuffers(VkDevice device, VkPhysicalDevice physicalDevice, std::vector<VkBuffer>& uniformBuffers, std::vector<VkDeviceMemory>& uniformBuffersMemory, std::vector<VkImage> swapChainImages);
void updateUniformBuffer(VkDevice device, std::vector<VkDeviceMemory> uniformBuffersMemory, VkExtent2D swapChainExtent, uint32_t currentImage);
void createDescriptorPool(VkDevice device, std::vector<VkImage> swapChainImages, VkDescriptorPool& descriptorPool);
void createDescriptorSets(VkDevice device, std::vector<VkImage> swapChainImages, VkDescriptorPool& descriptorPool, VkDescriptorSetLayout descriptorSetLayout, std::vector<VkDescriptorSet>& descriptorSets,
	std::vector<VkBuffer> uniformBuffers, VkImageView textureImageView, VkSampler textureSampler);
void createImage(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
	VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
void createTextureImage(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, VkImage& textureImage, VkDeviceMemory& textureImageMemory);
VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool);
void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue, VkCommandBuffer commandBuffer);
void transitionImageLayout(VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
void copyBufferToImage(VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
VkImageView createImageView(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
void createTextureImageView(VkDevice device, VkImage textureImage, VkImageView& textureImageView);
void createTextureSampler(VkDevice device, VkPhysicalDevice physicalDevice, VkSampler& textureSampler);
VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
VkFormat findDepthFormat(VkPhysicalDevice physicalDevice);
bool hasStencilComponent(VkFormat format);
void createDepthResources(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, VkExtent2D swapChainExtent, VkImage& depthImage, VkDeviceMemory& depthImageMemory, VkImageView& depthImageView);
void loadModel(std::vector<Vertex>& vertices, std::vector<uint32_t>& indexArr);

#endif