#include "renderer.h"

namespace MLE::RENDERER {

	auto camera::getViewMatrix() {
		return glm::lookAt(eye, center, up);
	}

	// translation forwards
	void camera::processKeyboardInput_W_() {
		eye += delta * center;
	}

	// eye translation in orthogonal direction - cross direction is right
	void camera::processKeyboardInput_A_() {
		eye -= delta * glm::cross(center, up);
	}

	// translation backwards
	void camera::processKeyboardInput_S_() {
		eye -= delta * center;
	}

	// eye translation in orthogonal direction
	void camera::processKeyboardInput_D_() {
		eye += delta * glm::cross(center, up);
	}

	void camera::processKeyboardInput(GLFWwindow* window) {
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { processKeyboardInput_W_(); }
		else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { processKeyboardInput_A_(); }
		else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { processKeyboardInput_S_(); }
		else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { processKeyboardInput_D_(); }
		else if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS){ glfwSetWindowShouldClose(window, true); }
	}

	void camera::processMouseMovement(float Xoffset, float Yoffset) {
		glm::vec3 rotationAxis = glm::cross(center, glm::vec3(Xoffset, Yoffset, 0.0));
		// calculate the rotation angle
		float sign = Xoffset * up.x + Yoffset * up.y;
		/*
		rotate center around the rotation axis - if the offset is upwards - rotate counterclockwise
		else - rotate clockwise. therefore, if the offset is upwards(mouse up) the center goes upwards
		and if offset is downwards(mouse down) the center goes downwards
		because y coordinates are from the top to bottom there is no need for addin a negative sign to the angle - 
		if the mouse goes up the offset is negative (yUp - yDown < 0)
		*/
		float angle = (sign/glm::abs(sign)) * glm::l2Norm(rotationAxis) * mouseSensitivity;
		quaternion rotation(angle, rotationAxis);
		RotateVector(center, rotation);
		RotateVector(up, rotation);

	}

	void camera::processMouseScrolling(float Yoffset) {
		FieldOfView = -Yoffset;
		// barriers - limiting the clip space between 1 to 45 degrees
		if (FieldOfView < 1.0f)FieldOfView = 1.0f;
		if (FieldOfView > 45.0f)FieldOfView = 45.0f;
	}

	void camera::mousePositionCallBack(GLFWwindow* window, double x, double y) {
		std::pair<float, float> pos = { static_cast<float>(x), static_cast<float>(y) };
		if (firstMove) { lastX = std::get<0>(pos); lastY = std::get<1>(pos); firstMove = false; }
		// update variables
		lastX = std::get<0>(pos); lastY = std::get<1>(pos); 
		// update view matrix
		processMouseMovement(std::get<0>(pos) - lastX, lastY - std::get<1>(pos));

	}

	// used for the projection matrix as the field of view setup in degrees
	void camera::mouseScrollCallBack(GLFWwindow* window, double Xoffset, double Yoffset) {
		processMouseScrolling(static_cast<float>(Yoffset));
	}

	void camera::RotateVector(glm::vec3& v, quaternion rotation) {
		quaternion vQuaternion(0, v);
		rotation.ConvertToRotationQuaternionRepresentation();
		// rotate the vector
		auto [s, vec] = rotation * vQuaternion * rotation.inverse();
		v = glm::normalize(vec);
	}
}