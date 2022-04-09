#include "glew.h"
#include <windows.h>
#include <iostream>
#include <vector>
#include <GL/gl.h>
#include <GL/glu.h>
#include <string>
#include <fstream>
#include <sstream>
#include "renderer.h"

namespace MLE::RENDERER {

    void shader::setShaderList() {
        shaders = {
            {"VERTEX", GL_VERTEX_SHADER}, {"FRAGMENT", GL_FRAGMENT_SHADER},
            {"GEOMETRY", GL_GEOMETRY_SHADER}, {"COMPUTE", GL_COMPUTE_SHADER} };
    }

    void shader::compileShader(unsigned int& shader, const char* shaderCode, std::string type) {
        shader = glCreateShader(shaders[type]);
        glShaderSource(shader, 1, &shaderCode, NULL);
        glCompileShader(shader);
        checkCompileErrors(shader, type);
    }

    shader::shader(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr) {
        setShaderList();
        // the code itself read from the shaders
        std::string vertexCode;
        std::string fragmentCode;
        std::string geometryCode;
        // file streams for each shader
        std::ifstream vShaderFile;
        std::ifstream fShaderFile;
        std::ifstream gShaderFile;

        // ensure ifstream objects can throw exceptions:
        vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        // try - catch for opening and reading the files
        try {
            // open files
            vShaderFile.open(vertexPath);
            fShaderFile.open(fragmentPath);
            std::stringstream vShaderStream, fShaderStream;
            // read file's buffer contents into streams
            vShaderStream << vShaderFile.rdbuf();
            fShaderStream << fShaderFile.rdbuf();
            // close file handlers
            vShaderFile.close();
            fShaderFile.close();
            // convert stream into string
            vertexCode = vShaderStream.str();
            fragmentCode = fShaderStream.str();
            // if geometry shader path is present, also load a geometry shader
            if (geometryPath != nullptr)
            {
                gShaderFile.open(geometryPath);
                std::stringstream gShaderStream;
                gShaderStream << gShaderFile.rdbuf();
                gShaderFile.close();
                geometryCode = gShaderStream.str();
            }
        }
        catch (std::ifstream::failure& Error) {
            std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << Error.what() << std::endl;
        }

        // compile and run shader program (if files are opened succesfully)
        const char* vShaderCode = vertexCode.c_str();
        const char* fShaderCode = fragmentCode.c_str();
        // compile shaders 
        unsigned int vertex, fragment, geometry;
        // vertex shader
        compileShader(vertex, vShaderCode, "VERTEX");
        // fragment shader
        compileShader(fragment, fShaderCode, "FRAGMENT");
        // geometry shader
        if (geometryPath != nullptr)
        {
            const char* gShaderCode = geometryCode.c_str();
            compileShader(geometry, gShaderCode, "GEOMETRY");
        }

        // run shaders and delete after execution
         // shader Program
        ID = glCreateProgram();
        // attach shaders
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        if (geometryPath != nullptr) { glAttachShader(ID, geometry); }
        // link the program
        glLinkProgram(ID);
        checkCompileErrors(ID, "PROGRAM");
        // delete the shaders as they're linked into our program now and no longer necessery
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        if (geometryPath != nullptr) { glDeleteShader(geometry); }
	}

    void shader::use() {
        glUseProgram(ID);
    }

    template<typename... Args>
    void shader::setUniformValue(const std::string& name, Args&... values) {
        auto argVec = { values... }; auto x = argVec.begin();
        if constexpr (sizeof...(values) > 1) {
            if (typeid(*x) == typeid(float)) {
                sizeof ...(values) == 3 ? glUniform3f(glGetUniformLocation(ID, name.c_str()), *x, *(x+1), *(x+2)) :
                    sizeof...(values) == 4 ? glUniform4f(glGetUniformLocation(ID, name.c_str()), *x, *(x+1), *(x+2), *(x+3))
                    : std::cout << "type not supported by openGL or GLSL" << std::endl;
            }
        }
        else {
            // boolean or integer value 
            if (typeid(*x) == typeid(bool) || typeid(*x) == typeid(int)) {
                glUniform1i(glGetUniformLocation(ID, name.c_str()), static_cast<int>(*x));
            }
            // specific floating point value
            else if (typeid(*x) == typeid(float))glUniform1f(glGetUniformLocation(ID, name.c_str()), *x);
            // vectors
            else if (typeid(*x) == typeid(glm::vec2))glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &(*x)[0]);
            else if (typeid(*x) == typeid(glm::vec3))glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &(*x)[0]);
            else if (typeid(*x) == typeid(glm::vec4))glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &(*x)[0]);
            // matrices
            else if (typeid(*x) == typeid(glm::mat2))glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &(*x)[0][0]);
            else if (typeid(*x) == typeid(glm::mat3))glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &(*x)[0][0]);
            else if (typeid(*x) == typeid(glm::mat4))glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &(*x)[0][0]);
            else { std::cout << "type not supported by openGL or GLSL" << std::endl; }
        }
    }

    void shader::checkCompileErrors(GLuint shader, std::string type) {
        GLint success;
        GLchar infoLog[1024];

        if (type == "PROGRAM") {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if(!success){
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n" << std::endl;
            }
        }
        else {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n" << std::endl;
            }
        }
    }

}