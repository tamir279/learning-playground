#include <unordered_map>
#include <string>
#include <tuple>
#include "quaternion_math.h"

// gravitational acceleration
const float G = 9.81;

enum EXT_pParam{
    FORCE_DISTRIBUTION,
    CENTER_MASS,
    ROTATION,
    LINEAR_MOMENTUM,
    ANGULAR_MOMENTUM,
    TOTAL_EXTERNAL_FORCE,
    TORQUE,
    INVERSE_INERTIA_TENSOR,
    ANGULAR_VELOCITY
};

enum INT_pParam{
    DAMPING_MATRIX,
    PARTICLE_DISPLACEMENT_VECTOR,
    EXTERNAL_PARTICLE_DISPLACEMENT,
    OUTER_SURFACE
};

struct particle {
    std::tuple<float, float, float> center;
    float radius;
    float mass;
};

class geometryLoader{
public:
    // loaded data from model .obj file.
    std::vector<std::tuple<float, float, float>> vertices;
    std::vector<std::tuple<float, float, float>> normals;
    std::vector<int> indices;

    // construction
    geometryLoader(const std::string modelPath){
        readData(modelPath);
    }

private:
    void readData(const std::string modelPath);

};

class rigid_body {
public:
    /*
    ---------------------------------------
    -------------- body data --------------
    ---------------------------------------
    */

    /*
    -------------- constants --------------
    */
    float mass;
    float rigidity; // range [0, 1] 
    float e; // restitution constant, range [0, 1] 

    /*
    -------------- dynamic data --------------
    */
    std::vector<particle> particles;
    // current body state
    std::unordered_map<EXT_pParam, std::vector<std::tuple<float, float, float>>> rigidState;
    // current internal body state
    std::unordered_map<INT_pParam, std::vector<float>> internalState;

    /*
    ---------------------------------------
    ------------ body methods -------------
    ---------------------------------------
    */

    rigid_body(const std::string modelPath, const float _mass, const float _rigidity){
        readGeometryToData(modelPath);
        calculateRestitutionConstant(_rigidity);
        mass = _mass; rigidity = _rigidity;
    }

    void init();
    void advance();
private:
    // data management
    void readGeometryToData(const std::string modelPath);
    void transferToRawGeometricData();
    void transferToParticleData();
    void calculateRestitutionConstant(const float rigidity);

    // -------- initialize body state --------
    // init rigid body state
    void initForceDistribution();
    void initCenterMass();
    void initRotation();
    void initLinearMomentum();
    void initAngularMomentum();
    void initTotalExternalForce();
    void initTorque();
    void initInverseInertiaTensor();
    void initAngularVelocity();

    // init internal particle state
    void initDampingMatrix();
    void initDisplacementVector();

    // -------- advance state one step --------
    // rigid body physical calculations - for the NEXT step!
    void calculateForceDistribution();
    void calculateCenterMass();
    void calculateRotation();
    void calculateLinearMomentum();
    void calculateAngularMomentum();
    void calculateTotalExternalForce();
    void calculateTorque();
    void calculateInverseInertiaTensor();
    void calculateAngularVelocity();

    // internal particle model calculations - for the NEXT step!
    void getDampingMatrix();
    void updateDisplacementVector();
    void decomposeExternalForces();
    void getOuterSurfaceDeformation();
};

