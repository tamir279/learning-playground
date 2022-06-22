#include <unordered_map>
#include <string>
#include <tuple>
#include <thrust/tuple.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "quaternion_math.h"
#include "thrustWrappers.cuh"
#include "accLinAlg.cuh"

// gravitational acceleration
const float G = 9.81;

enum EXT_pParam{
    FORCE_DISTRIBUTION,
    CENTER_MASS,
    LINEAR_VELOCITY,
    ROTATION,
    LINEAR_MOMENTUM,
    ANGULAR_MOMENTUM,
    TOTAL_EXTERNAL_FORCE,
    TORQUE,
    INVERSE_INERTIA_TENSOR,
    ANGULAR_VELOCITY
};

struct particle {
    thrust::tuple<float, float, float> center;
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
    float dt;

    // body inverse inertia tensor
    std::vector<std::tuple<float, float, float>> inverseBodyInertia; // body constant describing mass distributions
                                                                     // in world coordinates. time variant inertia tensor is local.
    /*
    -------------- dynamic data --------------
    */
    std::vector<particle> particles; 
    std::vector<particle> relativeParticles;
    int systemSize;
    // current body state
    std::unordered_map<EXT_pParam, std::vector<std::tuple<float, float, float>>> rigidState;
    // current internal body state
    // K_D
    mat<float> DampingDistribMatrix;
    // K_sigma
    mat<float> DampingMatrix;
    // M_h
    mat<float> infMassDistrib;
    // during simulation it is needed to save the displacement vector at times t, t-dt, t-2dt
    vector<float> Displacement; 
    vector<float> Displacement_t_dt;
    vector<float> Displacement_t_2dt;

    /*
    ---------------------------------------
    ------------ body methods -------------
    ---------------------------------------
    */

    rigid_body(const std::string modelPath, const float _mass, const float _rigidity, const float time_step, const int size) : 
    DampingDistribMatrix(3*size, 3*size, memLocation::DEVICE),
    DampingMatrix(3*size, 3*size, memLocation::DEVICE),
    infMassDistrib(3*size, 3*size, memLocation::DEVICE),
    Displacement(3*size, 1, memLocation::DEVICE),
    Displacement_t_dt(3*size, 1, memLocation::DEVICE),
    Displacement_t_2dt(3*size, 1, memLocation::DEVICE) {

        readGeometryToData(modelPath);
        calculateRestitutionConstant(_rigidity);
        mass = _mass; rigidity = _rigidity; dt = time_step; systemSize = size;
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
    void initRelativeDistances();
    void initLinearVelocity();
    void initRotation();
    //void initLinearMomentum();
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
    void calculateLinearVelocity();
    void calculateRotation();
    //void calculateLinearMomentum();
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
    // get the sum of all changes in place - linear + angular + inner - and update particle center positions
    void updatePosition();
};

