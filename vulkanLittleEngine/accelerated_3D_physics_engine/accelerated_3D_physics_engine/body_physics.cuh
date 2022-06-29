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
    SPRING_DIRECTION,
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
    float3 center;
    float radius;
    float mass;
};

class geometricData{
public:
    // loaded data from model .obj file.
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<thrust::tuple<particle, particle, particle>> surfacePolygons;
    std::vector<int> indices;

    // built data for fast calculations
    std::vector<float3> bounding_box;

    // construction
    geometricData(const std::string modelPath){
        // load vertices, normals, surface polygons and index arrays
        readData(modelPath);
    }

    // copy constructor
    geometricData(const geometricData& geometry) {
        copyData(geometry);
    }

    // assignment operator
    geometricData& operator=(const geometricData& geometry) {
        copyData(geometry);
    }

    // convert vertices to std::vector<particle> surface
    std::vector<particle> convertVerticesToBodySurface();
    // convert vertex normals to face normals : 
    // {nv1, nv2, nv3,...} in R^(vertices.size()x1) -> {nF1, nF2, nF3,...} in R^(indices.size()/3)
    void convertToFaceNormals(const bool normalize);
    // convert from face normals to vertex normals : 
    // {nF1, nF2, nF3,...} in R^(indices.size()/3) -> {nv1, nv2, nv3,...} in R^(vertices.size()x1)
    void convertToVertexNormals(const bool normalize);
    // update vertices and normals
    void updateData();

private:
    // fit bounding box to the model
    void fitBoundingBox();
    // build all triangles by using the indices along with corresponding vertices to create 
    // a polygon array
    void getSurfacePolygons();
    void readData(const std::string modelPath);
    // copy all data
    void copyData(const geometricData& geometry);
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
    float dt; // time delta
    float n; // gas mol number
    float R; // ideal gas constant
    float T; // temprature

    // body inverse inertia tensor
    std::vector<float3> inverseBodyInertia; // body constant describing mass distributions
                                            // in world coordinates. time variant inertia tensor is local.
    /*
    -------------- dynamic data --------------
    */
    // face particles - vertices + vertex normals
    geometricData bodySurface;
    // approximate body volum
    float V_approx;

    // all of the particles consisting the body
    std::vector<particle> particles; 
    std::vector<particle> relativeParticles;
    int systemSize;
    // current body state
    std::unordered_map<EXT_pParam, std::vector<float3>> rigidState;
    /*
    -------------- internal body state --------------
    */
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
    // linear matrix equation solver
    LinearSolver<float> solver;

    /*
    ---------------------------------------
    ------------ body methods -------------
    ---------------------------------------
    */

    rigid_body(const std::string modelPath, const float _mass, const float _rigidity, const float time_step, const int size, 
               const float moleNum, const float idealGasConst, const float temperature) : 
    DampingDistribMatrix(3*size, 3*size, memLocation::DEVICE),
    DampingMatrix(3*size, 3*size, memLocation::DEVICE),
    infMassDistrib(3*size, 3*size, memLocation::DEVICE),
    Displacement(3*size, 1, memLocation::DEVICE),
    Displacement_t_dt(3*size, 1, memLocation::DEVICE),
    Displacement_t_2dt(3*size, 1, memLocation::DEVICE),
    solver(CHOL, true),
    bodySurface(modelPath){

        readGeometryToData(modelPath);
        calculateRestitutionConstant(_rigidity);
        mass = _mass; rigidity = _rigidity; dt = time_step; systemSize = size;
        n = moleNum; R = idealGasConst; T = temperature;
    }

    rigid_body(const rigid_body& body) : 
        DampingDistribMatrix{ body.DampingDistribMatrix }, DampingMatrix{ body.DampingMatrix },
        infMassDistrib{ body.infMassDistrib }, Displacement{ body.Displacement }, 
        Displacement_t_dt{ body.Displacement_t_dt }, Displacement_t_2dt{ body.Displacement_t_2dt },
        solver{ body.solver }, bodySurface{ body.bodySurface }{

        copyBodyData(body);
    }

    rigid_body& operator=(const rigid_body& body) {
        copyBodyData(body);
    }

    void init();
    void advance();

private:
    // data management
    void readGeometryToData(const std::string modelPath);
    void transferToRawGeometricData();
    void transferToParticleData();
    void calculateRestitutionConstant(const float rigidity);
    void copyBodyData(const rigid_body& body);

    // -------- initialize body state --------
    // init rigid body state
    void initSpringDirection();
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
    void calculateBodyVolume();
    void calculatePressureForce();
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
    vector<float> decomposeExternalForces();
    // get the sum of all changes in place - linear + angular + inner - and update particle center positions
    void updatePosition();
};


// collision detection using scaning for collisions between specific particles in the surface of the body 
// (i.e vertex particles), there is a consideration with resting position.
class collision_detector {
public:

    collision_detector(const rigid_body _body1, const rigid_body _body2, float _epsilon) : 
        body1{ _body1 }, body2{ _body2 }, velocityEpsilon{ _epsilon } {}

    bool Collided();

    void applyCollisionForces();

private:

    float velocityEpsilon;
    rigid_body body1;
    rigid_body body2;

    void getCollitionParticles(std::vector<float3>& v1, std::vector<float3>& v2);

};
