#include <unordered_map>
#include <string>
#include <tuple>
#include <thrust/tuple.h>
#include <thrust/pair.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "quaternion_math.h"
#include "thrustWrappers.cuh"
#include "3x3fastLinearAlgebra.cuh"
#include "acceleratedLinearAlgebra.cuh"


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

enum body_type {
    REGULAR,
    IMMOVABLE,
    GROUND
};

struct particle {
    float3 center;
    float radius;
    float mass;
};

// for cuda kernels - new struct consisting polygon data
// it is created for using polygonal structures also with cuda kernels
struct cudaPoly {
    float3 v1;
    float3 v2;
    float3 v3;
};

class geometricData{
public:
    // loaded data from model .obj file.
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<cudaPoly> surfacePolygons;
    std::vector<int> indices;

    // built data for fast calculations
    std::vector<float3> bounding_box;
    std::tuple<float, float, float> boxDims;

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

    void buildParticleGrid(std::vector<float3>& data);
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
    body_type type; // defines if a body will have standard physical dynamics model
                    // if not - it will be immovable/ground body -> M = infinity, it is not affected by
                    // other bodies

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
    // K_spring
    mat<float> stiffnessMatrix;
    // Z_damping
    mat<float> DampingMatrix;
    // M^-1
    Sparse_mat<float> invMassMatrix;
    // linear approximation of the spring force derivative - hessian of the velocity
    mat<float> springForceHessian;
    // zeta * q_n = F_ext + (Lambda + zeta)q_n - q_n-1
    mat<float> Zeta;
    // helper identity matrix 
    mat<float> identity;
    // during simulation it is needed to save the displacement vector at times t, t-dt, t-2dt
    vector<float> Displacement_n; 
    vector<float> Displacement_n_1;
    vector<float> Displacement_n_2;
    // linear matrix equation solver
    LinearSolver<float> solver;

    /*
    ---------------------------------------
    ------------ body methods -------------
    ---------------------------------------
    */

    rigid_body(const std::string modelPath, const float _mass, const float _rigidity, const float time_step, const int size, 
               const float moleNum, const float idealGasConst, const float temperature, const body_type _type) : 
    stiffnessMatrix(3*size, 3*size, memLocation::DEVICE),
    DampingMatrix(3*size, 3*size, memLocation::DEVICE),
    invMassMatrix(3*size, 3*size, memLocation::DEVICE),
    springForceHessian(3*size, 3*size, memLocation::DEVICE),
    Zeta(3*size, 3*size, memLocation::DEVICE),
    identity(3*size, 3*size, memLocation::DEVICE),
    Displacement_n(3*size, 1, memLocation::DEVICE),
    Displacement_n_1(3*size, 1, memLocation::DEVICE),
    Displacement_n_2(3*size, 1, memLocation::DEVICE),
    solver((DECOMP)QR, false),
    bodySurface(modelPath){
        // get geometric data
        readGeometryToData(modelPath);
        calculateRestitutionConstant(_rigidity);
        mass = _mass; rigidity = _rigidity; dt = time_step; systemSize = size;
        n = moleNum; R = idealGasConst; T = temperature; type = _type;
    }

    rigid_body(const rigid_body& body) : 
        stiffnessMatrix{ body.stiffnessMatrix }, DampingMatrix{ body.DampingMatrix },
        invMassMatrix{ body.invMassMatrix }, springForceHessian{ body.springForceHessian }, Zeta{ body.Zeta },
        Displacement_n{ body.Displacement_n }, Displacement_n_1{ body.Displacement_n_1 }, 
        Displacement_n_2{ body.Displacement_n_2 }, solver{ body.solver }, bodySurface{ body.bodySurface }{

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
    void initIdentity();
    void initSpringForceHessian();
    void initStiffnessMatrix();
    void initInvMassMatrix();
    void initDampingMatrix();
    void initZeta();
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
    void updateDisplacementVector();
    void updateStiffnessMatrix();
    vector<float> decomposeExternalForces();
    // get the sum of all changes in place - linear + angular + inner - and update particle center positions
    void updatePosition();
};



/*
---------------------------------------------------------------------------------------
--------------------------------- COLLISION DETECTION ---------------------------------
---------------------------------------------------------------------------------------
*/


struct pairInfo {
    float priority; // alpha * (distance)^-1 + beta * (body_speeds) , alpha, beta >= 0 , alpha + beta = 1
    thrust::pair<int, int> bodies;
    thrust::pair<body_type, body_type> types;
};

// build a heap containing lower bound on distances between all bodies. body clusters that are closer 
// to one another will be given higher priority that will result in higher rate of collison checking
// MAX HEAP
class collision_heap{
public:
    std::vector<pairInfo> collisionHeap;

    collision_heap(std::vector<pairInfo> initialState) : collisionHeap{ initialState }{
        buildHeap();
    }

    // empty constructor for building a heap manually
    collision_heap(){}

    void heapify(int index, int n);

    void heapSort();

    void insert(const pairInfo element);

    // pops the highest priority -  gets the maximum value (A[0]) and pops it out of the heap
    pairInfo popMax();

    // pops specific element
    pairInfo pop(const int index);

    // update specific node
    void update(const pairInfo element, const int index);

    // update the entire heap at once
    void update(const std::vector<pairInfo> updatedHeap);

    pairInfo minHeap();

    pairInfo maxHeap();

    // assumes the heap is full of data
    void buildHeap();
};

// collision detection using scaning for collisions between specific particles in the surface of the body 
// (i.e vertex particles), there is a consideration with resting position.
// RIGHT NOW : NO FRICTION IS CONSIDERED
class collision_handler{
public:

    // the size of numBodies = |_bodyList|
    std::vector<float3> collisionImpulse;
    std::vector<int> particleIndices;
    std::vector<rigid_body> bodyList;

    collision_handler(const std::vector<rigid_body> _bodyList){
        setPriority(_bodyList, 0.5f, 0.5f, (EXT_pParam)CENTER_MASS, (EXT_pParam)LINEAR_VELOCITY);
        bodyList = _bodyList;
        initImpulseVector();
    }

    // advance collision state in time step
    void advanceInTime();

    void updateHeapData(std::vector<pairInfo> updatedHeap);

    void updateForces(std::vector<rigid_body>& _bodyList, EXT_pParam forceParam);

private:

    float velocityThreshold = 10e-2;
    float epsilon;
    collision_heap priorityHeap;

    // initializes impulse vector and particle index vector - we assume that one particle is contacted per body
    void initImpulseVector();
    void setPriority(const std::vector<rigid_body> bodies, float alpha, float beta, EXT_pParam cm, EXT_pParam v);
    void detectCollisions(); // using collisionAlgorithms
    float3 calculateImpulse(rigid_body body1, rigid_body body2, 
                            int ind1poly, int ind1vec,
                            int ind2poly, int ind2vec,
                            EXT_pParam stateParam, EXT_pParam linearV,
                            EXT_pParam angularV); // using collisionAlgorithms
};

// priority and body connection array are to be calculated in the total simulation class as a private parameter and
// a method for calculating. there it will also update by : collisons.updateHeapData(newArray);