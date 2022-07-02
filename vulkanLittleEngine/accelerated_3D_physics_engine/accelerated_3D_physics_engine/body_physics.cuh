#include <unordered_map>
#include <string>
#include <tuple>
#include <thrust/tuple.h>
#include <thrust/pair.h>
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
               const float moleNum, const float idealGasConst, const float temperature, const body_type _type) : 
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
        n = moleNum; R = idealGasConst; T = temperature; type = _type;
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



/*
---------------------------------------------------------------------------------------
--------------------------------- COLLISION DETECTION ---------------------------------
---------------------------------------------------------------------------------------
*/


struct pairInfo {
    float priority; // alpha * (distance)^-1 + beta * (body_speeds) , alpha, beta >= 0 , alpha + beta = 1
    thrust::pair<int, int> bodies;
    thrust::pair<body_type, body_type> types;
    int samplePeriod = 0; // number of time steps to wait between collision checks
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

    std::vector<float3> collisionImpulse;

    collision_handler(std::vector<pairInfo> initialSystemData) : priorityHeap(initialSystemData){
        setSamplingRates();
    }

    void updateHeapData(std::vector<pairInfo> updatedHeap);

    void updateForces(std::vector<float3>& forceDistribution);

private:

    float velocityThreshold = 10e-2;
    collision_heap priorityHeap;

    void setSamplingRates();
    void samplePairState();
    void detectCollision();
    void calculateImpulse();
    void solveRestitutionConstrints();
    void solveNonPenertrationConstraints();
};

// priority and body connection array are to be calculated in the total simulation class as a private parameter and
// a method for calculating. there it will also update by : collisons.updateHeapData(newArray);