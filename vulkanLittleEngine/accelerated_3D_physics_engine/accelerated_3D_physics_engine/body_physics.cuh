#include <unordered_map>
#include <string>

enum EXT_pParams{
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

enum INT_pParams{
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
    geometryLoader(std::string modelPath){
        readData(modelPath);
    }

private:
    void readData(std::string modelPath);

};

class rigid_body {
public:
    // body data
    std::vector<particle> particles;
    // current body state
    std::unordered_map<EXT_pParams, std::vector<float>> rigidState;
    // current internal body state
    std::unordered_map<INT_pParams, std::vector<float>> internalState;

    rigid_body(std::string modelPath){
        readGeometryToData(modelPath);
    }

    void advance();
private:
    // data management
    void readGeometryToData(std::string modelPath);
    void transferToRawGeometricData();
    void transferToParticleData();

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

