#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

namespace colDetect{

    void constructVoroniRegions();

    void checkForClosestPoints();

    void detectCollision();
}

namespace colReact{

    mat3 vecToStruct3x3(std::vector<float3> mat);
    std::vector<float3> structToVec3x3(mat3 mat);

    // calculate K = (1/m1 + 1/m2) * 1 - r1 * I1^-1 * r1 - r2 * I2^-1 * r2
    /*
    while : m1, m2 - body masses, 1 - identity matrix 3x3, ri = (0 -riz riy , Ii - inertia tensors(w. fixed center mass)
                                                                 riz 0 -rix
                                                                 -riy rix 0)
    */
    void collisionMatrixRoutine(const float m1, const float m2,
                                mat3 r1, mat3 r2, mat3 invI1, mat3 invI2, mat3& K) {

        mat3 m_identity({ make_float3(1.0f / m1 + 1.0f / m2, 0.0f, 0.0f),
                          make_float3(0.0f, 1.0f / m1 + 1.0f / m2, 0.0f),
                          make_float3(0.0f, 0.0f, 1.0f / m1 + 1.0f / m2) });

        K = m_identity - r1 * invI1 * r1 - r2 * invI2 * r2;
    }

    void calculateCollisionMatrix(const float m1,
                                  const float m2,
                                  float3 r1,
                                  float3 r2,
                                  std::vector<float3> invI1,
                                  std::vector<float3> invI2,
                                  std::vector<float3>& K) {

        auto K_m = vecToStruct3x3(K); auto inv1_m = vecToStruct3x3(invI1); auto inv2_m = vecToStruct3x3(invI2);
        mat3 r1_m({ make_float3(0.0f, -r1.z, r1.y), make_float3(r1.z, 0.0f, -r1.x), make_float3(-r1.y, r1.x, 0.0f) });
        mat3 r2_m({ make_float3(0.0f, -r2.z, r2.y), make_float3(r2.z, 0.0f, -r2.x), make_float3(-r2.y, r2.x, 0.0f) });
        collisionMatrixRoutine(m1, m2, r1_m, r2_m, inv1_m, inv2_m, K_m);
        K = structToVec3x3(K_m);
    }

    // calculate point speed (usually used before collisions for impulse calculation)
    void calculatePointSpeed(const float3 vlin, const float3 angv, const float3 r, float3& u) {

        // u = v + w x r
        u = make_float3(vlin.x + angv.y * r.z - angv.z * r.y,
                        vlin.y + angv.z * r.x - angv.x * r.z,
                        vlin.z + angv.x * r.y - angv.y * r.x);
    }

    // calculate collision impulse w.r.t. time : -(1+e) * du(t0) = KJ(t) => J(t) = -K^-1 * (1+e) * du(t0)
    // while : du(t0) := u1(t0) - u2(t0) relative point velocity before collision. u1 - point velocity on 
    // body1, u2 - point velocity on body2, e - constant of restitution
    void calculateHeadOnCollisionReactionImpulse(const float e,
                                                 const float3 u1_begin,
                                                 const float3 u2_begin,
                                                 std::vector<float3> K,
                                                 float3& J) {

        mat3 K_m = vecToStruct3x3(K); K_m.inverse();
        J = K_m * make_float3(-(1 + e) * (u1_begin.x - u2_begin.x),
                              -(1 + e) * (u1_begin.y - u2_begin.y),
                              -(1 + e) * (u1_begin.z - u2_begin.z));
    }
    
    // v < sqrt(2 * g * epsilon)
    void detectMicroCollisions(const float epsilon, const float g, const float3 relativeSpeed, bool& state);

    // if many micro collisions are detected and the change in movement direction is low enough - 
    // the bodies are in a static sliding state
    void detectSliding(const float epsilon,
                       const float g,
                       const float3 relativeSpeed,
                       const float speedEpsilon,
                       bool& state);

    // s.v = (-miu * ux/sqrt(ux * ux + uy * uy), -miu * uy/sqrt(ux * ux + uy * uy), 1)
    // u - average collision velocity (after), miu - kinetic friction constant
    void calculateSlidingVector(const float dynamicFrictionConstant, const float3 u, float3& SV);

    // calculate the impulse J using the following rule : dJ/dj_z = SV(j_z) => J = integrate(SV)d(j_z)
    // while : SV is the sliding vector, j_z - z coordinate of J
    void calculateSlidingCollisionImpulse(const float3 slidingVector, std::vector<float3> K, float3& J);

}