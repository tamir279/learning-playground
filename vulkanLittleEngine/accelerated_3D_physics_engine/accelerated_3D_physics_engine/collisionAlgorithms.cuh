#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "body_physics.cuh"

#define NOT_FOUND 420.69

namespace colDetect{

    __host__ __device__
    float3 negateVector(float3 v){
        return make_float3(-v.x, -v.y, -v.z);
    }

    __host__ __device__ 
    float tupleDot(float3 u, float3 vi, float3 ve){
        return u.x * (ve.x - vi.x) + u.y * (ve.y - ve.y) + u.z * (ve.z - ve.z);
    }

     __host__ __device__ 
    float tupleDot(float3 ui, float3 ue, float3 vi, float3 ve){
        return (ue.x - ui.x) * (ve.x - vi.x) +
               (ue.y - ui.y) * (ve.y - ve.y) + 
               (ue.z - ui.z) * (ve.z - ve.z);
    }

    __host__ __device__ 
    float3 geometricPolygonCenter(cudaPoly polygon){
        return make_float3((1.0f/3.0f) * (polygon.v1.x + polygon.v2.x + polygon.v3.x),
                           (1.0f/3.0f) * (polygon.v1.y + polygon.v2.y + polygon.v3.y),
                           (1.0f/3.0f) * (polygon.v1.z + polygon.v2.z + polygon.v3.z));
    }

    __host__ __device__
    float3 tupleCross(float3 u, float3 vi, float3 ve, cudaPoly polygon){
        // calculate geometric center as a reference point for correct "inside" of the region
        // direction
        auto pGeom = geometricPolygonCenter(polygon);
        // calculate the initial direction of the voronoi plane normal : Nei = Np x (ve - vi)
        auto N = make_float3(u.y * (ve.z - vi.z) - u.z * (ve.y - vi.y),
                             u.z * (ve.x - vi.x) - u.x * (ve.z - vi.z),
                             u.x * (ve.y - vi.y) - u.y * (ve.x - vi.x));
        // check if the normal is pointing in the "inside" region (i.e. dot product <pG - vi, Nei>)
        return (tupleDot(N, vi, pGeom) >= 0) ? N : negateVector(N);
        
    }

    // calculate the normals to the voronoi planes of each polygon - for specific polygon i
    __host__ __device__
    thrust::tuple<float3, float3, float3> constructVoronoiRegion(cudaPoly polygon, float3 normal){
        thrust::tuple<float3, float3, float3> resN;
        // calculate Ne1, Ne2, Ne3
        return thrust::make_tuple(tupleCross(normal, polygon.v3, polygon.v1, polygon),
                                  tupleCross(normal, polygon.v1, polygon.v2, polygon),
                                  tupleCross(normal, polygon.v2, polygon.v3, polygon));
    }

    __host__ __device__
    bool checkVoronoiConditionForClosestPoints(float3 u1, float3 u2,
                                               cudaPoly polygon1, 
                                               float3 normal1,
                                               cudaPoly polygon2,
                                               float3 normal2){
        // build voronoi regions for each polygon
        auto region1 = constructVoronoiRegion(polygon1, normal1);
        auto region2 = constructVoronoiRegion(polygon2, normal2);
        // check each point if it is inside the voronoi region of the other point
        // v1 in F(v2)
        bool in2 = (tupleDot(thrust::get<0>(region2), polygon2.v1, u1) >= 0) &&
                   (tupleDot(thrust::get<1>(region2), polygon2.v2, u1) >= 0) &&
                   (tupleDot(thrust::get<2>(region2), polygon2.v3, u1) >= 0) &&
                   (tupleDot(normal2, polygon2.v1, u1) >= 0);
        // v2 in F(v1)
        bool in1 = (tupleDot(thrust::get<0>(region1), polygon2.v1, u2) >= 0) &&
                   (tupleDot(thrust::get<1>(region1), polygon2.v2, u2) >= 0) &&
                   (tupleDot(thrust::get<2>(region1), polygon2.v3, u2) >= 0) &&
                   (tupleDot(normal1, polygon2.v1, u2) >= 0);
        return in1 && in2;
    }

    // create a cuda kernel to run over all options
    __global__ void detect(float3* normals1,
                           float3* normals2,
                           cudaPoly* polyArr1,
                           cudaPoly* polyArr2,
                           int p1Size,
                           int p2Size,
                           float eps_sq,
                           int* polyInd1,
                           int* vec1, 
                           int* polyInd2,
                           int* vec2,
                           bool* detected){
        
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
        if(i < p1Size && j < p2Size){
            float3 poly1[3] = { polyArr1[i].v1, polyArr1[i].v2, polyArr1[i].v3 };
            float3 poly2[3] = { polyArr2[j].v1, polyArr2[j].v2, polyArr2[j].v3 }; 
            for(int k = 0; k < 3; k++){
                for(int m = 0; m < 3; m++){
                    float dist = tupleDot(poly2[m], poly1[k], poly2[m], poly1[k]);
                    if(dist < eps_sq && checkVoronoiConditionForClosestPoints(poly1[k], poly2[m],
                                                                              polyArr1[i], normals1[i], 
                                                                              polyArr2[j], normals2[j])){
                        *polyInd1 = i; *polyInd2 = j; *vec1 = k; *vec2 = m; *detected = true;
                    }
                }
            }
        }
        *detected = false;
     }

    std::tuple<int, int, int, int> detectCollision(std::vector<cudaPoly> polygons1,
                                                   std::vector<cudaPoly> polygons2,
                                                   std::vector<float3> normals1,
                                                   std::vector<float3> normals2,
                                                   float epsilon){
        // transfer data to device
        thrust::device_vector<cudaPoly> p1(polygons1.begin(), polygons1.end());
        thrust::device_vector<cudaPoly> p2(polygons2.begin(), polygons2.end());
        thrust::device_vector<float3> n1(normals1.begin(), normals1.end());
        thrust::device_vector<float3> n2(normals2.begin(), normals2.end());
        
        // get detection result and the two affected particles if a collision is detected
        // indices of particles
        int polyInd1; int polyInd2; int vecInd1; int vecInd2; bool res;
        // activate the cuda kernel
        int N = (int)polygons1.size(); int M = (int)polygons2.size();
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
        detect <<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(n1.data()),
                                                     thrust::raw_pointer_cast(n2.data()),
                                                     thrust::raw_pointer_cast(p1.data()),
                                                     thrust::raw_pointer_cast(p2.data()),
                                                     (int)polygons1.size(),
                                                     (int)polygons2.size(),
                                                     epsilon * epsilon,
                                                     &polyInd1, &vecInd1,
                                                     &polyInd2, &vecInd2, &res);
                                                                
        return (res) ? std::make_tuple(polyInd1, vecInd1, polyInd2, vecInd2) : std::make_tuple(-1, -1, -1, -1);
    }
}

namespace colReact{

    mat3 vecToStruct3x3(std::vector<float3> mat) {
        mat3_data matData; 
        matData.row1 = mat[0]; matData.row2 = mat[1]; matData.row3 = mat[2];
        mat3 res(matData);
        return res;
    }
    std::vector<float3> structToVec3x3(mat3 mat) {
        return { mat.data.row1, mat.data.row2, mat.data.row3 };
    }

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