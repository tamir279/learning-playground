#include "body_physics.cuh"
//#include <execution>
//#include <algorithm>
/*
-------------------- utility functions -------------------- 
*/

//! tuple convertions
// straight forward, convert from thrust tuple to std tuple. the center of each particle
// is a thrust::tuple. it is more convenient to use structured binding for unpacking data.
// since thrust tuple doesn't support c++17 structured binding this very function is used
template<typename T1, typename T2, typename T3>
std::tuple<T1, T2, T3> THRUSTtoSTDtuple(thrust::tuple<T1, T2, T3> dev_tuple){
    return std::make_tuple(thrust::get<0>(dev_tuple), thrust::get<1>(dev_tuple), thrust::get<2>(dev_tuple));
}

template<typename T1, typename T2, typename T3>
thrust::tuple<T1, T2, T3> STDtoTHRUSTtuple(std::tuple<T1, T2, T3> host_tuple) {
    return thrust::make_tuple(std::get<0>(host_tuple), std::get<1>(host_tuple), std::get<2>(host_tuple));
}

//! vector algebra
// thurst tuple operations
// dot product
__host__ __device__
float thrust_dot(thrust::tuple<float, float, float> v1, thrust::tuple<float, float, float> v2) {
    return thrust::get<0>(v1) * thrust::get<0>(v2) +
           thrust::get<1>(v1) * thrust::get<1>(v2) +
           thrust::get<2>(v1) * thrust::get<2>(v2);
}

float tuple_dot(std::tuple<float, float, float> v1, std::tuple<float, float, float> v2) {
    return std::get<0>(v1) * std::get<0>(v2) +
           std::get<1>(v1) * std::get<1>(v2) +
           std::get<2>(v1) * std::get<2>(v2);
}

// vector product (r*r^T) - outer product
__host__ __device__
thrust::device_vector<float> outerProduct(thrust::tuple<float, float, float> v1,
                                          thrust::tuple<float, float, float> v2) {
    thrust::device_vector<float> res(9);
    // in row major
    res[0] = thrust::get<0>(v1) * thrust::get<0>(v2); res[1] = thrust::get<0>(v1) * thrust::get<1>(v2);
    res[2] = thrust::get<0>(v1) * thrust::get<2>(v2); res[3] = thrust::get<1>(v1) * thrust::get<0>(v2);
    res[4] = thrust::get<1>(v1) * thrust::get<1>(v2); res[5] = thrust::get<1>(v1) * thrust::get<2>(v2);
    res[6] = thrust::get<2>(v1) * thrust::get<0>(v2); res[7] = thrust::get<2>(v1) * thrust::get<1>(v2);
    res[8] = thrust::get<2>(v1) * thrust::get<2>(v2);

    // return resulting matrix
    return res;
}

//! add matrices together (in flattened row major thrust device vector format) - in host =>
//! for small matrices ONLY (2x2, 3x3, ...), in device => could be appropriate for bigger sizes. 
//! but no need for that, accLigAlg contains enough optimization for large matrix operations...
__host__ __device__
thrust::device_vector<float> add_matrices(thrust::device_vector<float> m1,
                                          thrust::device_vector<float> m2,
                                          const int size,
                                          bool exec_policy,
                                          bool sign) {
    // result
    thrust::device_vector<float> res(size);
    // determine operator
    (sign) ? thrust_wrapper_transform(exec_policy, m1.begin(), m1.end(), m2.begin(), m2.end(), res.begin(), thrust::plus<float>()) :
             thrust_wrapper_transform(exec_policy, m1.begin(), m1.end(), m2.begin(), m2.end(), res.begin(), thrust::minus<float>());
    return res;
}

//! inertia tensor element calculations
// compute inertia kernel - single particle inertia - m_i*(r_i^T * r_i * identity - r_i * r_i^T)
// current calculation depends on a previous calculation of the current center mass coordinates - 
// r_i is relative to center mass -> I represents mass distribution in local coordinates.
__host__ __device__
thrust::device_vector<float> compute_ineria_element(particle p) {
    // get matrix data
    auto scale1 = thrust_dot(p.center, p.center);  auto m1 = outerProduct(p.center, p.center);
    // scale the identity
    thrust::device_vector<float> scaled_identity(9);
    for (int i = 0; i < 9; i++) {
        scaled_identity[i] = (!(i % 4)) ? p.mass * scale1 : 0.0f;
        m1[i] *= p.mass;
    }
    // subtruct the outer product from the scaled identity
    return add_matrices(scaled_identity, m1, 9, true, false);
}

//! kernel functor for two tasks at once :
/*
1) - calculating single particle inertia - m_i*(r_i^T * r_i * identity - r_i * r_i^T)
2) - summig over all inertia matrices for getting the total inertia tensor :
     I = sum_i (m_i*(r_i^T * r_i * identity - r_i * r_i^T)) i = 1, 2, ... , N_particles
*/
struct addInertiaElements : public thrust::binary_function<particle, particle, thrust::device_vector<float>> {
    __host__ __device__ thrust::device_vector<float> operator()(particle p1, particle p2) {
        auto p1_inertia_element = compute_ineria_element(p1);
        auto p2_inertia_element = compute_ineria_element(p2);
        return add_matrices(p1_inertia_element, p2_inertia_element, 9, true, true);
    }
};

/*
A^-1 = (1/det(A))*(minors(i,j))
*/
thrust::device_vector<float> inverse_3x3_mat(thrust::device_vector<float> mat) {
    float det = mat[0] * (mat[4] * mat[8] - mat[5] * mat[7]) +
                mat[1] * (mat[5] * mat[6] - mat[3] * mat[8]) +
                mat[2] * (mat[3] * mat[7] - mat[4] * mat[6]);

    if (!det) throw std::runtime_error("sinular inertia matrix is not possible!");
    // get inverse
    thrust::device_vector<float> inv(9);
    inv[0] = (mat[4] * mat[8] - mat[5] * mat[7]) / det; inv[1] = (mat[2] * mat[7] - mat[1] * mat[8]) / det;
    inv[2] = (mat[1] * mat[5] - mat[2] * mat[4]) / det; inv[3] = (mat[5] * mat[6] - mat[3] * mat[8]) / det;
    inv[4] = (mat[0] * mat[8] - mat[2] * mat[6]) / det; inv[5] = (mat[2] * mat[3] - mat[0] * mat[5]) / det;
    inv[6] = (mat[3] * mat[7] - mat[4] * mat[6]) / det; inv[7] = (mat[1] * mat[6] - mat[0] * mat[7]) / det;
    inv[8] = (mat[0] * mat[4] - mat[1] * mat[3]) / det;

    return inv;
}

// flatten matrix types to vectors
// matrix type definition: container of tuples/ other containers
//! -> container of other containers => container of types
std::vector<float> flatten_3(std::vector<std::tuple<float, float, float>> matrix_type) {
    std::vector<float> res;
    for (auto& tuple_elem : matrix_type) {
        auto [ex, ey, ez] = tuple_elem;
        res.push_back(ex); res.push_back(ey); res.push_back(ez);
    }
    return res;
}

// reverses flattening
std::vector<std::tuple<float, float, float>> deflatten_3(std::vector<float> vector_type) {
    std::vector<std::tuple<float, float, float>> res;
    for (int i = 0; i < vector_type.size(); i += 3) {
        res.push_back(std::make_tuple(vector_type[i], vector_type[i + 1], vector_type[i + 2]));
    }
    return res;
}

enum operation {
    RIGHT_TRANSPOSE_ONLY,
    LEFT_TRANSPOSE_ONLY,
    BOTH_TRANSPOSE,
    NO_TRANSPOSE
};

/*
specific case for matrix multiplication - A*B^T = |a1|                      |a1b1 a1b2 a1b3|
                                                  |a2|  x [b1 ,b2, b3] =    |a2b1 a2b2 a3b3|   , aibj = dot(ai, bj) 
                                                  |a3|                      |a3b1 a3b2 a3b3|          = aix*bjx + aiy*bjy + aiz*bjz
*/
auto multiply_3_rTranspose(std::vector<std::tuple<float, float, float>> m1,
                           std::vector<std::tuple<float, float, float>> m2) {

    std::vector<float> flattenResult;
    for (auto& row1 : m1) {
        for (auto& row2 : m2) {
            /*
              if the rows of m2 are columns - flatten result will be : [row01*row02, row01*row12, row01*row22,
                                                                        row11*row02, row11*row12, row11*row22,
                                                                        row21*row02, row21*row12, row21*row22]
              */
            flattenResult.push_back(tuple_dot(row1, row2));
        }
    }
    return deflatten_3(flattenResult);
}

// change columns to be rows
std::vector<std::tuple<float, float, float>> reorganize_mat3(std::vector<std::tuple<float, float, float>> m) {
    auto [r1x, r1y, r1z] = m[0]; auto [r2x, r2y, r2z] = m[1]; auto [r3x, r3y, r3z] = m[2];
    return { std::make_tuple(r1x, r2x, r3x),
             std::make_tuple(r1y, r2y, r3y),
             std::make_tuple(r1z, r2z, r3z) };
}

// multiply two tuple matrices
auto multiply_3(std::vector<std::tuple<float, float, float>> m1,
                std::vector<std::tuple<float, float, float>> m2,
                operation status) {

    // transposed matrices - for different transpose cases - the function reorganizes the matrices
    // in order to be in right transpose format for using the multiply_3_rTranspose function
    std::vector<std::tuple<float, float, float>> transposed1 = reorganize_mat3(m1);
    std::vector<std::tuple<float, float, float>> transposed2 = reorganize_mat3(m2);
    // calculation of multiplication results using tuple dot - first case : result is all of the combinations
    // of row dot products
    return (status == RIGHT_TRANSPOSE_ONLY) ? multiply_3_rTranspose(m1, m2) :
           (status == LEFT_TRANSPOSE_ONLY) ? multiply_3_rTranspose(transposed1, transposed2) :
           (status == BOTH_TRANSPOSE) ? multiply_3_rTranspose(transposed1, m2) :
           multiply_3_rTranspose(m1, transposed2);
}

/*
-------------------- library functions -------------------- 
*/

// calculate ideal gas pressure that approximate the pressure on the surface of a soft body
// p = nRT/V 
float calculatePressure(float n, float R, float T, float V) {
    return n * R * T / V;
}



// the force distribution will consist of a flatten vector of triplets representing coordinates
// in 3D space (force direction)
void rigid_body::initForceDistribution(){
    // reset all initial forces on all particles to be gravity (uniform distribution) 
    for(auto& elem : particles){
        rigidState[FORCE_DISTRIBUTION].push_back(std::make_tuple(0.0f, 0.0f, -elem.mass * G));
    }
}

// done after loading the model data into particles array
void rigid_body::initCenterMass(){
    // center mass is 3 size vector
    rigidState[CENTER_MASS] = { std::make_tuple(0.0f, 0.0f, 0.0f) };
    for(auto& particle : particles){
        // uniform distribution initialization
        particle.mass = mass / (float)particles.size();
        // get particle center mass 
        auto [x, y, z] = THRUSTtoSTDtuple<float, float, float>(particle.center);
        // same as adding coordinate/particles.size() 
        std::get<0>(rigidState[CENTER_MASS][0]) +=  x * particle.mass / mass;
        std::get<1>(rigidState[CENTER_MASS][0]) +=  y * particle.mass / mass;
        std::get<2>(rigidState[CENTER_MASS][0]) +=  z * particle.mass / mass;
    }
}

// calculate distances in local coordinates relative to center mass
void rigid_body::initRelativeDistances() {
    for (auto& p : particles) {
        // get elem data
        auto [x, y, z] = THRUSTtoSTDtuple<float, float, float>(p.center);
        auto [cx, cy, cz] = rigidState[CENTER_MASS][0];
        relativeParticles.push_back({ thrust::make_tuple(x - cx, y - cy, z - cz), p.radius, p.mass });
    }
}

// initialize linear velocity vector - of center mass
void rigid_body::initLinearVelocity() {
    rigidState[LINEAR_VELOCITY] = { std::make_tuple(0.0f, 0.0f, 0.0f) };
}

/*
gereral rotation breaks down into multiplication of rotations on all directions:
R = R_x * R_y * R_z = Ix * Iy * Iz = I
*/
void rigid_body::initRotation(){
    rigidState[ROTATION] = { std::make_tuple(1.0f, 0.0f, 0.0f),
                             std::make_tuple(0.0f, 1.0f, 0.0f),
                             std::make_tuple(0.0f, 0.0f, 1.0f) };
}

/*
void rigid_body::initLinearMomentum(){
    rigidState[LINEAR_MOMENTUM] = { std::make_tuple(0.0f, 0.0f, 0.0f) };
}
*/

// L_init = (0,0,0)
void rigid_body::initAngularMomentum(){
    rigidState[ANGULAR_MOMENTUM] = { std::make_tuple(0.0f, 0.0f, 0.0f) };
}

void rigid_body::initTotalExternalForce() {
    rigidState[TOTAL_EXTERNAL_FORCE] = { std::make_tuple(0.0f, 0.0f, 0.0f) };
    for (auto& f : rigidState[FORCE_DISTRIBUTION]) {
        auto [x, y, z] = f;
        std::get<0>(rigidState[TOTAL_EXTERNAL_FORCE][0]) += x;
        std::get<1>(rigidState[TOTAL_EXTERNAL_FORCE][0]) += y;
        std::get<2>(rigidState[TOTAL_EXTERNAL_FORCE][0]) += z;
    }
}

// initial torque is corresponding to inital force - gravity
void rigid_body::initTorque(){
    rigidState[TORQUE] = { std::make_tuple(0.0f, 0.0f, 0.0f) };
    for(auto& particle : relativeParticles){
        // get particle centers and force on particular particle
        auto [p_x, p_y, p_z] = THRUSTtoSTDtuple<float, float, float>(particle.center);

        // accumulate torque elements - Ti = cross(ri, fi), 
        // in init the only force applied is gravity, hence fi = (0, 0, -G*mi).
        // since the cross product yields results perpendicular to the force, the z torque is 0
        std::get<0>(rigidState[TORQUE][0]) -= p_y * G * particle.mass;
        std::get<1>(rigidState[TORQUE][0]) += p_x * G * particle.mass;
    }
}

// calculate initial inertia tensor
void rigid_body::initInverseInertiaTensor(){
    // loop over all particles :
    /*
    kernel_i = m_i * (r_i^T * r_i *I - r_i * r_i^T), I = {{1,0,0},{0,1,0},{0,0,1}}, r_i particle center
    => I0 = sum(kernel_i), i >= 1, i <= numParticles
    */
    thrust::device_vector<float> init(9); thrust::fill_n(thrust::device, init.begin(), 9, (const float)0);
    auto inertiaMatrix = thrust_wrapper_reduce(true, relativeParticles.begin(), relativeParticles.end(), init, addInertiaElements());
    // calculate the inverse matrix
    auto invI = inverse_3x3_mat(inertiaMatrix);
    // get the inverse matrix to the bodyState mapping
    rigidState[INVERSE_INERTIA_TENSOR] = { std::make_tuple(invI[0], invI[1], invI[2]),
                                           std::make_tuple(invI[3], invI[4], invI[5]),
                                           std::make_tuple(invI[6], invI[7], invI[8]) };
    //! get invariant body inverse inertia tensor - the inertia tensor is changes only via rotations!
    //! this is the same as the body inertia tensor in world coordinates if the initial rotation is the IDENTITY
    //! which the simulation resets to be.
    inverseBodyInertia = rigidState[INVERSE_INERTIA_TENSOR];
}

void rigid_body::initAngularVelocity() {
    rigidState[ANGULAR_VELOCITY] = { std::make_tuple(0.0f, 0.0f, 0.0f) };
}

void rigid_body::initDampingMatrix() {

}

void rigid_body::initDisplacementVector() {
    for (int i = 0; i < systemSize; i++) {
        (Displacement_t_2dt.data)[i] = 0.0f;
        (Displacement_t_dt.data)[i] = 0.0f;
        (Displacement.data)[i] = 0.0f;
    }
}

void rigid_body::init() {
    initForceDistribution();
    initCenterMass();
    initRelativeDistances(); 
    initLinearVelocity();
    initRotation();
    //initLinearMomentum();
    initAngularMomentum();
    initTotalExternalForce();
    initTorque();
    initInverseInertiaTensor();
    initAngularVelocity();
    initDampingMatrix();
    initDisplacementVector();
}

/*
------------------------------------------------------------------------------------
------------------------------ advance a step in time ------------------------------
------------------------------------------------------------------------------------
*/

void rigid_body::calculateCenterMass() {
    auto [cx, cy, cz] = rigidState[CENTER_MASS][0];
    auto [vx, vy, vz] = rigidState[LINEAR_VELOCITY][0];
    rigidState[CENTER_MASS][0] = std::make_tuple(cx + dt * vx, cy + dt * vy, cz + dt * vz);
}

void rigid_body::calculateLinearVelocity() {
    auto [vx, vy, vz] = rigidState[LINEAR_VELOCITY][0];
    auto [fx, fy, fz] = rigidState[TOTAL_EXTERNAL_FORCE][0];
    rigidState[LINEAR_VELOCITY][0] = std::make_tuple(vx + dt * fx / mass,
                                                     vy + dt * fy / mass,
                                                     vz + dt * fz / mass);
}

/*
process:
R_n -> q_n -> q_n+1 = q_n + DT/2 w_n*q_n -> R_n+1
*/
void rigid_body::calculateRotation(){
    // represent angular velocity as a quaternion
    auto [wx, wy, wz] = rigidState[ANGULAR_VELOCITY][0];
    std::valarray<float> w_vector = { wx, wy, wz };
    quaternion w(0.0f, w_vector); 
    // represent rotation matrix as a quaternion
    quaternion q_n; q_n.createUnitQuarenion(flatten_3(rigidState[ROTATION]));
    // make sure q_n represents rotation
    q_n.convertToRotationQuaternionRepresentation();
    // advance in time
    q_n = q_n + (dt / 2.0f) * w * q_n;
    // get updated rotation matrix
    rigidState[ROTATION] = deflatten_3(q_n.getRotationMatrixFromUnitQuaternion());
}

void rigid_body::calculateAngularMomentum() {
    auto [tx, ty, tz] = rigidState[TORQUE][0];
    auto [Lx, Ly, Lz] = rigidState[ANGULAR_MOMENTUM][0];
    rigidState[ANGULAR_MOMENTUM][0] = std::make_tuple(Lx + dt * tx, Ly + dt * ty, Lz + dt * tz);
}

void rigid_body::calculateForceDistribution() {
    // for calculating with thrust - switch to thrust::tuples
}

void rigid_body::calculateTotalExternalForce() {
   // for calculating with thrust - switch to thrust::tuples
}

void rigid_body::calculateTorque() {
    // loop over force distribution and particles to get the cross products.
    // for calculating with thrust - switch to thrust::tuples
}

/*
I^-1 = R(t) (I_body)^-1 R(t)^T. initial 
*/
void rigid_body::calculateInverseInertiaTensor() {
    auto rt_Iinv_res = multiply_3(rigidState[ROTATION], inverseBodyInertia, NO_TRANSPOSE);
    rigidState[INVERSE_INERTIA_TENSOR] = multiply_3(rt_Iinv_res, rigidState[ROTATION], RIGHT_TRANSPOSE_ONLY);
}

// w_n+1 = (I_n+1)^-1 * L_n+1
void rigid_body::calculateAngularVelocity() {
    std::get<0>(rigidState[ANGULAR_VELOCITY][0]) = tuple_dot(rigidState[INVERSE_INERTIA_TENSOR][0],
                                                             rigidState[ANGULAR_MOMENTUM][0]);
    std::get<1>(rigidState[ANGULAR_VELOCITY][0]) = tuple_dot(rigidState[INVERSE_INERTIA_TENSOR][1],
                                                             rigidState[ANGULAR_MOMENTUM][0]);
    std::get<2>(rigidState[ANGULAR_VELOCITY][0]) = tuple_dot(rigidState[INVERSE_INERTIA_TENSOR][2],
                                                             rigidState[ANGULAR_MOMENTUM][0]);
}

/*
equation : 
(Mh + Kd - Ks) * Q_t = Mh * (2Q_(t-dt) - Q_(t-2dt)) - N * Ftotal_t
*/
void rigid_body::updateDisplacementVector() {
    LinearSolver<float> solver(CHOL, true);
    auto lhsMat = infMassDistrib + DampingDistribMatrix - DampingMatrix;
    // for sparse equation
    Sparse_mat<float> lhs_sparse(lhsMat.M, lhsMat.N, lhsMat.memState);
    lhs_sparse.copyData(lhsMat); lhs_sparse.createCSR();
    // TODO : add the calculation with the total external force
    vector<float> rhsVec = infMassDistrib % (Displacement_t_dt + Displacement_t_dt - Displacement_t_2dt);
    // feed to solver
    solver.I_matrix(lhs_sparse); solver.I_vector(rhsVec); 
    // transfer old data to displacement in times t-dt, t-2dt before calculating solution
    Displacement_t_2dt = Displacement_t_dt;
    Displacement_t_dt = Displacement;
    // solve
    Displacement = solver.Solve();
}

/*
r_total = r_cm + q_r*r0*q_r^-1 + dr, dr = Q_t[i]
*/
void rigid_body::updatePosition() {

}

void rigid_body::advance() {
    // rigid body data
    calculateCenterMass();
    calculateLinearVelocity();
    calculateRotation();
    calculateAngularMomentum();
    calculateInverseInertiaTensor();
    calculateAngularVelocity();
    calculateForceDistribution();
    calculateTotalExternalForce();
    calculateTorque();
    // soft body data
    decomposeExternalForces();
    updateDisplacementVector();
    getOuterSurfaceDeformation();
    // updata the body position - all of particle position changes
    // r_new = r_cm + q_r*r0*q_r^-1 - angular + linear
    // r_total = r_cm + q_r*r0*q_r^-1 + dr, dr is the inner particle pertubation made from external forces.
    updatePosition();
}