#include "body_physics.cuh"

/*
-------------------- utility functions -------------------- 
*/
// straight forward, convert from thrust tuple to std tuple. the center of each particle
// is a thrust::tuple. it is more convenient to use structured binding for unpacking data.
// since thrust tuple doesn't support c++17 structured binding this very function is used
template<typename T1, typename T2, typename T3>
std::tuple<T1, T2, T3> THRUSTtoSTDtuple(thrust::tuple<T1, T2, T3> dev_tuple){
    return std::make_tuple(thrust::get<0>(dev_tuple), thrust::get<1>(dev_tuple), thrust::get<2>(dev_tuple));
}

// thurst tuple operations
// dot product
float thrust_dot(thrust::tuple<float, float, float> v1, thrust::tuple<float, float, float> v2) {
    return thrust::get<0>(v1) * thrust::get<0>(v2) +
           thrust::get<1>(v1) * thrust::get<1>(v2) +
           thrust::get<2>(v1) * thrust::get<2>(v2);
}

// vector product (r*r^T) - outer product
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

// add matrices together (in flattened row major thrust device vector format) - in host =>
// for small matrices ONLY (2x2, 3x3, ...), in device => could be appropriate for bigger sizes. 
// but no need for that, accLigAlg contains enough optimization for large matrix operations...
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

// compute inertia kernel - single particle inertia - m_i*(r_i^T * r_i * identity - r_i * r_i^T)
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

// kernel functor for two tasks at once :
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
/*
-------------------- library functions -------------------- 
*/

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

/*
gereral rotation breaks down into multiplication of rotations on all directions:
R = R_x * R_y * R_z = Ix * Iy * Iz = I
*/
void rigid_body::initRotation(){
    rigidState[ROTATION] = { std::make_tuple(1.0f, 0.0f, 0.0f),
                             std::make_tuple(0.0f, 1.0f, 0.0f),
                             std::make_tuple(0.0f, 0.0f, 1.0f) };
}

// P_init = (0,0,0) 
void rigid_body::initLinearMomentum(){
    rigidState[LINEAR_MOMENTUM] = { std::make_tuple(0.0f, 0.0f, 0.0f) };
}

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
    for(auto& particle : particles){
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
    auto inertiaMatrix = thrust_wrapper_reduce(true, particles.begin(), particles.end(), init, addInertiaElements());
    // calculate the inverse matrix
    auto invI = inverse_3x3_mat(inertiaMatrix);
    // get the inverse matrix to the bodyState mapping
    rigidState[INVERSE_INERTIA_TENSOR] = { std::make_tuple(invI[0], invI[1], invI[2]),
                                           std::make_tuple(invI[3], invI[4], invI[5]),
                                           std::make_tuple(invI[6], invI[7], invI[8]) };
}

void rigid_body::initAngularVelocity() {
    rigidState[ANGULAR_VELOCITY] = { std::make_tuple(0.0f, 0.0f, 0.0f) };
}

void rigid_body::initDampingMatrix() {

}

void rigid_body::initDisplacementVector() {
    for (float* it = DisplacementVector.data; it != DisplacementVector.data + systemSize; ++it) {
        *it = 0.0f;
    }
}

void rigid_body::init() {
    initForceDistribution();
    initCenterMass();
    initRotation();
    initLinearMomentum();
    initAngularMomentum();
    initTotalExternalForce();
    initTorque();
    initInverseInertiaTensor();
    initAngularVelocity();
    initDampingMatrix();
    initDisplacementVector();
}