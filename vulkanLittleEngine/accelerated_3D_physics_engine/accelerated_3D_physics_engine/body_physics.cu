#include "body_physics.cuh"
/*
-------------------- utility functions -------------------- 
*/

//! tuple convertions
// straight forward, convert from float3 to std tuple. the center of each particle
// is a float3. it is more convenient to use structured binding for unpacking data.
// since float3/double3 doesn't support c++17 structured binding this very function is used
std::tuple<float, float, float> THRUSTtoSTDtuple(float3 dev_tuple){
    return std::make_tuple(dev_tuple.x, dev_tuple.y, dev_tuple.z);
}

float3 STDtoTHRUSTtuple(std::tuple<float, float, float> host_tuple) {
    return make_float3(std::get<0>(host_tuple), std::get<1>(host_tuple), std::get<2>(host_tuple));
}

//! vector algebra
// cuda float3 operations
// dot product
__host__ __device__
float thrust_dot(float3 v1, float3 v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

float tuple_dot(std::tuple<float, float, float> v1, std::tuple<float, float, float> v2) {
    return std::get<0>(v1) * std::get<0>(v2) +
           std::get<1>(v1) * std::get<1>(v2) +
           std::get<2>(v1) * std::get<2>(v2);
}

// cross product
__host__ __device__
float3 thrust_cross(float3 v1, float3 v2){
    return make_float3(v1.y * v2.z - v1.z * v2.y,
                       v1.z * v2.x - v1.x * v2.z,
                       v1.x * v2.y - v1.y * v2.x);
}

// addition/subrtaction
__host__ __device__ 
float3 thrust_plus(float3 v1, float3 v2){
    return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ 
float3 thrust_minus(float3 v1, float3 v2){
    return make_float3(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z);
}

// vector size ||v||_2 = sqrt(vx^2 + vy^2 + vz^2)
__host__ __device__
float thrust_L2(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// vector product (r*r^T) - outer product
__host__ __device__
thrust::device_vector<float> outerProduct(float3 v1, float3 v2) {
    thrust::device_vector<float> res(9);
    // in row major
    res[0] = v1.x * v2.x; res[1] = v1.x * v2.y;
    res[2] = v1.x * v2.z; res[3] = v1.y * v2.x;
    res[4] = v1.y * v2.y; res[5] = v1.y * v2.z;
    res[6] = v1.z * v2.x; res[7] = v1.z * v2.y;
    res[8] = v1.z * v2.z;

    // return resulting matrix
    return res;
}

// multiply thrust tuple by a scalar
__host__ __device__
void multiply_scalar(const float scalar, float3& v) {
    v.x *= scalar;
    v.y *= scalar;
    v.z *= scalar;
}

// normalize tuple vector
__host__ __device__
void normalize_thrust_tuple(float3& v) {
    const float normFactor = 1.0f / thrust_L2(v);
    multiply_scalar(normFactor, v);
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
    __host__ __device__ thrust::device_vector<float> operator()(const particle& p1, const particle& p2) {
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
std::vector<float> flatten_3(std::vector<float3> matrix_type) {
    std::vector<float> res;
    for (auto& tuple_elem : matrix_type) {
        auto [ex, ey, ez] = THRUSTtoSTDtuple(tuple_elem);
        res.push_back(ex); res.push_back(ey); res.push_back(ez);
    }
    return res;
}

// reverses flattening
std::vector<float3> deflatten_3(std::vector<float> vector_type) {
    std::vector<float3> res;
    for (int i = 0; i < vector_type.size(); i += 3) {
        res.push_back(make_float3(vector_type[i], vector_type[i + 1], vector_type[i + 2]));
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
auto multiply_3_rTranspose(std::vector<float3> m1, std::vector<float3> m2) {

    std::vector<float> flattenResult;
    for (auto& row1 : m1) {
        for (auto& row2 : m2) {
            /*
              if the rows of m2 are columns - flatten result will be : [row01*row02, row01*row12, row01*row22,
                                                                        row11*row02, row11*row12, row11*row22,
                                                                        row21*row02, row21*row12, row21*row22]
              */
            flattenResult.push_back(thrust_dot(row1, row2));
        }
    }
    return deflatten_3(flattenResult);
}

// change columns to be rows
std::vector<float3> reorganize_mat3(std::vector<float3> m) {
    auto [r1x, r1y, r1z] = THRUSTtoSTDtuple(m[0]); 
    auto [r2x, r2y, r2z] = THRUSTtoSTDtuple(m[1]);
    auto [r3x, r3y, r3z] = THRUSTtoSTDtuple(m[2]);
    return { make_float3(r1x, r2x, r3x),
             make_float3(r1y, r2y, r3y),
             make_float3(r1z, r2z, r3z) };
}

// multiply two tuple matrices
auto multiply_3(std::vector<float3> m1, std::vector<float3> m2, operation status) {

    // transposed matrices - for different transpose cases - the function reorganizes the matrices
    // in order to be in right transpose format for using the multiply_3_rTranspose function
    std::vector<float3> transposed1 = reorganize_mat3(m1);
    std::vector<float3> transposed2 = reorganize_mat3(m2);
    // calculation of multiplication results using tuple dot - first case : result is all of the combinations
    // of row dot products
    return (status == RIGHT_TRANSPOSE_ONLY) ? multiply_3_rTranspose(m1, m2) :
           (status == LEFT_TRANSPOSE_ONLY) ? multiply_3_rTranspose(transposed1, transposed2) :
           (status == BOTH_TRANSPOSE) ? multiply_3_rTranspose(transposed1, m2) :
           multiply_3_rTranspose(m1, transposed2);
}

// struct made for calculating a polygon area and summing over all areas to get the surface
// area of the body.
// for triangles with vertices v1i, v2i, v3i for all i in mesh, the areas:
// s_i = 1/2|u1i x u2i| => sum_i(s_i) = sum_i(1/2|u1i x u2i|), u1i = v2i - v1i, u2i = v3i - v1i
struct polygonAreaAddition : public thrust::binary_function<thrust::tuple<particle, particle, particle>,
                                                            thrust::tuple<particle, particle, particle>,
                                                            float> {

    __host__ __device__ float operator()(const thrust::tuple<particle, particle, particle>& polygon1,
                                         const thrust::tuple<particle, particle, particle>& polygon2) {

        // get cross product in polygon1 : cross1 = u11 x u21
        auto cross1 = thrust_cross(thrust_minus(thrust::get<0>(polygon1).center, thrust::get<1>(polygon1).center),
                                   thrust_minus(thrust::get<0>(polygon1).center, thrust::get<2>(polygon1).center));
        
        // get cross product in polygon2 : cross2 = u21 x u22
        auto cross2 = thrust_cross(thrust_minus(thrust::get<0>(polygon2).center, thrust::get<1>(polygon2).center),
                                   thrust_minus(thrust::get<0>(polygon2).center, thrust::get<2>(polygon2).center));

        // return the sum of areas : 1/2 |cross1| + 1/2 |cross2| = 1/2 |u11 x u21| + 1/2 |u21 x u22| = s1 + s2
        return 0.5f * thrust_L2(cross1) + 0.5 * thrust_L2(cross2);
    }
};

// calculate ideal gas pressure that approximate the pressure on the surface of a soft body
// p = nRT/V 
float calculatePressure(float n, float R, float T, float V) {
    return n * R * T / V;
}

// calculate the magnitude of the pressure force working on each particle of
// the soft body
float calculatePressureForceMagnitude(float n, float R, float T, float V, geometricData model){
    float S = thrust_wrapper_reduce(true, 
                                    model.surfacePolygons.begin(),
                                    model.surfacePolygons.end(), 
                                    0.0f,
                                    polygonAreaAddition());
    return S * calculatePressure(n, R, T, V);
}

// adjust the spring directions to suit the correct direction of the pressure force - 
// pushing outward of the body (away from cnter mass)
struct negateDir : public thrust::unary_function<thrust::tuple<float3, particle>, float3>{

    float3 r_cm;

    // constructor for taking the center mass
    negateDir(const float3 _r_cm) : r_cm{ _r_cm } {}

    // operator for negating particle movement direction if needed
    __host__ __device__ 
    float3 operator()(thrust::tuple<float3, particle> zip_tuple) {

        // calculating r_cm - particle.center
        auto dirFromCm = thrust_minus(thrust::get<1>(zip_tuple).center, r_cm);
        auto springDir = thrust::get<0>(zip_tuple);
        if (thrust_dot(thrust::get<0>(zip_tuple), dirFromCm) < 0) {
            multiply_scalar(-1.0f, springDir);
        }
        return springDir;
    }
};

// adjust spring direction vectors according to outward direction related to center mass
void outwardDirAdjust(std::vector<float3>& springDir, std::vector<particle>& particles, float3 center_mass) {

    auto zip_iter1 = thrust::make_zip_iterator(thrust::make_tuple(springDir.begin(),
                                               particles.begin()));
    auto zip_iter2 = thrust::make_zip_iterator(thrust::make_tuple(springDir.end(),
                                               particles.end()));
    thrust_wrapper_transform(true, zip_iter1, zip_iter2, springDir.begin(), negateDir(center_mass));
}

// multiply point array by value
struct multiplyByScalar : public thrust::unary_function<float3, float3>{

    float val;

    multiplyByScalar(const float _val) : val{ _val } {}

    __host__ __device__ float3 operator()(float3 v) {
        return make_float3(v.x * val, v.y * val, v.z * val);
    }
};

// add a constant scalar
struct addConst : public thrust::unary_function<float3, float3> {

    float3 val;
    bool plus;

    addConst(const float3 _val, const bool _plus) : val{ _val }, plus{ _plus }{}

    __host__ __device__ float3 operator()(float3 v) {
        return (plus) ? thrust_plus(val, v) : thrust_minus(val, v);
    }
};

// add two arrays of points together
struct thrustAdd : public thrust::binary_function<float3, float3, float3>{

    bool plus;

    thrustAdd(bool _sign) : plus{ _sign } {}

    __host__ __device__ 
    float3 operator()(float3 v1, float3 v2) {
        return (plus) ? thrust_plus(v1, v2) : thrust_minus(v1, v2);
    }
};
                                                 

enum Dir{X, Y, Z};

struct DirExtremum : public thrust::binary_function<float3, float3, bool> {
    Dir direction;

    // constructor for initializing direction choice
    DirExtremum(const Dir dir) : direction{ dir } {}

    __host__ __device__ bool operator()(float3 p1, float3 p2) {

        if (direction == X) return p1.x > p2.x;
        else if (direction == Y) return p1.y > p2.y;
        else return p1.z > p2.z;
    }
};

std::tuple<float, float, float> findBoxDimensions(const std::vector<float3> box){
    // x dimension
    auto max_x = thrust_wrapper_max_element(true, box.begin(), box.end(), DirExtremum(X));
    auto min_x = thrust_wrapper_min_element(true, box.begin(), box.end(), DirExtremum(X));
    // y direction
    auto max_y = thrust_wrapper_max_element(true, box.begin(), box.end(), DirExtremum(Y));
    auto min_y = thrust_wrapper_min_element(true, box.begin(), box.end(), DirExtremum(Y));
    // z direction
    auto max_z = thrust_wrapper_max_element(true, box.begin(), box.end(), DirExtremum(Z));
    auto min_z = thrust_wrapper_min_element(true, box.begin(), box.end(), DirExtremum(Z));
    // return the lengths
    return std::make_tuple((*max_x).x - (*min_x).x,
                           (*max_y).y - (*min_y).y,
                           (*max_z).z - (*min_z).z);
}

// project external Force exF applied on each particle onto the spring direction spD
// this limits inner particles movement only to the spring direction.
// input - zip iterator of external force distribution and spring directions for each particle
// output - zip iterator of the data of the external force vector<float>
// the output zip iterator : zip_iterator(thrust::tuple<vec, vec + N/3, vec + 2N/3>) therefore the 
// kernel runs N times - N is the number of particles.
struct vectorProjection : public thrust::unary_function<thrust::tuple<float3, float3>,
    thrust::tuple<float, float, float>>{

    __host__ __device__
    thrust::tuple<float, float, float> operator()(thrust::tuple<float3, float3> forceDirs) {
        float3 exF = thrust::get<0>(forceDirs); float3 spD = thrust::get<1>(forceDirs);
        return thrust::make_tuple(thrust_dot(exF, spD) * spD.x / thrust_L2(spD),
            thrust_dot(exF, spD) * spD.y / thrust_L2(spD),
            thrust_dot(exF, spD) * spD.z / thrust_L2(spD));
    }
};

// r_total = r_cm + q_r*r0*q_r^-1 + dr
struct RotAndTranslate : public thrust::binary_function<particle, 
                                                        thrust::tuple<float, float, float>, particle> {

    float3 cm;
    quaternion rot;

    RotAndTranslate(const float3 _cm, const quaternion _rot) : cm{ _cm }, rot{ _rot }{}

    __host__ __device__
    particle operator()(particle currentPosition, thrust::tuple<float, float, float> fluctuations) {
        // rotate particle
        // define the particle quaternion [0, currentPoistionCenter]
        quaternion particleQuaternion(0, currentPosition.center);
        particleQuaternion = rot * particleQuaternion * rot.inverse();
        // get rotated center and translate by r_cm + dr
        float3 updatedPos = thrust_plus(cm, particleQuaternion.vector);
        // add the fluctuations
        updatedPos.x += thrust::get<0>(fluctuations);
        updatedPos.y += thrust::get<1>(fluctuations);
        updatedPos.z += thrust::get<2>(fluctuations);
        // return an updated particle
        return { updatedPos, currentPosition.radius, currentPosition.mass };
    }
};

// calculate cross between two vector arrays
struct cross_product : public thrust::unary_function<thrust::tuple<float3, particle>, float3> {
    
    __host__ __device__ float3 operator()(thrust::tuple<float3, particle> particleState) {
        return thrust_cross(thrust::get<1>(particleState).center, thrust::get<0>(particleState));
    }
};

/*
-----------------------------------------------------------
-------------------- library functions --------------------
----------------------------------------------------------- 
*/

// calculate thee direction of particle movement using the daming matrix to 
// get the total direction of spring compression
void rigid_body::initSpringDirection() {
    // systemSize = particleSize = particles.size()
    // rows
    for (int i = 0; i < 3 * systemSize; i += 3) {
        rigidState[SPRING_DIRECTION].push_back(make_float3(0.0f, 0.0f, 0.0f));
        // columns
        for (int j = 0; j < 3 * systemSize; j += 3) {
            if (DampingMatrix.data[3 * systemSize * j + i]) {
                rigidState[SPRING_DIRECTION][i] = thrust_plus(rigidState[SPRING_DIRECTION][i],
                                                              thrust_minus(particles[j].center, particles[i].center));
            }
            normalize_thrust_tuple(rigidState[SPRING_DIRECTION][i]);
        }
    }
    // adjust directions to point with positive correlation to surface normals (outward...)
    outwardDirAdjust(rigidState[SPRING_DIRECTION], particles, rigidState[CENTER_MASS][0]);
}

// the force distribution will consist of a flatten vector of triplets representing coordinates
// in 3D space (force direction). adding pressure force. using the movement directions calculated by 
// initSpringDirection() and adjusting them to point outside of the body - with positive correlation
// with the direction of r_i - r_cm , r_i center of i-th particle => directions pushing against the surface. 
void rigid_body::initForceDistribution(){
    // reset all initial forces on all particles to be gravity (uniform distribution) 
    for(auto& elem : particles){
        rigidState[FORCE_DISTRIBUTION].push_back(make_float3(0.0f, 0.0f, -elem.mass * G));
    }
}

// done after loading the model data into particles array
void rigid_body::initCenterMass(){
    // center mass is 3 size vector
    rigidState[CENTER_MASS] = { make_float3(0.0f, 0.0f, 0.0f) };
    for(auto& particle : particles){
        // uniform distribution initialization
        particle.mass = mass / (float)particles.size();
        // get particle center mass 
        auto [x, y, z] = THRUSTtoSTDtuple(particle.center);
        // same as adding coordinate/particles.size() 
        (rigidState[CENTER_MASS][0]).x +=  x * particle.mass / mass;
        (rigidState[CENTER_MASS][0]).y +=  y * particle.mass / mass;
        (rigidState[CENTER_MASS][0]).z +=  z * particle.mass / mass;
    }
}

// calculate distances in local coordinates relative to center mass
void rigid_body::initRelativeDistances() {
    for (auto& p : particles) {
        // get elem data
        auto [x, y, z] = THRUSTtoSTDtuple(p.center);
        auto [cx, cy, cz] = THRUSTtoSTDtuple(rigidState[CENTER_MASS][0]);
        relativeParticles.push_back({ make_float3(x - cx, y - cy, z - cz), p.radius, p.mass });
    }
}

// initialize linear velocity vector - of center mass
void rigid_body::initLinearVelocity() {
    rigidState[LINEAR_VELOCITY] = { make_float3(0.0f, 0.0f, 0.0f) };
}

/*
gereral rotation breaks down into multiplication of rotations on all directions:
R = R_x * R_y * R_z = Ix * Iy * Iz = I
*/
void rigid_body::initRotation(){
    rigidState[ROTATION] = { make_float3(1.0f, 0.0f, 0.0f),
                             make_float3(0.0f, 1.0f, 0.0f),
                             make_float3(0.0f, 0.0f, 1.0f) };
}

/*
void rigid_body::initLinearMomentum(){
    rigidState[LINEAR_MOMENTUM] = { std::make_tuple(0.0f, 0.0f, 0.0f) };
}
*/

// L_init = (0,0,0)
void rigid_body::initAngularMomentum(){
    rigidState[ANGULAR_MOMENTUM] = { make_float3(0.0f, 0.0f, 0.0f) };
}

void rigid_body::initTotalExternalForce() {
    rigidState[TOTAL_EXTERNAL_FORCE] = { make_float3(0.0f, 0.0f, -mass * G) };
}

// initial torque is corresponding to inital force - gravity
// TODO : change function to calculate the torque including the internal body pressure force
void rigid_body::initTorque(){
    rigidState[TORQUE] = { make_float3(0.0f, 0.0f, 0.0f) };
    for(auto& particle : relativeParticles){
        // get particle centers and force on particular particle
        auto [p_x, p_y, p_z] = THRUSTtoSTDtuple(particle.center);

        // accumulate torque elements - Ti = cross(ri, fi), 
        // in init the only force applied is gravity, hence fi = (0, 0, -G*mi).
        // since the cross product yields results perpendicular to the force, the z torque is 0
        (rigidState[TORQUE][0]).x -= p_y * G * particle.mass;
        (rigidState[TORQUE][0]).y += p_x * G * particle.mass;
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
    rigidState[INVERSE_INERTIA_TENSOR] = { make_float3(invI[0], invI[1], invI[2]),
                                           make_float3(invI[3], invI[4], invI[5]),
                                           make_float3(invI[6], invI[7], invI[8]) };
    //! get invariant body inverse inertia tensor - the inertia tensor is changes only via rotations!
    //! this is the same as the body inertia tensor in world coordinates if the initial rotation is the IDENTITY
    //! which the simulation resets to be.
    inverseBodyInertia = rigidState[INVERSE_INERTIA_TENSOR];
}

void rigid_body::initAngularVelocity() {
    rigidState[ANGULAR_VELOCITY] = { make_float3(0.0f, 0.0f, 0.0f) };
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
    initDampingMatrix();
    initCenterMass();
    initSpringDirection();
    initForceDistribution();
    calculatePressureForce();
    initRelativeDistances(); 
    initLinearVelocity();
    initRotation();
    //initLinearMomentum();
    initAngularMomentum();
    initTotalExternalForce();
    initTorque();
    initInverseInertiaTensor();
    initAngularVelocity();
    initDisplacementVector();
}

/*
------------------------------------------------------------------------------------
------------------------------ advance a step in time ------------------------------
------------------------------------------------------------------------------------
*/

// calculate updated body volume - for pressure force calculations and
// for deformation measuring - approximating with a box  
// option for accurate integration : gauss' law
void rigid_body::calculateBodyVolume() {
    auto [x, y, z] = findBoxDimensions(bodySurface.bounding_box);
    V_approx = x * y * z;
}

void rigid_body::calculatePressureForce(){
    float magnitude = calculatePressureForceMagnitude(n, R, T, V_approx, bodySurface);
    thrust::device_vector<float3> res(rigidState[SPRING_DIRECTION].begin(),
                                      rigidState[SPRING_DIRECTION].end());
    // multiply the spring directions by the pressure force magnitude
    auto first = thrust::make_transform_iterator(res.begin(), multiplyByScalar(magnitude));
    auto last = thrust::make_transform_iterator(res.end(), multiplyByScalar(magnitude));
    // add the result force distribution to force distribution
    thrust_wrapper_transform(true, first, last, rigidState[FORCE_DISTRIBUTION].begin(),
                             rigidState[FORCE_DISTRIBUTION].end(), rigidState[FORCE_DISTRIBUTION].begin(),
                             thrustAdd(true));
}

// r_cm_n+1 = r_cm_n + dt*vx_n
void rigid_body::calculateCenterMass() {
    auto [cx, cy, cz] = THRUSTtoSTDtuple(rigidState[CENTER_MASS][0]);
    auto [vx, vy, vz] = THRUSTtoSTDtuple(rigidState[LINEAR_VELOCITY][0]);
    rigidState[CENTER_MASS][0] = make_float3(cx + dt * vx, cy + dt * vy, cz + dt * vz);
}

// v_n+1 = v_n + dt*a_n = v_n + dt*f_n/m
void rigid_body::calculateLinearVelocity() {
    auto [vx, vy, vz] = THRUSTtoSTDtuple(rigidState[LINEAR_VELOCITY][0]);
    auto [fx, fy, fz] = THRUSTtoSTDtuple(rigidState[TOTAL_EXTERNAL_FORCE][0]);
    rigidState[LINEAR_VELOCITY][0] = make_float3(vx + dt * fx / mass,
                                                 vy + dt * fy / mass,
                                                 vz + dt * fz / mass);
}

/*
process:
R_n -> q_n -> q_n+1 = q_n + DT/2 w_n*q_n -> R_n+1
*/
void rigid_body::calculateRotation(){
    // represent angular velocity as a quaternion
    quaternion w(0.0f, rigidState[ANGULAR_VELOCITY][0]);
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
    auto [tx, ty, tz] = THRUSTtoSTDtuple(rigidState[TORQUE][0]);
    auto [Lx, Ly, Lz] = THRUSTtoSTDtuple(rigidState[ANGULAR_MOMENTUM][0]);
    rigidState[ANGULAR_MOMENTUM][0] = make_float3(Lx + dt * tx, Ly + dt * ty, Lz + dt * tz);
}

void rigid_body::calculateForceDistribution() {
    // for calculating with thrust - switch to thrust::tuples
}

void rigid_body::calculateTotalExternalForce() {
    rigidState[TOTAL_EXTERNAL_FORCE][0] = thrust_wrapper_reduce(true, rigidState[FORCE_DISTRIBUTION].begin(),
                                                                rigidState[FORCE_DISTRIBUTION].end(),
                                                                make_float3(0.0f, 0.0f, 0.0f),
                                                                thrustAdd(true));
}

// t = sum_i{r_i x f_i}, r_i = r_world_i - r_cm
void rigid_body::calculateTorque() {
    // zip iterator for the force acting on each particle, and particle data
    auto zip_first = thrust::make_zip_iterator(thrust::make_tuple(rigidState[FORCE_DISTRIBUTION].begin(),
                                                                  relativeParticles.begin()));
    auto zip_last = thrust::make_zip_iterator(thrust::make_tuple(rigidState[FORCE_DISTRIBUTION].end(),
                                                                 relativeParticles.end()));
    rigidState[TORQUE][0] = thrust_wrapper_reduce(true, 
                                                  thrust::make_transform_iterator(zip_first, cross_product()),
                                                  thrust::make_transform_iterator(zip_last, cross_product()),
                                                  make_float3(0.0f, 0.0f, 0.0f), thrustAdd(true));
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
    (rigidState[ANGULAR_VELOCITY][0]).x = thrust_dot(rigidState[INVERSE_INERTIA_TENSOR][0],
                                                     rigidState[ANGULAR_MOMENTUM][0]);
    (rigidState[ANGULAR_VELOCITY][0]).y = thrust_dot(rigidState[INVERSE_INERTIA_TENSOR][1],
                                                     rigidState[ANGULAR_MOMENTUM][0]);
    (rigidState[ANGULAR_VELOCITY][0]).z = thrust_dot(rigidState[INVERSE_INERTIA_TENSOR][2],
                                                     rigidState[ANGULAR_MOMENTUM][0]);
}

vector<float> rigid_body::decomposeExternalForces() {
    vector<float> externalForce(3 * systemSize, 1, memLocation::DEVICE);
    // begining operator - consisting of the begining of the external force array and spring direction
    auto zipExtForceB = thrust::make_zip_iterator(thrust::make_tuple(rigidState[FORCE_DISTRIBUTION].begin(),
                                                                     rigidState[SPRING_DIRECTION].begin()));
    // end operator
    auto zipExtForceE = thrust::make_zip_iterator(thrust::make_tuple(rigidState[FORCE_DISTRIBUTION].begin(),
                                                                     rigidState[SPRING_DIRECTION].begin()));
    // result tuple - flattened, stride is 3 - size of a single 3D point...                                           
    auto x_data = strided_iterator<float*>(externalForce.data, externalForce.data + 3 * systemSize - 2, 3);
    auto y_data = strided_iterator<float*>(externalForce.data + 1, externalForce.data + 3 * systemSize - 1, 3);
    auto z_data = strided_iterator<float*>(externalForce.data + 2, externalForce.data + 3 * systemSize, 3);
    // create a zip iterator
    auto zipExtForceR = thrust::make_zip_iterator(thrust::make_tuple(x_data.begin(),
                                                                     y_data.begin(),
                                                                     z_data.begin()));

    thrust_wrapper_transform(true, zipExtForceB, zipExtForceE, zipExtForceR, vectorProjection()); 
    return externalForce;                                                               
}

/*
equation : 
(Mh + Kd - Ks) * Q_t = Mh * (2Q_(t-dt) - Q_(t-2dt)) - N * Ftotal_t
*/
void rigid_body::updateDisplacementVector() {
    auto F = decomposeExternalForces();
    auto lhsMat = infMassDistrib + DampingDistribMatrix - DampingMatrix;
    // for sparse equation
    Sparse_mat<float> lhs_sparse(lhsMat.M, lhsMat.N, lhsMat.memState);
    lhs_sparse.copyData(lhsMat); lhs_sparse.createCSR();
    // TODO : add the calculation with the total external force
    vector<float> rhsVec = infMassDistrib % (Displacement_t_dt + Displacement_t_dt - Displacement_t_2dt) - F;
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
 - for particle positions. for relative positions:
 r_total_rel = r_total - r_cm
*/
void rigid_body::updatePosition() {
    quaternion rotation; rotation.createUnitQuarenion(flatten_3(rigidState[ROTATION]));
    // create dispacement vector iterator
    auto x_iter = strided_iterator<float*>(Displacement.data, Displacement.data + 3 * systemSize - 2, 3);
    auto y_iter = strided_iterator<float*>(Displacement.data + 1, Displacement.data + 3 * systemSize - 1, 3);
    auto z_iter = strided_iterator<float*>(Displacement.data + 2, Displacement.data + 3 * systemSize, 3);
    // iterate over all particles
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(x_iter.begin(),
                                                                  y_iter.begin(),
                                                                  z_iter.begin()));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(x_iter.end(),
                                                                y_iter.end(),
                                                                z_iter.end()));
    thrust_wrapper_transform(true, particles.begin(), 
                             particles.end(), 
                             zip_begin, zip_end,
                             particles.begin(),
                             RotAndTranslate(rigidState[CENTER_MASS][0], rotation));
    // update all relative particle positions
    thrust_wrapper_transform(true, particles.begin(),
                             particles.end(), relativeParticles.begin(),
                             addConst(rigidState[CENTER_MASS][0], false));
}

void rigid_body::advance() {
    // rigid body data
    calculateBodyVolume();
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
    updateDisplacementVector();
    // updata the body position - all of particle position changes
    // r_new = r_cm + q_r*r0*q_r^-1 - angular + linear
    // r_total = r_cm + q_r*r0*q_r^-1 + dr, dr is the inner particle pertubation made from external forces.
    updatePosition();
}