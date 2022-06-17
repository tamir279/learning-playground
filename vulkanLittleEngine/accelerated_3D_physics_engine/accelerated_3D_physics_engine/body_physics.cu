#include "body_physics.cuh"

// the force distribution will consist of a flatten vector of triplets representing coordinates
// in 3D space (force direction)
void rigid_body::initForceDistribution(){
    // reset all initial forces on all particles to be gravity (uniform distribution) 
    for(auto& elem : particles){
        rigidState[FORCE_DISTRIBUTION].push_back(std::make_tuple(0.0f, 0.0f, - elem.mass * G));
    }
}

// done after loading the model data into particles array
void rigid_body::initCenterMass(){
    // center mass is 3 size vector
    rigidState[CENTER_MASS] = {std::make_tuple(0.0f, 0.0f, 0.0f)};
    for(auto& particle : particles){
        // uniform distribution initialization
        particle.mass = mass / (float)particles.size();
        // get particle center mass 
        auto [x, y, z] = particle.center;
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
    rigidState[ROTATION] = {std::make_tuple(1.0f, 0.0f, 0.0f),
                            std::make_tuple(0.0f, 1.0f, 0.0f),
                            std::make_tuple(0.0f, 0.0f, 1.0f)};
}

// P_init = (0,0,0) 
void rigid_body::initLinearMomentum(){
    rigidState[LINEAR_MOMENTUM] = {std::make_tuple(0.0f, 0.0f, 0.0f)};
}

// L_init = (0,0,0)
void rigid_body::initAngularMomentum(){
    rigidState[ANGULAR_MOMENTUM] = {std::make_tuple(0.0f, 0.0f, 0.0f)};
}

// initial torque is corresponding to inital force - gravity
void rigid_body::initTorque(){
    rigidState[TORQUE] = {std::make_tuple(0.0f, 0.0f, 0.0f)};
    for(auto& particle : particles){
        // get particle centers and force on particular particle
        auto [p_x, p_y, p_z] = particle.center;

        // accumulate torque elements - Ti = cross(ri, fi), 
        // in init the only force applied is gravity, hence fi = (0, 0, -G*mi).
        // since the cross product yields results perpendicular to the force, the z torque is 0
        std::get<0>(rigidState[TORQUE][0]) -= p_y * G * particle.mass;
        std::get<1>(rigidState[TORQUE][0]) += p_x * G * particle.mass;
    }
}

// calculate initial inertia tensor
void rigid_body::initInverseInertiaTensor(){
    rigidState[INVERSE_INERTIA_TENSOR] = {std::make_tuple(0.0f, 0.0f, 0.0f),
                                          std::make_tuple(0.0f, 0.0f, 0.0f),
                                          std::make_tuple(0.0f, 0.0f, 0.0f)};
    // loop over all particles :
    /*
    kernel_i = m_i * (r_i^T * r_i *I - r_i * r_i^T), I = {{1,0,0},{0,1,0},{0,0,1}}, r_i particle center
    => I0 = sum(kernel_i), i >= 1, i <= numParticles
    */

    // calculate the inverse matrix
}

