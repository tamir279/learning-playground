#include "body_physics.cuh"
#include "collisionAlgorithms.cuh"

/*
-------------------------------- HEAP PROPERTIES --------------------------------
*/

// reorganize a max heap with priority values
void collision_heap::heapify(int index, int n){
    int largest = index;
    int leftNode = 2 * index + 1;
    int rightNode = 2 * index + 2;

    // check if the child nodes are larger than the subtree root
    if(leftNode < n && collisionHeap[leftNode].priority > collisionHeap[largest].priority){
        largest = leftNode;
    }

    if(rightNode < n && collisionHeap[rightNode].priority > collisionHeap[largest].priority){
        largest = rightNode;
    }

    // if the largest node is not the root, swap values
    if(largest != index){
        std::swap(collisionHeap[index], collisionHeap[largest]);
        // recursivly continue to check until the heap is in correct order
        heapify(largest, n);
    }
}

void collision_heap::heapSort() {
    int size = (int)collisionHeap.size();
    std::vector<pairInfo> res(size);
    // extract all maximums and copy res data to collisionHeap at the end
    for (int i = size - 1; i >= 0; i--) {
        res[i] = popMax();
    }
    collisionHeap = res;
}

void collision_heap::insert(const pairInfo element){
    // push back the value at place n-1
    collisionHeap.push_back(element);
    // heapify to reorganize the heap
    heapify((int)collisionHeap.size() - 1, (int)collisionHeap.size());
}

// pops the maximum priority value out of the heap
pairInfo collision_heap::popMax(){
    pairInfo maxVal = collisionHeap[0];
    collisionHeap[0] = collisionHeap[(int)collisionHeap.size() - 1];
    // remove the last element
    collisionHeap.pop_back();
    // heapify to reorganize the heap
    heapify(0, (int)collisionHeap.size() - 1);
    return maxVal;
}

pairInfo collision_heap::pop(const int index){
    pairInfo val = collisionHeap[index];
    collisionHeap[index] = collisionHeap[(int)collisionHeap.size() - 1];
     // remove the last element
    collisionHeap.pop_back();
    // heapify to reorganize the heap
    heapify(index, (int)collisionHeap.size() - 1);
}

void collision_heap::update(const pairInfo element, const int index){
    collisionHeap[index] = element;
}

void collision_heap::update(const std::vector<pairInfo> updatedHeap){
    if (updatedHeap.size() != collisionHeap.size()) throw std::length_error("sizes do not match!");
    for(int i = 0; i < (int)updatedHeap.size(); i++){
        collisionHeap[i] = updatedHeap[i];
    }
}

pairInfo collision_heap::minHeap(){
    
}

pairInfo collision_heap::maxHeap(){
    return collisionHeap[0];
}

void collision_heap::buildHeap(){
    int heap_s = (int)collisionHeap.size();
    // modify each node from the first non leaf node
    for(int i = (heap_s/2) - 1; i >= 0; i--){
        heapify(i, heap_s);
    }
}

/*
-------------------------------- COLLISION FUNCTIONS --------------------------------
*/

void collision_handler::updateHeapData(std::vector<pairInfo> updatedHeap) {
    priorityHeap.update((const std::vector<pairInfo>)updatedHeap);
}

void collision_handler::initImpulseVector() {
    for (int i = 0; i < (int)bodyList.size(); i++) {
        collisionImpulse.push_back(make_float3(0.0f, 0.0f, 0.0f));
        particleIndices.push_back(-1);
    }
}

// TODO : approximate the distance between bodies better....
// IDEA : to use the bounding boxes to approximate the distance between two bodies : d = sqrt(dx^2 + dy^2 + dz^2)
float calculatePriority(float3 cm1, float3 cm2, float3 v1, float3 v2, float alpha, float beta) {

    float corr12 = colDetect::tupleDot(v2, cm2, cm1);
    float corr21 = colDetect::tupleDot(v1, cm1, cm2);
    float3 totalRelativeVelocity = make_float3(corr21 * v1.x + corr12 * v2.x,
                                               corr21 * v1.y + corr12 * v2.y,
                                               corr21 * v1.z + corr12 * v2.z);
    return alpha * 1.0f / (colDetect::tupleDot(cm2, cm1, cm2, cm1)) +
           beta * colDetect::tupleDot(totalRelativeVelocity, make_float3(0.0f, 0.0f, 0.0f), totalRelativeVelocity);
}

void collision_handler::setPriority(const std::vector<rigid_body> bodies, float alpha, float beta, EXT_pParam cm, EXT_pParam v) {
    std::vector<pairInfo> currentHeap; 
    for (int i = 0; i < (int)bodies.size(); i++) {
        for (int j = 0; j < (int)bodies.size(); i++) {
            if (i != j) {
                // center mass position
                auto cm1 = bodies[i].rigidState.find(cm);
                auto cm2 = bodies[j].rigidState.find(cm);
                // center mass velocity
                auto v1 = bodies[i].rigidState.find(v);
                auto v2 = bodies[j].rigidState.find(v);
                // get priority
                float priority = calculatePriority(cm1->second[0], cm2->second[0], v1->second[0], v2->second[0], alpha, beta);
                currentHeap.push_back({ priority, thrust::make_pair(i, j), thrust::make_pair(bodies[i].type, bodies[j].type) });
            }
        }
    }
    priorityHeap.collisionHeap = currentHeap;
    priorityHeap.buildHeap();
}

float3 collision_handler::calculateImpulse(rigid_body body1, rigid_body body2,
                                           int ind1poly, int ind1vec,
                                           int ind2poly, int ind2vec, 
                                           EXT_pParam stateParam, EXT_pParam linearV,
                                           EXT_pParam angularV) {
    std::vector<float3> K; float3 J;
    // get data about the bodies
    // get indices of affected particles - assuming all polygons are triangles
    int index1 = body1.bodySurface.indices[3 * ind1poly + ind1vec];
    int index2 = body2.bodySurface.indices[3 * ind2poly + ind2vec];
    // get particle data
    auto p1 = body1.relativeParticles[index1];
    auto p2 = body2.relativeParticles[index2];
    // get inertia tensor
    auto Iinv1 = body1.rigidState.find(stateParam);
    auto Iinv2 = body2.rigidState.find(stateParam);
    // compute K
    colReact::calculateCollisionMatrix(p1.mass, p2.mass, p1.center, p2.center, Iinv1->second, Iinv2->second, K);
    // calculate center mass velocities (angular + linear)
    const auto v1 = body1.rigidState.find(linearV); const auto v2 = body2.rigidState.find(linearV);
    const auto w1 = body1.rigidState.find(angularV); const auto w2 = body2.rigidState.find(angularV);
    // calculate velocities (tangent + radial)
    float3 u1; colReact::calculatePointSpeed(v1->second[0], w1->second[0], (const float3)p1.center, u1);
    float3 u2; colReact::calculatePointSpeed(v2->second[0], w2->second[0], (const float3)p2.center, u2);
    // calculate reaction
    colReact::calculateHeadOnCollisionReactionImpulse(0.5f * (body1.e + body2.e), (const float3)u1, (const float3)u2, K, J);
    return J;
}

// TODO : fix possible problem - what happends if 3 or more bodies are simultaniously infinitesimaly close to each other
void collision_handler::detectCollisions() {
    pairInfo maxPriority = priorityHeap.maxHeap();
    auto body1 = bodyList[thrust::get<0>(maxPriority.bodies)];
    auto body2 = bodyList[thrust::get<1>(maxPriority.bodies)];
    auto j = make_float3(0.0f, 0.0f, 0.0f);
    // calculate epsilon according to body dimensions of the top priority pair - taking an average of all dimensions
    // so epsilon will be 3 orders of magnitude smaller than the average size of a body
    auto [x1, y1, z1] = body1.bodySurface.boxDims; auto [x2, y2, z2] = body2.bodySurface.boxDims;
    epsilon = (1.0f / 6.0f) * (x1 + x2 + y1 + y2 + z1 + z2) * 10e-3;
    // detect collision
    auto [x, y, z, w] = colDetect::detectCollision(body1.bodySurface.surfacePolygons,
                                                   body2.bodySurface.surfacePolygons,
                                                   body1.bodySurface.normals,
                                                   body2.bodySurface.normals,
                                                   epsilon);
    if (x < 0 || y < 0 || z < 0 || w < 0)printf_s("no collisions have been detected");
    else {
        j = calculateImpulse(body1 ,body2, x, y, z, w, 
                             (EXT_pParam)INVERSE_INERTIA_TENSOR,
                             (EXT_pParam)LINEAR_VELOCITY,
                             (EXT_pParam)LINEAR_MOMENTUM);
    }
    // get impulses
    collisionImpulse[thrust::get<0>(maxPriority.bodies)] = j;
    collisionImpulse[thrust::get<1>(maxPriority.bodies)] = make_float3(-j.x, -j.y, -j.z);
    // get particle indices
    particleIndices[thrust::get<0>(maxPriority.bodies)] = body1.bodySurface.indices[3 * x + y];
    particleIndices[thrust::get<1>(maxPriority.bodies)] = body2.bodySurface.indices[3 * z + w];
}

void collision_handler::advanceInTime() {

    // detect collisions and calculate reaction
    detectCollisions();

    // update heap (priority)
}

// forceParam must be EXT_pParam::FORCE_DISTRIBUTION
void collision_handler::updateForces(std::vector<rigid_body>& _bodyList, EXT_pParam forceParam) {
    float dt = _bodyList[0].dt;
    for (int i = 0; i < (int)collisionImpulse.size(); i++) {
        if (particleIndices[i] > 0) {
            float3 currForce = _bodyList[i].rigidState.find(forceParam)->second[particleIndices[i]];
            _bodyList[i].rigidState.find(forceParam)->second[particleIndices[i]] = make_float3(currForce.x + collisionImpulse[i].x * dt,
                                                                                               currForce.y + collisionImpulse[i].y * dt,
                                                                                               currForce.z + collisionImpulse[i].z * dt);
        }
    }
}