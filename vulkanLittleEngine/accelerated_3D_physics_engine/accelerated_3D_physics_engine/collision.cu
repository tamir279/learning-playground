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

void collision_handler::calculateImpulse() {

}

void collision_handler::detectCollisions() {
    pairInfo maxPriority = priorityHeap.maxHeap();
    auto body1 = bodyList[thrust::get<0>(maxPriority.bodies)];
    auto body2 = bodyList[thrust::get<1>(maxPriority.bodies)];
    auto [x, y, z,w] = colDetect::detectCollision(body1.bodySurface.surfacePolygons,
                                                  body2.bodySurface.surfacePolygons,
                                                  body1.bodySurface.normals,
                                                  body2.bodySurface.normals,
                                                  epsilon);
    if (x < 0 || y < 0 || z < 0 || w < 0)printf_s("no collisions have been detected");
    else {
        calculateImpulse();
    }
}

void collision_handler::advanceInTime() {
    // calculate epsilons

    // detect collisions and calculate reaction
    detectCollisions();

    // update heap (priority)
}