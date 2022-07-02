#include "body_physics.cuh"


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
    setSamplingRates();
}

void collision_handler::setSamplingRates() {
    priorityHeap.heapSort();
    for (auto& elem : priorityHeap.collisionHeap) {
        elem.samplePeriod++;
    }
}