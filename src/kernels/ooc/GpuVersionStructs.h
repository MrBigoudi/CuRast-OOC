#pragma once

/// A custom implementation of double linked list
template<typename T>
struct CDoubleLinkedList {
    /// An entry in the list
    struct Iterator {
        T value;
        Iterator* next = nullptr;
        Iterator* prev = nullptr;
    };

    struct FirstEntry {
        Iterator* next = nullptr;
    };
    struct LastEntry {
        Iterator* prev = nullptr;
    };

    /// A pointer to the first entry
    FirstEntry* first = nullptr;
    /// A pointer to the last entry
    LastEntry* last = nullptr;
    uint32_t size = 0;

    void init(){
        first = new FirstEntry();
        last = new LastEntry();
    }

    ~CDoubleLinkedList(){
        if(!first){return;}
        clear();
        delete(first);
        delete(last);
    }

    void clear() {
        Iterator* cur_entry = first->next;
        while(cur_entry){
            Iterator* next_entry = cur_entry->next;
            delete(cur_entry);
            cur_entry = next_entry;
        }
        first->next = nullptr;
        last->prev = nullptr;
        size = 0;
    }

    bool isEmpty() const {
        return size == 0;
    }

    void pushFront(T new_value) {
        Iterator* new_entry = new Iterator();
        new_entry->value = new_value;
        
        // Check if the list was empty
        if(isEmpty()){
            first->next = new_entry;
            last->prev = new_entry;
        } else {
            new_entry->next = first->next;
            first->next->prev = new_entry;
            first->next = new_entry;
        }
        size++;
    };


    void pushBack(T new_value) {
        Iterator* new_entry = new Iterator();
        new_entry->value = new_value;
        
        // Check if the list was empty
        if(isEmpty()){
            first->next = new_entry;
            last->prev = new_entry;
        } else {
            new_entry->prev = last->prev;
            last->prev->next = new_entry;
            last->prev = new_entry;
        }
        size++;
    }


    T* back() {
        return last->prev ? &last->prev->value : nullptr;
    }
    T* front() {
        return first->next ? &first->next->value : nullptr;
    }
    Iterator* end() {
        return last->prev;
    }
    Iterator* begin() {
        return first->next;
    }

    void erase(Iterator* it) {
        if(!it){return;}
        Iterator* prev = it->prev;
        Iterator* next = it->next;
        if(prev){prev->next = next;} else {first->next = next;}
        if(next){next->prev = prev;} else {last->prev = prev;}
        size--;
    }
    void erase(T value){
        Iterator* it = first->next;
        while(it){
            if(it->value == value){
                break;
            }
            it = it->next;
        }
        erase(it);
    }


    void popBack() {
        if(isEmpty()){return;}
        if(first->next == last->prev){
            delete(first->next);
            first->next = nullptr;
            last->prev = nullptr;
            return;
        }
        Iterator* old_last = last->prev;
        last->prev = old_last->prev;
        old_last->prev->next = nullptr;
        delete(old_last);
        size--;
    }

    void popFront() {
        if(isEmpty()){return;}
        if(first->next == last->prev){
            delete(first->next);
            first->next = nullptr;
            last->prev = nullptr;
            return;
        }
        Iterator* old_first = first->next;
        first->next = old_first->next;
        old_first->next->prev = nullptr;
        delete(old_first);
        size--;
    }


};




/// A custom implementation of hash maps
template<typename Key, typename T>
struct CHashMap {
    uint64_t size = 42;
    uint64_t capacity = 12;

    /// The array of elements
    T* elements = nullptr;
    /// The corresponding keys
    uint64_t* keys = nullptr;

    /// How often we probe for an empty slot before giving up
	constexpr static uint32_t MAX_ATTEMPTS = 10;
    constexpr static uint64_t INVALID_KEY = 0xffffffffffffffff;
    constexpr static uint64_t SEED = 2915580697;

    /// Murmur originates from here: https://github.com/aappleby/smhasher
	/// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash2.cpp
	static uint64_t hash_murmur(Key key) {
		const uint64_t m = 0xc6a4a7935bd1e995;
        const int r = 47;

        uint32_t len = sizeof(Key);

        uint64_t h = SEED ^ (len * m);

        const uint64_t* data = (const uint64_t*)key;
        const uint64_t* end = data + (len/8);

        while(data != end){
            uint64_t k = *data++;

            k *= m; 
            k ^= k >> r; 
            k *= m; 
            
            h ^= k;
            h *= m; 
        }

        const unsigned char * data2 = (const unsigned char*)data;

        switch(len & 7)
        {
        case 7: h ^= uint64_t(data2[6]) << 48;
        case 6: h ^= uint64_t(data2[5]) << 40;
        case 5: h ^= uint64_t(data2[4]) << 32;
        case 4: h ^= uint64_t(data2[3]) << 24;
        case 3: h ^= uint64_t(data2[2]) << 16;
        case 2: h ^= uint64_t(data2[1]) << 8;
        case 1: h ^= uint64_t(data2[0]);
                h *= m;
        };
        
        h ^= h >> r;
        h *= m;
        h ^= h >> r;

        return h;
	}

    /// Create a map of a given size
    void init(uint64_t new_size){
        printf("this      %p\n", this);
        printf("&capacity %p\n", &capacity);
        printf("&size %p\n", &size);
        capacity = new_size;
        printf("after = %llu\n", capacity);
        size = new_size;

        // elements = (T*)malloc(new_size * sizeof(T));
        // keys = (uint64_t*)malloc(new_size * sizeof(uint64_t));
        // for(uint32_t i=0; i<capacity; i++){
        //     keys[i] = INVALID_KEY;
        // }
    }

    ~CHashMap(){
        free(elements);
        free(keys);
    }

    /// Crash if couldn't find empty spot
    /// Replace the value if the key is already in the map
    void insert_or_replace(Key key, T value){
        uint64_t hash = hash_murmur(key) % capacity;

        uint64_t first_free_index = INVALID_KEY;
        for(uint32_t nb_tries=0; nb_tries < MAX_ATTEMPTS; nb_tries++){
            uint64_t index = (hash + nb_tries) % capacity;
            uint64_t old_key = keys[index];

            if(old_key == INVALID_KEY && first_free_index == INVALID_KEY){ // Check if found empty space
                first_free_index = index;
            } else if(old_key == hash){ // Check if found existing key
                elements[index] = value;
                return;
            }
        }
        if(first_free_index != INVALID_KEY){
            elements[first_free_index] = value;
            keys[first_free_index] = hash;
            size++;
        }

        printf("ERROR: can't insert new element in map");
        // __trap();
    }

    bool contains(Key key){
        uint64_t hash = hash_murmur(key) % capacity;
        for(uint32_t nb_tries=0; nb_tries < MAX_ATTEMPTS; nb_tries++){
            uint64_t index = (hash + nb_tries) % capacity;
            uint64_t old_key = keys[index];
            if(keys[index] == hash){return true;}
        }
        return false;
    }

    /// Crash if couldn't find the key
    T find(Key key){
        uint64_t hash = hash_murmur(key) % capacity;
        for(uint32_t nb_tries=0; nb_tries < MAX_ATTEMPTS; nb_tries++){
            uint64_t index = (hash + nb_tries) % capacity;
            uint64_t old_key = keys[index];

            if(keys[index] == hash){
                return elements[index];
            }
        }

        printf("ERROR: can't find element in map");
        // __trap();
    }

    void erase(Key key){
        uint64_t hash = hash_murmur(key) % capacity;
        for(uint32_t nb_tries=0; nb_tries < MAX_ATTEMPTS; nb_tries++){
            uint64_t index = (hash + nb_tries) % capacity;
            uint64_t old_key = keys[index];

            if(keys[index] == hash){
                keys[index] = INVALID_KEY;
                size--;
                return;
            }
        }
    }

    void clear(){
        for(uint32_t i=0; i<capacity; i++){
            keys[i] = INVALID_KEY;
        }
        size = 0;
    }


    T& operator[](Key key){
        uint64_t hash = hash_murmur(key) % capacity;
        uint64_t first_free_index = INVALID_KEY;
        for(uint32_t nb_tries=0; nb_tries < MAX_ATTEMPTS; nb_tries++){
            uint64_t index = (hash + nb_tries) % capacity;
            uint64_t old_key = keys[index];
            if(old_key == INVALID_KEY && first_free_index == INVALID_KEY){ // Check if found empty space
                first_free_index = index;
            } else if(old_key == hash){ // Check if found existing key
                return elements[index];
            }
        }

        if(first_free_index != INVALID_KEY){
            elements[first_free_index] = T();
            keys[first_free_index] = hash;
            size++;
            return elements[first_free_index];
        }

        printf("ERROR: can't find or insert element in map");
        // __trap();
    }

    const T operator[](Key key) const {
        uint64_t hash = hash_murmur(key) % capacity;
        uint64_t first_free_index = INVALID_KEY;
        for(uint32_t nb_tries=0; nb_tries < MAX_ATTEMPTS; nb_tries++){
            uint64_t index = (hash + nb_tries) % capacity;
            uint64_t old_key = keys[index];
            if(old_key == INVALID_KEY && first_free_index == INVALID_KEY){ // Check if found empty space
                first_free_index = index;
            } else if(old_key == hash){ // Check if found existing key
                return elements[index];
            }
        }

        printf("ERROR: can't get element in map");
        // __trap();
    }
};