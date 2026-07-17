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



/// From claude
/// Minimal trait, no <type_traits> dependency (NVRTC-safe)
template<typename U> struct CIsPointer       { static constexpr bool value = false; };
template<typename U> struct CIsPointer<U*>   { static constexpr bool value = true;  };




/// A custom implementation of hash maps
template<typename Key, typename T>
struct CHashMap {
    uint64_t capacity = 0;
    uint64_t size = 0;

    /// The array of elements
    struct Entry {
        T element;
        uint64_t key = INVALID_KEY;
    };
    CDoubleLinkedList<Entry>* elements = nullptr;

    /// How often we probe for an empty slot before giving up
    constexpr static uint64_t INVALID_KEY = 0xffffffffffffffff;
    constexpr static uint64_t SEED = 2915580697;

    /// Murmur originates from here: https://github.com/aappleby/smhasher
    /// Here is a version proposed by claude
	/// 64-bit avalanche mix (Murmur3 finalizer) — same core for every branch below
    static uint64_t mix64(uint64_t h) {
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return h;
    }

    static uint64_t hash_murmur(const Key& key) {
        if constexpr (CIsPointer<Key>::value) {
            // Hash the pointer's address value directly
            return mix64((uint64_t)(uintptr_t)key ^ SEED);
        } else if constexpr (sizeof(Key) <= 8) {
            // Small integral/enum/etc: widen and hash the value itself
            uint64_t v = 0;
            __builtin_memcpy(&v, &key, sizeof(Key)); // avoids sign/UB issues from a raw cast
            return mix64(v ^ SEED);
        } else {
            // General fallback: hash the raw bytes of the object itself
            // NOTE: byte-hashes &key (the object's storage), never `key` reinterpreted as a pointer
            const uint64_t m = 0xc6a4a7935bd1e995ULL;
            const int r = 47;
            uint32_t len = sizeof(Key);
            uint64_t h = SEED ^ (len * m);

            const uint64_t* data = reinterpret_cast<const uint64_t*>(&key);
            const uint64_t* end  = data + (len / 8);

            while (data != end) {
                uint64_t k = *data++;
                k *= m; k ^= k >> r; k *= m;
                h ^= k; h *= m;
            }

            const unsigned char* data2 = reinterpret_cast<const unsigned char*>(data);
            switch (len & 7) {
                case 7: h ^= uint64_t(data2[6]) << 48;
                case 6: h ^= uint64_t(data2[5]) << 40;
                case 5: h ^= uint64_t(data2[4]) << 32;
                case 4: h ^= uint64_t(data2[3]) << 24;
                case 3: h ^= uint64_t(data2[2]) << 16;
                case 2: h ^= uint64_t(data2[1]) << 8;
                case 1: h ^= uint64_t(data2[0]);
                        h *= m;
            }
            h ^= h >> r; h *= m; h ^= h >> r;
            return h;
        }
    }

    /// Create a map of a given size
    void init(uint64_t new_size){
        capacity = new_size;
        elements = (CDoubleLinkedList<Entry>*)malloc(capacity * sizeof(CDoubleLinkedList<Entry>));
        for(uint64_t i=0; i<capacity; i++){
            elements[i].init();
        }
    }

    ~CHashMap(){
        free(elements);
    }

    CDoubleLinkedList<Entry>& get_list(uint64_t murmur) {
        uint64_t hash = murmur % capacity;
        return elements[hash];
    }

    CDoubleLinkedList<Entry>::Iterator* get_iterator(Key& key){
        return get_iterator(hash_murmur(key));
    }

    CDoubleLinkedList<Entry>::Iterator* get_iterator(uint64_t murmur){
        uint64_t hash = murmur % capacity;
        CDoubleLinkedList<Entry>& elems_list = elements[hash];
        auto list_it = elems_list.begin();
        while(list_it){
            if(list_it->value.key == murmur){
                return list_it;
            }
            list_it = list_it->next;
        }
        return nullptr;
    }

    /// Replace the value if the key is already in the map
    void insert_or_replace(Key key, T new_value){
        uint64_t murmur = hash_murmur(key);
        CDoubleLinkedList<Entry>& elems_list = get_list(murmur);
        typename CDoubleLinkedList<Entry>::Iterator* list_it = get_iterator(murmur);
        if(list_it){
            // Replace if already in the list
            list_it->value.element = new_value;
        } else {
            // Add if not already in the list
            elems_list.pushFront({new_value, murmur});
            size++;
        }
    }

    /// Check if the map already contains the key
    bool contains(Key key){
        typename CDoubleLinkedList<Entry>::Iterator* list_it = get_iterator(key);
        return list_it != nullptr;
    }

    /// Crash if couldn't find the key
    T find(Key key){
        typename CDoubleLinkedList<Entry>::Iterator* list_it = get_iterator(key);
        if(list_it){return list_it->value.element;}
        printf("ERROR: can't find element in map\n");
        // __trap();
    }

    void erase(Key key){
        uint64_t murmur = hash_murmur(key);
        CDoubleLinkedList<Entry>& elems_list = get_list(murmur);
        typename CDoubleLinkedList<Entry>::Iterator* list_it = get_iterator(murmur);
        if(list_it){
            elems_list.erase(list_it);
            size--;
        }
    }

    void clear(){
        for(uint32_t i=0; i<capacity; i++){
            elements[i].clear();
        }
        size = 0;
    }

    /// Create an empty value for the given key if not already in the map
    T& operator[](Key key){
        uint64_t murmur = hash_murmur(key);
        CDoubleLinkedList<Entry>& elems_list = get_list(murmur);
        typename CDoubleLinkedList<Entry>::Iterator* list_it = get_iterator(murmur);
        if(list_it){
            // Return if already in the list
            return list_it->value.element;
        } else {
            // Add if not already in the list
            elems_list.pushFront({T(), murmur});
            size++;
        }
    }

    const T operator[](Key key) const {
        typename CDoubleLinkedList<Entry>::Iterator* list_it = get_iterator(key);
        if(list_it){return list_it->value.element;}
        printf("ERROR: can't get element in map\n");
        // __trap();
    }
};