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
    bool initialised = false;

    void init(){
        first = new FirstEntry();
        last = new LastEntry();
        size = 0;
        initialised = true;
    }

    CDoubleLinkedList(){}

    CDoubleLinkedList(const CDoubleLinkedList& cpy){
        init();
        Iterator* it = cpy.begin();
        while(it){
            pushBack(it->value);
            it = it->next;
        }
    }

    ~CDoubleLinkedList(){
        if(!initialised){return;}
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

    bool sanityCheck() const {
        Iterator* it = first->next;
        uint32_t cpt_left = 0;
        while(it){
            it = it->next;
            cpt_left++;
        }
        uint32_t cpt_right = 0;
        it = last->prev;
        while(it){
            it = it->prev;
            cpt_right++;
        }

        return (cpt_left == cpt_right) && (cpt_left == size);
    }

    /// Expect the iterator to already be a part of the list
    void moveBegin(Iterator* it) {
        Iterator* prev = it->prev;
        Iterator* next = it->next;
        if(prev){
            // Update old neighbours
            prev->next = next;
            if(next){
                next->prev = prev;
            } else {
                last->prev = prev;
            }
            // Update new neighbours
            Iterator* old_front = first->next;
            it->next = first->next;
            it->prev = nullptr;
            first->next = it;
            old_front->prev = it;
        }
    };

    /// Moves the given iterator and all it's next iterations at the beginning of the list
    /// The given iterator will become the first element of the list followed by its previous next iterations
    /// The elements that was before this iterator becomes the last element of the list
    /// Expect the iterator to already be a part of the list
    void moveBeginWithNexts(Iterator* it){
        Iterator* prev = it->prev;
        Iterator* next = it->next;
        if(prev){
            Iterator* old_front = first->next;
            Iterator* old_last = last->prev;
            old_last->next = old_front;
            old_front->prev = old_last;

            prev->next = nullptr;
            last->prev = prev;

            it->prev = nullptr;
            first->next = it;
        }
    }


    /// Expect the iterator to already be a part of the list
    void moveEnd(Iterator* it) {
        Iterator* prev = it->prev;
        Iterator* next = it->next;
        if(next){
            // Update old neighbours
            next->prev = prev;
            if(prev){
                prev->next = next;
            } else {
                first->next = next;
            }
            // Update new neighbours
            Iterator* old_back = last->prev;
            it->prev = last->prev;
            it->next = nullptr;
            last->prev = it;
            old_back->next = it;
        }
    };

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
        delete(it);
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
            size--;
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
            size--;
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
    bool initialised = false;

    /// The array of elements
    struct Entry {
        T element;
        Key real_key;
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

    static uint64_t hashMurmur(const Key& key) {
        if constexpr (CIsPointer<Key>::value) {
            // Hash the pointer's address value directly
            return mix64((uint64_t)(uintptr_t)key ^ SEED);
        } else if constexpr (sizeof(Key) <= 8) {
            // Small integral/enum/etc: widen and hash the value itself
            uint64_t v = 0;
            const unsigned char* src = reinterpret_cast<const unsigned char*>(&key);
            for (uint32_t i = 0; i < sizeof(Key); ++i) {
                v |= (uint64_t)src[i] << (8 * i);
            }
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
        initialised = true;
        capacity = new_size;
        elements = (CDoubleLinkedList<Entry>*)malloc(capacity * sizeof(CDoubleLinkedList<Entry>));
        for(uint64_t i=0; i<capacity; i++){
            elements[i].init();
        }
    }

    CHashMap(){}

    CHashMap(const CHashMap& cpy){
        init(cpy.capacity);
        for(uint64_t i=0; i<capacity; i++){
            elements[i] = cpy.elements[i];
        }
        size = cpy.size;
    }

    ~CHashMap(){
        if(initialised){
            free(elements);
        }
    }

    CDoubleLinkedList<Entry>& getList(uint64_t murmur) {
        uint64_t hash = murmur % capacity;
        return elements[hash];
    }

    CDoubleLinkedList<Entry>::Iterator* getIterator(Key& key){
        return getIterator(hashMurmur(key));
    }

    CDoubleLinkedList<Entry>::Iterator* getIterator(uint64_t murmur){
        uint64_t hash = murmur % capacity;
        CDoubleLinkedList<Entry>& elems_list = elements[hash];

        // If the list doesn't exist yet, create it
        if(!elems_list.first){
            elems_list.init();
            return nullptr;
        }

        auto list_it = elems_list.begin();
        while(list_it){
            if(list_it->value.key == murmur){
                return list_it;
            }
            if(list_it->value.key == INVALID_KEY){
                return nullptr;
            }
            list_it = list_it->next;
        }
        return nullptr;
    }

    /// Replace the value if the key is already in the map
    void insertOrReplace(Key key, T new_value){
        uint64_t murmur = hashMurmur(key);
        CDoubleLinkedList<Entry>& elems_list = getList(murmur);
        typename CDoubleLinkedList<Entry>::Iterator* list_it = getIterator(murmur);
        if(list_it){
            // Replace if already in the list
            list_it->value.element = new_value;
        } else {
            // Add if not already in the list
            // Check if some previously deleted elements are available
            typename CDoubleLinkedList<Entry>::Iterator* end = elems_list.end();
            if(end && end->value.key == INVALID_KEY){ // If so, replace it
                end->value = {new_value, key, murmur};
                elems_list.moveBegin(end);
            } else { // Else, insert a brand new element
                elems_list.pushFront({new_value, key, murmur});
            }
            size++;
        }
    }

    /// Check if the map already contains the key
    bool contains(Key key){
        typename CDoubleLinkedList<Entry>::Iterator* list_it = getIterator(key);
        return list_it != nullptr;
    }

    /// Return nullptr if couldn't find the key
    T* find(Key key){
        typename CDoubleLinkedList<Entry>::Iterator* list_it = getIterator(key);
        if(list_it){return &list_it->value.element;}
        return nullptr;
    }

    bool partialErase(Key key){
        uint64_t murmur = hashMurmur(key);
        CDoubleLinkedList<Entry>& elems_list = getList(murmur);
        typename CDoubleLinkedList<Entry>::Iterator* list_it = getIterator(murmur);
        if(list_it){
            // Flag the element as wrong and put it in the back of the list
            list_it->value.key = INVALID_KEY;
            elems_list.moveEnd(list_it);
            size--;
            return true;
        }
        return false;
    }

    bool erase(Key key){
        uint64_t murmur = hashMurmur(key);
        CDoubleLinkedList<Entry>& elems_list = getList(murmur);
        typename CDoubleLinkedList<Entry>::Iterator* list_it = getIterator(murmur);
        if(list_it){
            elems_list.erase(list_it);
            size--;
            return true;
        }
        return false;
    }

    void clear(){
        for(uint32_t i=0; i<capacity; i++){
            elements[i].clear();
        }
        size = 0;
    }

    /// Create an empty value for the given key if not already in the map
    T& operator[](Key key){
        uint64_t murmur = hashMurmur(key);
        CDoubleLinkedList<Entry>& elems_list = getList(murmur);
        typename CDoubleLinkedList<Entry>::Iterator* list_it = getIterator(murmur);
        if(list_it){
            // Return if already in the list
            return list_it->value.element;
        } else {
            // Add if not already in the list
            // Check if some previously deleted elements are available
            typename CDoubleLinkedList<Entry>::Iterator* end = elems_list.end();
            if(end && end->value.key == INVALID_KEY){ // If so, replace it
                end->value = {T(), key, murmur};
                elems_list.moveBegin(end);
            } else { // Else, insert a brand new element
                elems_list.pushFront({T(), key, murmur});
            }
            size++;
            return elems_list.front()->element;
        }
    }

    const T& operator[](Key key) const {
        return (*this)[key];
    }

    template<typename Func>
    void map(Func f){
        for(uint32_t i=0; i<capacity; i++){
            typename CDoubleLinkedList<Entry>::Iterator* it = elements[i].begin();
            while(it){
                f(it->value.element);
                it = it->next;
            }
        }
    }

    template<typename Func>
    void mapWithKey(Func f){
        for(uint32_t i=0; i<capacity; i++){
            typename CDoubleLinkedList<Entry>::Iterator* it = elements[i].begin();
            while(it){
                f(it->value.real_key, it->value.element);
                it = it->next;
            }
        }
    }
};