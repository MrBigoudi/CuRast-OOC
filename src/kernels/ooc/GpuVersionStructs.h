#pragma once

/// A custom implementation of double linked list
template<typename T>
struct DoubleLinkedList {
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
    FirstEntry* first = new FirstEntry();
    /// A pointer to the last entry
    LastEntry* last = new LastEntry();

    ~DoubleLinkedList<T>(){
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
    }

    bool isEmpty() const {
        return first->next;
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
    }


};