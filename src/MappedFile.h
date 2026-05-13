#pragma once

#include <memory>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream>

using std::shared_ptr;
using namespace std;

namespace Mapping{

#ifdef _WIN32
	#define NOMINMAX
	#include "windows.h"
#elif defined(__linux__)
#endif

// Usage:
//
// // Map file so that OS automatically loads accessed bytes
// auto file = Mapping::mapFile(path);
// 
// // Now load whatever data from anywhere in the file with this helper:
// file->read<uint32_t>(byteOffset);
// 
// // Or use the pointer to the mapped file data, and maybe cast it to whatever you want
// uint32_t* intArray = (uint32_t*)file->data;
//
struct MappedFile{

	string path;
	void* data = nullptr;

	#ifdef _WIN32
		HANDLE h_file;
		HANDLE h_mapping;
	#elif defined(__linux__)
		std::fstream h_file;
	#endif

	~MappedFile(){
		unmap();
	}

	void unmap(){
		#ifdef _WIN32
			if(data != nullptr){
				UnmapViewOfFile(data);
				CloseHandle(h_mapping);
				CloseHandle(h_file);
				data = nullptr;
			}
		#elif defined(__linux__)
			h_file.close();
			data = nullptr;
		#endif
	}

	template<typename T>
	T read(int64_t byteOffset) {
		uint8_t* buffer_u8 = (uint8_t*)data;

		T value;
		std::memcpy(&value, buffer_u8 + byteOffset, sizeof(T));

		return value;
	}

};

shared_ptr<MappedFile> mapFile(string path){

	shared_ptr<MappedFile> file = make_shared<MappedFile>();
	file->path = path;

	#ifdef _WIN32
		file->h_file = CreateFileA(
			path.c_str(),
			GENERIC_READ,
			FILE_SHARE_READ,
			nullptr,
			OPEN_EXISTING,
			FILE_ATTRIBUTE_NORMAL,
			nullptr
		);

		if (file->h_file == INVALID_HANDLE_VALUE) {
			println("failed to map file {}", path);
			exit(143146);
		}

		file->h_mapping = CreateFileMappingA(
			file->h_file,
			nullptr,
			PAGE_READONLY,
			0,
			0,
			nullptr
		);

		if (!file->h_mapping) {
			println("CreateFileMapping failed");
			exit(524631);
		}

		file->data = MapViewOfFile(
			file->h_mapping,
			FILE_MAP_READ,
			0,
			0,
			0
		);

		if (!file->data) {
			println("MapViewOfFile failed");
			exit(642324);
		}

	#elif defined(__linux__)
		file->h_file.open(path);
		if(!file->h_file.is_open()) {
			fprintf(stderr, "Failed to open file");
			exit(642325);
		}
		file->h_file.seekg(0, file->h_file.end);
		std::streampos size = file->h_file.tellg();
		file->data = new uint8_t[size];
		file->h_file.seekg(0, file->h_file.beg);
		file->h_file.read((char*)(file->data), size);
	#endif

	return file;

}


}