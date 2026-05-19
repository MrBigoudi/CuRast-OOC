#pragma once

#include "CuRast.h"
#include "laszip/laszip_api.h"

/// The maximum number of points in a batch
constexpr uint64_t MAX_BATCH_SIZE = 1'000'000;

/// A loaded point
struct Point {
	vec3 position = vec3();
	uint8_t color[4] = {0,0,0,1};
};

/// A set of points read from a las / laz file
struct PointBatch {
	uint64_t first = 0;
	uint64_t count = 0;
	shared_ptr<string> file = nullptr;
	shared_ptr<vector<Point>> points = nullptr;
	shared_ptr<laszip_header> header = nullptr;
	vector<vec3> getPositions() const;
	vector<uint32_t> getColors() const;
};