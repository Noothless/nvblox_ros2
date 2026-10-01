// Copyright (c) 2026 The agentic-humanoid authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <nvblox/mapper/mapper.h>
#include <nvblox/sensors/lidar.h>

#include "nvblox_ros/mapper_initialization.hpp"

namespace nvblox
{

constexpr float kVoxelSizeM = 0.05f;
constexpr float kBlockSizeM = kVoxelSizeM * TsdfBlock::kVoxelsPerSide;
// The robot's bound: just below a block boundary, so the map stops at z = 0.
constexpr float kMaxHeightM = -0.01f;

std::unique_ptr<Mapper> makeBoundedMapper()
{
  auto mapper = std::make_unique<Mapper>(kVoxelSizeM);
  MapperParams params;
  params.view_calculator_params.workspace_bounds_type = WorkspaceBoundsType::kHeightBounds;
  params.view_calculator_params.workspace_bounds_min_height_m = -2.0f;
  params.view_calculator_params.workspace_bounds_max_height_m = kMaxHeightM;
  mapper->setMapperParams(params);
  return mapper;
}

// A scan with a return 2 m away in every pixel, from a LiDAR at the origin
// that sees up to 53 degrees above horizontal: most returns are above z = 0.
int numBlocksAboveBound(Mapper * mapper)
{
  const Lidar lidar(360, 62, 0.3f, 20.0f, 0.1396f, 0.9250f);
  DepthImage range(lidar.rows(), lidar.cols(), MemoryType::kUnified);
  for (int row = 0; row < lidar.rows(); ++row) {
    for (int col = 0; col < lidar.cols(); ++col) {
      range(row, col) = 2.0f;
    }
  }
  mapper->integrateLidarDepth(range, Transform::Identity(), lidar);
  int num_above = 0;
  for (const Index3D & block : mapper->tsdf_layer().getAllBlockIndices()) {
    if (block.z() * kBlockSizeM > kMaxHeightM) {
      ++num_above;
    }
  }
  return num_above;
}

TEST(LidarWorkspaceBounds, MapperLeavesTheLidarUnbounded) {
  auto mapper = makeBoundedMapper();
  EXPECT_GT(numBlocksAboveBound(mapper.get()), 0);
}

TEST(LidarWorkspaceBounds, LidarMapsNothingAboveTheBound) {
  auto mapper = makeBoundedMapper();
  applyWorkspaceBoundsToLidar(mapper.get());
  const ViewCalculator & lidar_view = mapper->lidar_tsdf_integrator().view_calculator();
  EXPECT_EQ(lidar_view.workspace_bounds_type(), WorkspaceBoundsType::kHeightBounds);
  EXPECT_FLOAT_EQ(lidar_view.workspace_bounds_max_corner_m().z(), kMaxHeightM);
  EXPECT_EQ(numBlocksAboveBound(mapper.get()), 0);
  EXPECT_FALSE(mapper->tsdf_layer().getAllBlockIndices().empty());
}

}  // namespace nvblox
