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

#include <chrono>
#include <memory>
#include <vector>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "nvblox_ros/body_filter.hpp"
#include "nvblox_ros/conversions/pointcloud_conversions.hpp"

namespace nvblox
{

constexpr float kEps = 1e-4f;

sensor_msgs::msg::PointCloud2::SharedPtr makeCloud(const std::vector<Vector3f> & points)
{
  auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
  sensor_msgs::PointCloud2Modifier modifier(*cloud);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(points.size());
  sensor_msgs::PointCloud2Iterator<float> x(*cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> y(*cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> z(*cloud, "z");
  for (const Vector3f & p : points) {
    *x = p.x();
    *y = p.y();
    *z = p.z();
    ++x;
    ++y;
    ++z;
  }
  return cloud;
}

// A MID-360 scan has about 20,000 points at 10 Hz. Before the fork passed its
// stream to push_back, converting one took about 50 ms here.
TEST(PointcloudConversions, ConvertsALargeScanQuickly) {
  const Lidar lidar(360, 62, 0.3f, 20.0f, 0.1396f, 0.9250f);
  constexpr int kNumPoints = 20000;
  ASSERT_LE(kNumPoints, lidar.numel());
  // One point in each pixel's center, at depths from 1 to 4.5 m.
  std::vector<Vector3f> points;
  std::vector<float> depths;
  for (int i = 0; i < kNumPoints; ++i) {
    const Index2D u(i % lidar.cols(), i / lidar.cols());
    depths.push_back(1.0f + 0.5f * (i % 8));
    points.push_back(lidar.unprojectFromPixelIndices(u, depths.back()));
  }
  const auto cloud = makeCloud(points);

  conversions::PointcloudConverter converter;
  DepthImage range(MemoryType::kDevice);
  // The first call allocates the buffers.
  converter.depthImageFromPointcloudGPU(cloud, lidar, &range);

  constexpr int kNumRuns = 5;
  const auto start = std::chrono::steady_clock::now();
  for (int run = 0; run < kNumRuns; ++run) {
    converter.depthImageFromPointcloudGPU(cloud, lidar, &range);
  }
  const std::chrono::duration<double, std::milli> elapsed =
    std::chrono::steady_clock::now() - start;
  EXPECT_LT(elapsed.count() / kNumRuns, 5.0);

  // Every point landed in its own pixel.
  DepthImage range_host(MemoryType::kHost);
  range_host.copyFrom(range);
  for (int i = 0; i < kNumPoints; ++i) {
    ASSERT_NEAR(range_host(i / lidar.cols(), i % lidar.cols()), depths[i], kEps) << "point " << i;
  }
  // The rest of the image has no returns.
  EXPECT_EQ(range_host(lidar.rows() - 1, lidar.cols() - 1), 0.0f);
}

// The Livox driver sends missing returns at the sensor's origin, inside the
// torso's capsule. They are skipped before the body test and not counted.
TEST(PointcloudConversions, SkipsPointsNearerThanTheMinimumRange) {
  const Lidar lidar(360, 31, 0.3f, 20.0f, M_PI / 6.0);
  const Vector3f origin(0.0f, 0.0f, 0.0f);
  const Vector3f near(0.1f, 0.0f, 0.0f);
  const Vector3f torso(0.4f, 0.0f, 0.0f);
  const Vector3f wall(3.0f, 1.0f, 0.0f);
  const auto cloud = makeCloud({origin, origin, near, torso, wall});

  auto cuda_stream = std::make_shared<CudaStreamOwning>();
  const std::vector<BodyCapsule> model = {
    {"torso", {"link", Vector3f(-0.5f, 0.0f, 0.0f)}, {"link", Vector3f(0.5f, 0.0f, 0.0f)}, 0.2f},
  };
  BodyFilter body_filter(model, 0.0f, cuda_stream);
  body_filter.setLinkPoses({Transform::Identity()});

  conversions::PointcloudConverter converter;
  DepthImage range(MemoryType::kDevice);
  EXPECT_EQ(converter.depthImageFromPointcloudGPU(cloud, lidar, &range, &body_filter.capsules()),
    1);
  DepthImage range_host(MemoryType::kHost);
  range_host.copyFrom(range);
  Index2D u;
  ASSERT_TRUE(lidar.project(wall, &u));
  EXPECT_NEAR(range_host(u.y(), u.x()), lidar.getDepth(wall), kEps);
}

}  // namespace nvblox
