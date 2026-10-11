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

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "nvblox_ros/body_filter.hpp"
#include "nvblox_ros/conversions/pointcloud_conversions.hpp"
#include "nvblox_ros/held_objects.hpp"

namespace nvblox
{

constexpr float kEps = 1e-4f;

Transform rotationAboutZ(const float angle_rad, const Vector3f & translation)
{
  Transform T = Transform::Identity();
  T.linear() = Eigen::AngleAxisf(angle_rad, Vector3f::UnitZ()).toRotationMatrix();
  T.translation() = translation;
  return T;
}

TEST(BodyFilter, PointInCapsuleAlongAnAxis) {
  // Along y, so the boundary values are exact in float.
  const Capsule capsule{Vector3f(0.0f, 0.0f, 0.0f), Vector3f(0.0f, 2.0f, 0.0f), 0.25f};
  EXPECT_TRUE(isPointInCapsule(Vector3f(0.0f, 1.0f, 0.0f), capsule));
  EXPECT_TRUE(isPointInCapsule(Vector3f(0.1f, 0.5f, -0.1f), capsule));
  // On the cylinder and on the end caps.
  EXPECT_TRUE(isPointInCapsule(Vector3f(0.25f, 1.0f, 0.0f), capsule));
  EXPECT_TRUE(isPointInCapsule(Vector3f(0.0f, 2.25f, 0.0f), capsule));
  EXPECT_TRUE(isPointInCapsule(Vector3f(0.0f, -0.25f, 0.0f), capsule));
  // Just outside them.
  EXPECT_FALSE(isPointInCapsule(Vector3f(0.25f + kEps, 1.0f, 0.0f), capsule));
  EXPECT_FALSE(isPointInCapsule(Vector3f(0.0f, 2.25f + kEps, 0.0f), capsule));
  EXPECT_FALSE(isPointInCapsule(Vector3f(0.0f, -0.25f - kEps, 0.0f), capsule));
  // Next to the segment's line but beyond its end: only the cap counts.
  EXPECT_FALSE(isPointInCapsule(Vector3f(0.2f, 2.2f, 0.0f), capsule));
}

TEST(BodyFilter, PointInRotatedCapsule) {
  // A 1 m capsule along the link's x axis, the link turned 30 degrees about z
  // and tilted 45 degrees about x.
  Transform T = rotationAboutZ(M_PI / 6.0, Vector3f(0.3f, -0.2f, 1.0f));
  T.linear() = T.linear() * Eigen::AngleAxisf(M_PI / 4.0, Vector3f::UnitX()).toRotationMatrix();
  const Vector3f start = T * Vector3f(0.0f, 0.0f, 0.0f);
  const Vector3f end = T * Vector3f(1.0f, 0.0f, 0.0f);
  const Capsule capsule{start, end, 0.1f};
  const Vector3f side = T.linear() * Vector3f(0.0f, 0.0f, 1.0f);
  const Vector3f axis = T.linear() * Vector3f(1.0f, 0.0f, 0.0f);

  EXPECT_TRUE(isPointInCapsule(start + 0.5f * axis, capsule));
  EXPECT_TRUE(isPointInCapsule(start + 0.5f * axis + (0.1f - kEps) * side, capsule));
  EXPECT_FALSE(isPointInCapsule(start + 0.5f * axis + (0.1f + kEps) * side, capsule));
  EXPECT_TRUE(isPointInCapsule(end + (0.1f - kEps) * axis, capsule));
  EXPECT_FALSE(isPointInCapsule(end + (0.1f + kEps) * axis, capsule));
  // The same offset in the unrotated axes is outside.
  EXPECT_FALSE(isPointInCapsule(start + Vector3f(0.5f, 0.0f, 0.0f), capsule));
}

TEST(BodyFilter, DegenerateCapsuleIsASphere) {
  const Capsule sphere{Vector3f(1.0f, 1.0f, 1.0f), Vector3f(1.0f, 1.0f, 1.0f), 0.5f};
  EXPECT_TRUE(isPointInCapsule(Vector3f(1.0f, 1.5f, 1.0f), sphere));
  EXPECT_FALSE(isPointInCapsule(Vector3f(1.0f, 1.5f + kEps, 1.0f), sphere));
}

TEST(BodyFilter, HoldsAsManyCapsulesAsAKernelArgumentTakes) {
  // More than kMaxBodyCapsules fail a CHECK in the constructor (not tested
  // here: a death test forks, and CUDA does not work in the child).
  const std::vector<BodyCapsule> model(kMaxBodyCapsules,
    BodyCapsule{"c", {"link", Vector3f::Zero()}, {"link", Vector3f::UnitX()}, 0.1f});
  auto cuda_stream = std::make_shared<CudaStreamOwning>();
  BodyFilter body_filter(model, 0.0f, cuda_stream);
  body_filter.setLinkPoses({Transform::Identity()});
  EXPECT_EQ(body_filter.capsules().num_capsules, kMaxBodyCapsules);
}

TEST(BodyFilter, NonFinitePointsAreNeverInside) {
  const Capsule capsule{Vector3f(0.0f, 0.0f, 0.0f), Vector3f(1.0f, 0.0f, 0.0f), 10.0f};
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(isPointInAnyCapsule(Vector3f(inf, 0.0f, 0.0f), &capsule, 1));
  EXPECT_FALSE(isPointInAnyCapsule(Vector3f(0.0f, nan, 0.0f), &capsule, 1));
  EXPECT_TRUE(isPointInAnyCapsule(Vector3f(0.5f, 0.0f, 0.0f), &capsule, 1));
}

TEST(BodyFilter, PlacesCapsulesFromLinkPoses) {
  const std::vector<BodyCapsule> model = {
    {"forearm", {"elbow", Vector3f(0.0f, 0.0f, 0.0f)}, {"wrist", Vector3f(0.1f, 0.0f, 0.0f)},
      0.05f},
    {"upper_arm", {"shoulder", Vector3f(0.0f, 0.0f, 0.0f)}, {"elbow", Vector3f::Zero()}, 0.06f},
  };
  const std::vector<std::string> links = bodyLinks(model);
  ASSERT_EQ(links, (std::vector<std::string>{"elbow", "wrist", "shoulder"}));

  // The wrist is turned 90 degrees about z: its x offset points along the sensor's y.
  const std::vector<Transform> T_S_links = {
    rotationAboutZ(0.0f, Vector3f(1.0f, 0.0f, 0.0f)),
    rotationAboutZ(M_PI / 2.0, Vector3f(1.0f, 0.3f, 0.0f)),
    rotationAboutZ(0.0f, Vector3f(0.0f, 0.0f, 0.5f)),
  };
  const std::vector<Capsule> capsules = placeCapsules(model, links, T_S_links, 0.02f);
  ASSERT_EQ(capsules.size(), 2u);
  EXPECT_TRUE(capsules[0].start.isApprox(Vector3f(1.0f, 0.0f, 0.0f), kEps));
  EXPECT_TRUE(capsules[0].end.isApprox(Vector3f(1.0f, 0.4f, 0.0f), kEps));
  EXPECT_NEAR(capsules[0].radius, 0.07f, kEps);
  EXPECT_TRUE(capsules[1].start.isApprox(Vector3f(0.0f, 0.0f, 0.5f), kEps));
  EXPECT_TRUE(capsules[1].end.isApprox(Vector3f(1.0f, 0.0f, 0.0f), kEps));
  EXPECT_NEAR(capsules[1].radius, 0.08f, kEps);
}

TEST(BodyFilter, MasksTheCapsuleInADepthImage) {
  // A wall 2 m in front of the camera and a band 0.5 m in front of it, of
  // which the forearm is the middle part. Two pixels on the arm have no return.
  constexpr int kRows = 48;
  constexpr int kCols = 64;
  const Camera camera(50.0f, 50.0f, 32.0f, 24.0f, kCols, kRows);
  DepthImage depth_host(kRows, kCols, MemoryType::kHost);
  for (int row = 0; row < kRows; ++row) {
    for (int col = 0; col < kCols; ++col) {
      depth_host(row, col) = (row >= 20 && row < 28) ? 0.5f : 2.0f;
    }
  }
  depth_host(24, 30) = 0.0f;
  depth_host(24, 31) = std::numeric_limits<float>::infinity();
  DepthImage depth(MemoryType::kDevice);
  depth.copyFrom(depth_host);

  // One link at the camera: the arm from x = -0.2 to 0.2 at depth 0.5, so
  // columns 6 to 58 of the band (x = (col - 32) * 0.01 m), none on the boundary.
  const std::vector<BodyCapsule> model = {
    {"forearm", {"link", Vector3f(-0.2f, 0.0f, 0.5f)}, {"link", Vector3f(0.2f, 0.0f, 0.5f)},
      0.055f},
  };
  auto cuda_stream = std::make_shared<CudaStreamOwning>();
  BodyFilter body_filter(model, 0.01f, cuda_stream);
  body_filter.setLinkPoses({Transform::Identity()});
  const Capsule capsule = placeCapsules(model, body_filter.links(), {Transform::Identity()},
      0.01f)[0];

  MonoImage mask(MemoryType::kDevice);
  body_filter.maskDepthImage(depth, camera, &mask);
  const int num_masked = body_filter.numMaskedPixels();
  MonoImage mask_host(MemoryType::kHost);
  mask_host.copyFrom(mask);
  ASSERT_EQ(mask_host.rows(), kRows);
  ASSERT_EQ(mask_host.cols(), kCols);

  int expected_masked = 0;
  for (int row = 0; row < kRows; ++row) {
    for (int col = 0; col < kCols; ++col) {
      const float d = depth_host(row, col);
      const bool on_body = d > 0.0f && std::isfinite(d) &&
        isPointInCapsule(camera.unprojectFromPixelIndices(Index2D(col, row), d), capsule);
      expected_masked += on_body ? 1 : 0;
      EXPECT_EQ(mask_host(row, col), on_body ? 0 : image::kMaskedValue)
        << "row " << row << " col " << col;
    }
  }
  EXPECT_EQ(num_masked, expected_masked);
  // The arm, but not the wall, the band beside the arm, or missing returns.
  EXPECT_GT(num_masked, 8 * 45);
  EXPECT_LT(num_masked, 8 * 55);
  EXPECT_EQ(mask_host(24, 32), 0);
  EXPECT_EQ(mask_host(10, 32), image::kMaskedValue);
  EXPECT_EQ(mask_host(24, 1), image::kMaskedValue);
  EXPECT_EQ(mask_host(24, 62), image::kMaskedValue);
  EXPECT_EQ(mask_host(24, 30), image::kMaskedValue);
  EXPECT_EQ(mask_host(24, 31), image::kMaskedValue);

  // Without capsules nearby, nothing is masked.
  body_filter.setLinkPoses({rotationAboutZ(0.0f, Vector3f(0.0f, 0.0f, 10.0f))});
  body_filter.maskDepthImage(depth, camera, &mask);
  EXPECT_EQ(body_filter.numMaskedPixels(), 0);
}

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

TEST(BodyFilter, DropsBodyPointsFromLidarScans) {
  const Lidar lidar(360, 31, 0.1f, 20.0f, M_PI / 6.0);
  const Vector3f wall(0.0f, 3.0f, 0.0f);
  const Vector3f arm(0.4f, 0.0f, 0.0f);
  const Vector3f behind_arm(3.0f, 0.1f, 0.0f);
  const auto cloud = makeCloud({wall, arm, behind_arm});

  auto cuda_stream = std::make_shared<CudaStreamOwning>();
  const std::vector<BodyCapsule> model = {
    {"arm", {"link", Vector3f(0.3f, 0.0f, 0.0f)}, {"link", Vector3f(0.5f, 0.0f, 0.0f)}, 0.05f},
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
  ASSERT_TRUE(lidar.project(arm, &u));
  EXPECT_EQ(range_host(u.y(), u.x()), 0.0f);
  ASSERT_TRUE(lidar.project(behind_arm, &u));
  EXPECT_NEAR(range_host(u.y(), u.x()), lidar.getDepth(behind_arm), kEps);

  // Without capsules every point is converted, as before the fork change.
  EXPECT_EQ(converter.depthImageFromPointcloudGPU(cloud, lidar, &range), 0);
  range_host.copyFrom(range);
  ASSERT_TRUE(lidar.project(arm, &u));
  EXPECT_NEAR(range_host(u.y(), u.x()), lidar.getDepth(arm), kEps);
}

TEST(BodyFilter, HeldCapsulesOnlyWhileHolding) {
  // The G1's left hand (g1_mapping's body_capsules.yaml): the hand from the
  // wrist to 0.13 m past the palm, and the held capsule along the palm's z
  // through the grasp centre. A bottle held upright reaches above and below
  // the palm, outside the hand's capsule.
  const std::vector<BodyCapsule> model = {
    {"hand", {"wrist", Vector3f::Zero()}, {"palm", Vector3f(0.13f, 0.0f, 0.0f)}, 0.07f},
    {"held", {"palm", Vector3f(0.08f, -0.04f, -0.10f)}, {"palm", Vector3f(0.08f, -0.04f, 0.10f)},
      0.05f, true},
  };
  // The sensor 1 m behind the palm, looking along its x.
  const std::vector<Transform> T_S_links = {
    rotationAboutZ(0.0f, Vector3f(0.9f, 0.0f, 0.0f)),
    rotationAboutZ(0.0f, Vector3f(1.0f, 0.0f, 0.0f)),
  };
  const Vector3f hand(1.05f, 0.0f, 0.0f);
  // On the bottle 0.15 m above and below the palm: 0.155 m from the hand's
  // axis (radius 0.10 with the margin), inside the held capsule.
  const Vector3f bottle_top(1.08f, -0.04f, 0.15f);
  const Vector3f bottle_bottom(1.08f, -0.04f, -0.15f);
  // Beside the hand, so its own pixel of the range image.
  const Vector3f wall(0.0f, 3.0f, 0.0f);
  const auto cloud = makeCloud({hand, bottle_top, bottle_bottom, wall});
  const Lidar lidar(360, 61, 0.1f, 20.0f, M_PI / 3.0);

  auto cuda_stream = std::make_shared<CudaStreamOwning>();
  BodyFilter body_filter(model, 0.03f, cuda_stream);
  EXPECT_TRUE(body_filter.hasHeldCapsules());
  EXPECT_FALSE(body_filter.holding());
  conversions::PointcloudConverter converter;
  DepthImage range(MemoryType::kDevice);
  DepthImage range_host(MemoryType::kHost);
  auto converted = [&](const Vector3f & p) {
      Index2D u;
      EXPECT_TRUE(lidar.project(p, &u));
      return range_host(u.y(), u.x()) > 0.0f;
    };

  // An empty hand: only the hand is dropped; what it is about to grasp is mapped.
  body_filter.setLinkPoses(T_S_links);
  EXPECT_EQ(body_filter.capsules().num_capsules, 1);
  EXPECT_TRUE(isPointInAnyCapsule(hand, body_filter.capsules()));
  EXPECT_FALSE(isPointInAnyCapsule(bottle_top, body_filter.capsules()));
  EXPECT_FALSE(isPointInAnyCapsule(bottle_bottom, body_filter.capsules()));
  EXPECT_EQ(
    converter.depthImageFromPointcloudGPU(cloud, lidar, &range, &body_filter.capsules()), 1);
  range_host.copyFrom(range);
  EXPECT_FALSE(converted(hand));
  EXPECT_TRUE(converted(bottle_top));
  EXPECT_TRUE(converted(bottle_bottom));
  EXPECT_TRUE(converted(wall));

  // Holding: the held object is dropped too, from the next placement on.
  body_filter.setHolding(true);
  EXPECT_EQ(body_filter.capsules().num_capsules, 1);
  body_filter.setLinkPoses(T_S_links);
  EXPECT_EQ(body_filter.capsules().num_capsules, 2);
  EXPECT_TRUE(isPointInAnyCapsule(bottle_top, body_filter.capsules()));
  EXPECT_EQ(
    converter.depthImageFromPointcloudGPU(cloud, lidar, &range, &body_filter.capsules()), 3);
  range_host.copyFrom(range);
  EXPECT_FALSE(converted(hand));
  EXPECT_FALSE(converted(bottle_top));
  EXPECT_FALSE(converted(bottle_bottom));
  EXPECT_TRUE(converted(wall));

  // Let go: mapped again.
  body_filter.setHolding(false);
  body_filter.setLinkPoses(T_S_links);
  EXPECT_FALSE(isPointInAnyCapsule(bottle_top, body_filter.capsules()));

  // A model without held capsules filters everything always, as before.
  BodyFilter plain({model[0]}, 0.03f, cuda_stream);
  EXPECT_FALSE(plain.hasHeldCapsules());
}

// ---- the held object (agentic-humanoid's decision 29) ----

// The held object's capsule placed on its link at the identity.
Capsule placedAtIdentity(const BodyCapsule & capsule)
{
  return placeCapsules({capsule}, {capsule.start.link}, {Transform::Identity()}, 0.0f)[0];
}

TEST(BodyFilter, HeldObjectCapsuleEnclosesBoxesAndCylinders) {
  // Any size and pose, no margin: every corner of a box and every point on a
  // cylinder's rims inside its capsule (as g1_definitions' test_body).
  std::mt19937 rng(29);
  std::uniform_real_distribution<float> length(0.02f, 0.3f);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  for (int i = 0; i < 200; ++i) {
    Transform T = Transform::Identity();
    T.linear() =
      Eigen::Quaternionf(unit(rng), unit(rng), unit(rng), unit(rng)).normalized().toRotationMatrix();
    T.translation() = 0.1f * Vector3f(unit(rng), unit(rng), unit(rng));
    const Vector3f box(length(rng), length(rng), length(rng));
    const Capsule box_capsule = placedAtIdentity(heldObjectCapsule("palm", HeldShape::kBox, box, T, 0.0f));
    for (const float x : {-0.5f, 0.5f}) {
      for (const float y : {-0.5f, 0.5f}) {
        for (const float z : {-0.5f, 0.5f}) {
          // Just inside the corner, for float rounding.
          const Vector3f corner = T * (0.99999f * Vector3f(x * box.x(), y * box.y(), z * box.z()));
          EXPECT_TRUE(isPointInCapsule(corner, box_capsule)) << i;
        }
      }
    }
    const Vector3f cylinder(box.x(), box.x(), box.z());
    const Capsule cylinder_capsule =
      placedAtIdentity(heldObjectCapsule("palm", HeldShape::kCylinder, cylinder, T, 0.0f));
    for (int k = 0; k < 16; ++k) {
      const float a = 2.0f * M_PI * k / 16.0f;
      for (const float z : {-0.5f, 0.5f}) {
        const Vector3f rim = T * (0.99999f * Vector3f(0.5f * cylinder.x() * std::cos(a),
          0.5f * cylinder.x() * std::sin(a), z * cylinder.z()));
        EXPECT_TRUE(isPointInCapsule(rim, cylinder_capsule)) << i;
      }
    }
  }
}

TEST(BodyFilter, HeldObjectCapsuleAsG1Definitions) {
  // test_body's test_the_held_objects_capsule_is_no_larger_than_needed: the
  // mustard's box, the capsule along its height, the margin added once.
  const BodyCapsule capsule = heldObjectCapsule(
    "right_hand_palm_link", HeldShape::kBox, Vector3f(0.096f, 0.058f, 0.191f),
    Transform::Identity(), 0.02f);
  EXPECT_EQ(capsule.start.link, "right_hand_palm_link");
  EXPECT_EQ(capsule.end.link, "right_hand_palm_link");
  EXPECT_NEAR(capsule.radius, 0.5f * std::hypot(0.096f, 0.058f) + 0.02f, 1e-6f);
  EXPECT_TRUE(capsule.start.offset.isApprox(Vector3f(0.0f, 0.0f, -0.0955f), kEps));
  EXPECT_TRUE(capsule.end.offset.isApprox(Vector3f(0.0f, 0.0f, 0.0955f), kEps));
  EXPECT_FALSE(capsule.held);
}

TEST(BodyFilter, HeldObjectsFollowTheirPalmUntilCleared) {
  // The left hand of HeldCapsulesOnlyWhileHolding, without held capsules: the
  // 134922 mustard's box in the palm is dropped while published, whatever
  // holding() says, and mapped once cleared.
  const std::vector<BodyCapsule> model = {
    {"hand", {"wrist", Vector3f::Zero()}, {"palm", Vector3f(0.13f, 0.0f, 0.0f)}, 0.09f},
  };
  const std::vector<Transform> T_S_links = {
    rotationAboutZ(0.0f, Vector3f(0.9f, 0.0f, 0.0f)),
    rotationAboutZ(0.3f, Vector3f(1.0f, 0.2f, 0.1f)),
  };
  Transform T_palm_object = Transform::Identity();
  T_palm_object.translation() = Vector3f(0.0598f, -0.0800f, -0.0272f);
  const Vector3f size(0.096f, 0.058f, 0.191f);
  // The box's far bottom corner, outside the hand's capsule.
  const Vector3f corner = T_S_links[1] * T_palm_object * Vector3f(0.047f, -0.028f, -0.095f);
  const Vector3f wall(0.0f, 3.0f, 0.0f);
  const auto cloud = makeCloud({corner, wall});
  const Lidar lidar(360, 61, 0.1f, 20.0f, M_PI / 3.0);

  auto cuda_stream = std::make_shared<CudaStreamOwning>();
  BodyFilter body_filter(model, 0.03f, cuda_stream);
  body_filter.setLinkPoses(T_S_links);
  ASSERT_FALSE(isPointInAnyCapsule(corner, body_filter.capsules()));

  EXPECT_TRUE(body_filter.setHeldObjects(
    {heldObjectCapsule("palm", HeldShape::kBox, size, T_palm_object, 0.0f)}));
  EXPECT_EQ(body_filter.numHeldObjects(), 1);
  EXPECT_EQ(body_filter.capsules().num_capsules, 1);  // from the next placement on
  body_filter.setLinkPoses(T_S_links);
  EXPECT_EQ(body_filter.capsules().num_capsules, 2);
  EXPECT_FALSE(body_filter.holding());
  EXPECT_TRUE(isPointInAnyCapsule(corner, body_filter.capsules()));
  conversions::PointcloudConverter converter;
  DepthImage range(MemoryType::kDevice);
  EXPECT_EQ(
    converter.depthImageFromPointcloudGPU(cloud, lidar, &range, &body_filter.capsules()), 1);

  EXPECT_TRUE(body_filter.setHeldObjects({}));
  body_filter.setLinkPoses(T_S_links);
  EXPECT_EQ(body_filter.capsules().num_capsules, 1);
  EXPECT_FALSE(isPointInAnyCapsule(corner, body_filter.capsules()));

  // A held object on a link without a pose, or more than fit: refused, none kept.
  EXPECT_TRUE(body_filter.setHeldObjects(
    {heldObjectCapsule("palm", HeldShape::kBox, size, T_palm_object, 0.0f)}));
  EXPECT_FALSE(body_filter.setHeldObjects(
    {heldObjectCapsule("elbow", HeldShape::kBox, size, T_palm_object, 0.0f)}));
  EXPECT_EQ(body_filter.numHeldObjects(), 0);
  EXPECT_FALSE(body_filter.setHeldObjects(std::vector<BodyCapsule>(kMaxBodyCapsules,
    heldObjectCapsule("palm", HeldShape::kBox, size, T_palm_object, 0.0f))));
  EXPECT_EQ(body_filter.numHeldObjects(), 0);
}

visualization_msgs::msg::Marker heldMarker(
  const std::string & frame, int type, const Vector3f & position, const Vector3f & size)
{
  visualization_msgs::msg::Marker m;
  m.header.frame_id = frame;
  m.ns = frame.substr(0, frame.find('_'));
  m.type = type;
  m.pose.position.x = position.x();
  m.pose.position.y = position.y();
  m.pose.position.z = position.z();
  m.pose.orientation.w = 1.0;
  m.scale.x = size.x();
  m.scale.y = size.y();
  m.scale.z = size.z();
  return m;
}

TEST(BodyFilter, HeldObjectsFromMarkers) {
  using Marker = visualization_msgs::msg::Marker;
  const std::vector<std::string> links = {"left_hand_palm_link", "right_hand_palm_link"};
  visualization_msgs::msg::MarkerArray msg;
  msg.markers.push_back(heldMarker("left_hand_palm_link", Marker::CUBE,
    Vector3f(0.06f, -0.08f, -0.03f), Vector3f(0.096f, 0.058f, 0.191f)));
  msg.markers.push_back(heldMarker("right_hand_palm_link", Marker::CYLINDER,
    Vector3f(0.06f, 0.08f, 0.0f), Vector3f(0.07f, 0.07f, 0.1f)));
  std::vector<BodyCapsule> capsules;
  std::string error;
  ASSERT_TRUE(heldObjectsFromMarkers(msg, links, 0.02f, &capsules, &error)) << error;
  ASSERT_EQ(capsules.size(), 2u);
  EXPECT_EQ(capsules[0].start.link, "left_hand_palm_link");
  EXPECT_NEAR(capsules[0].radius, 0.5f * std::hypot(0.096f, 0.058f) + 0.02f, 1e-6f);
  EXPECT_TRUE(capsules[0].end.offset.isApprox(Vector3f(0.06f, -0.08f, 0.0655f), kEps));
  EXPECT_EQ(capsules[1].start.link, "right_hand_palm_link");
  EXPECT_NEAR(capsules[1].radius, 0.035f + 0.02f, 1e-6f);

  // Nothing held.
  ASSERT_TRUE(heldObjectsFromMarkers(visualization_msgs::msg::MarkerArray(), links, 0.02f,
    &capsules, &error));
  EXPECT_TRUE(capsules.empty());

  // Not a held object: refused whole, with the reason.
  auto refused = [&](const Marker & bad) {
      visualization_msgs::msg::MarkerArray one = msg;
      one.markers.push_back(bad);
      const bool ok = heldObjectsFromMarkers(one, links, 0.02f, &capsules, &error);
      return !ok && capsules.empty() && !error.empty();
    };
  const Vector3f size(0.1f, 0.06f, 0.19f);
  EXPECT_TRUE(refused(heldMarker("torso_link", Marker::CUBE, Vector3f::Zero(), size)));
  EXPECT_TRUE(refused(heldMarker("left_hand_palm_link", Marker::SPHERE, Vector3f::Zero(), size)));
  EXPECT_TRUE(refused(heldMarker("left_hand_palm_link", Marker::CUBE, Vector3f::Zero(),
    Vector3f(0.1f, 0.0f, 0.19f))));
  Marker zero = heldMarker("left_hand_palm_link", Marker::CUBE, Vector3f::Zero(), size);
  zero.pose.orientation.w = 0.0;
  EXPECT_TRUE(refused(zero));
  Marker nan = heldMarker("left_hand_palm_link", Marker::CUBE, Vector3f::Zero(), size);
  nan.pose.position.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(refused(nan));
}

}  // namespace nvblox
