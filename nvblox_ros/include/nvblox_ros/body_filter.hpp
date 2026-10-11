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

#ifndef NVBLOX_ROS__BODY_FILTER_HPP_
#define NVBLOX_ROS__BODY_FILTER_HPP_

#include <nvblox/nvblox.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nvblox
{

// agentic-humanoid fork: the robot's own body, as capsules around its links,
// dropped from depth images and LiDAR scans before integration. A sensor on a
// humanoid sees its arms and legs; mapped, they become obstacles inside the
// robot's own footprint.

/// A capsule placed in a sensor frame: every point within radius of the
/// segment from start to end.
struct Capsule
{
  Vector3f start;
  Vector3f end;
  float radius;
};

/// True if p lies inside the capsule or on its boundary.
__host__ __device__ inline bool isPointInCapsule(const Vector3f & p, const Capsule & capsule)
{
  const Vector3f axis = capsule.end - capsule.start;
  const float length_sq = axis.squaredNorm();
  float t = length_sq > 0.0f ? (p - capsule.start).dot(axis) / length_sq : 0.0f;
  t = fminf(fmaxf(t, 0.0f), 1.0f);
  return (p - (capsule.start + t * axis)).squaredNorm() <= capsule.radius * capsule.radius;
}

/// Most capsules a body model may have. The placed capsules go to the GPU as a
/// kernel argument (about 1 KB), so placing them needs no upload and no sync.
constexpr int kMaxBodyCapsules = 32;

/// Capsules placed in one sensor frame, passed to kernels by value.
struct CapsuleSet
{
  int num_capsules = 0;
  Capsule capsules[kMaxBodyCapsules];
};

/// True if p lies inside any of the capsules. Non-finite points never do.
__host__ __device__ inline bool isPointInAnyCapsule(
  const Vector3f & p, const Capsule * capsules, const int num_capsules)
{
  // Eigen's allFinite() is host-only. NaN fails every comparison.
  constexpr float kMaxFloat = 3.402823466e+38f;
  if (!(fabsf(p.x()) <= kMaxFloat && fabsf(p.y()) <= kMaxFloat && fabsf(p.z()) <= kMaxFloat)) {
    return false;
  }
  for (int i = 0; i < num_capsules; ++i) {
    if (isPointInCapsule(p, capsules[i])) {
      return true;
    }
  }
  return false;
}

__host__ __device__ inline bool isPointInAnyCapsule(
  const Vector3f & p, const CapsuleSet & capsule_set)
{
  return isPointInAnyCapsule(p, capsule_set.capsules, capsule_set.num_capsules);
}

/// One end of a body capsule: a point fixed to a link, in the link's axes.
struct CapsuleEnd
{
  std::string link;
  Vector3f offset = Vector3f::Zero();
};

/// A capsule of the body model, between points on one or two links.
struct BodyCapsule
{
  std::string name;
  CapsuleEnd start;
  CapsuleEnd end;
  float radius = 0.0f;
  /// Around what a hand holds: filtered only while the robot holds something
  /// (BodyFilter::setHolding), so an empty hand does not hide what it is
  /// about to grasp.
  bool held = false;
};

/// The links a body model hangs from, each once, in a fixed order.
std::vector<std::string> bodyLinks(const std::vector<BodyCapsule> & model);

/// The shape of a held object (agentic-humanoid's decision 29).
enum class HeldShape { kBox, kCylinder };

/// The capsule on link that encloses a held box or cylinder of size (in m; a
/// cylinder's diameter in x and y, its height in z) at T_link_object, margin_m
/// added: along a box's longest side, the radius half the diagonal of the
/// other two; along a cylinder's z, half its larger diameter. The same capsule
/// as g1_definitions.held.capsule. CHECKs a positive, finite size.
BodyCapsule heldObjectCapsule(
  const std::string & link, HeldShape shape, const Vector3f & size,
  const Transform & T_link_object, float margin_m);

/// Places the body model in a sensor frame.
/// @param model The body model.
/// @param links The links, as returned by bodyLinks(model).
/// @param T_S_links The pose of each link in the sensor frame, in the order of links.
/// @param margin_m Added to every radius.
/// @return One capsule per model capsule, in the sensor frame.
std::vector<Capsule> placeCapsules(
  const std::vector<BodyCapsule> & model, const std::vector<std::string> & links,
  const std::vector<Transform> & T_S_links, float margin_m);

/// Drops the robot's body from depth images and LiDAR scans, on the GPU.
/// Place the capsules in the sensor frame at the measurement's stamp first,
/// then mask the image or pass capsules() to the LiDAR conversion.
class BodyFilter
{
public:
  BodyFilter(
    std::vector<BodyCapsule> model, float margin_m,
    std::shared_ptr<CudaStream> cuda_stream);

  /// The links whose poses place the capsules, each once.
  const std::vector<std::string> & links() const {return links_;}

  /// Places the capsules in the sensor frame, the held ones only while holding().
  /// @param T_S_links The pose of each link in the sensor frame, in the order of links().
  void setLinkPoses(const std::vector<Transform> & T_S_links);

  /// Whether a hand holds something; false until set. Takes effect at the next
  /// setLinkPoses(). Safe to call from another thread.
  void setHolding(bool holding) {holding_ = holding;}
  bool holding() const {return holding_;}

  /// Whether the model has held capsules.
  bool hasHeldCapsules() const;

  /// The held objects' capsules (heldObjectCapsule), each on a link of
  /// links(), placed after the model's at every setLinkPoses() from the next
  /// one on; replaces the last set (empty: nothing held). Independent of
  /// holding(): the publisher clears them. Safe to call from another thread.
  /// Returns false, and keeps none, if a capsule hangs from a link the model
  /// has no pose for or the model and the held objects exceed kMaxBodyCapsules.
  bool setHeldObjects(std::vector<BodyCapsule> held_objects);
  int numHeldObjects() const;

  /// The capsules placed by the last setLinkPoses().
  const CapsuleSet & capsules() const {return capsules_;}

  /// Writes a mask of the pixels to integrate: image::kMaskedValue (active)
  /// where the back-projected depth lies outside every capsule, and 0 on the
  /// body. Pass it to Mapper::integrateDepth as a MaskedDepthImageConstView:
  /// an inactive pixel still clears the space in front of it, but adds no
  /// surface. The mask is resized to the image if needed, otherwise reused.
  /// Asynchronous on the filter's stream.
  void maskDepthImage(
    const DepthImage & depth_image, const Camera & camera, MonoImage * mask);

  /// The number of body pixels in the last maskDepthImage(). Synchronizes the
  /// stream, so call it once the mask has been used.
  int numMaskedPixels();

private:
  std::vector<BodyCapsule> model_;
  std::vector<std::string> links_;
  float margin_m_;
  std::shared_ptr<CudaStream> cuda_stream_;
  std::atomic<bool> holding_{false};
  mutable std::mutex held_objects_mutex_;
  std::vector<BodyCapsule> held_objects_;

  CapsuleSet capsules_;
  unified_ptr<int> num_masked_device_;
  unified_ptr<int> num_masked_host_;
};

}  // namespace nvblox

#endif  // NVBLOX_ROS__BODY_FILTER_HPP_
