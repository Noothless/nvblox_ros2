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

#include "nvblox_ros/held_objects.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace nvblox {

bool heldObjectsFromMarkers(const visualization_msgs::msg::MarkerArray& msg,
                            const std::vector<std::string>& links,
                            const float margin_m,
                            std::vector<BodyCapsule>* capsules,
                            std::string* error) {
  using Marker = visualization_msgs::msg::Marker;
  CHECK_NOTNULL(capsules);
  CHECK_NOTNULL(error);
  capsules->clear();
  error->clear();
  for (const Marker& m : msg.markers) {
    std::stringstream why;
    const std::string& frame = m.header.frame_id;
    const auto& p = m.pose.position;
    const auto& o = m.pose.orientation;
    const Vector3f size(static_cast<float>(m.scale.x), static_cast<float>(m.scale.y),
                        static_cast<float>(m.scale.z));
    const double norm = std::sqrt(o.w * o.w + o.x * o.x + o.y * o.y + o.z * o.z);
    if (std::find(links.begin(), links.end(), frame) == links.end()) {
      why << "frame " << frame << " is not a body link";
    } else if (m.type != Marker::CUBE && m.type != Marker::CYLINDER) {
      why << "marker type " << m.type << ", not CUBE or CYLINDER";
    } else if (!size.allFinite() || !(size.array() > 0.0f).all()) {
      why << "size " << size.transpose() << " is not positive and finite";
    } else if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
               !std::isfinite(norm) || norm <= 0.0) {
      why << "the pose is not finite";
    }
    if (!why.str().empty()) {
      *error = "held object of " + m.ns + ": " + why.str();
      capsules->clear();
      return false;
    }
    Transform T = Transform::Identity();
    T.linear() = Eigen::Quaterniond(o.w / norm, o.x / norm, o.y / norm, o.z / norm)
                     .toRotationMatrix()
                     .cast<float>();
    T.translation() = Eigen::Vector3d(p.x, p.y, p.z).cast<float>();
    capsules->push_back(heldObjectCapsule(
        frame, m.type == Marker::CUBE ? HeldShape::kBox : HeldShape::kCylinder,
        size, T, margin_m));
  }
  return true;
}

}  // namespace nvblox
