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

#ifndef NVBLOX_ROS__HELD_OBJECTS_HPP_
#define NVBLOX_ROS__HELD_OBJECTS_HPP_

#include <string>
#include <vector>

#include <visualization_msgs/msg/marker_array.hpp>

#include "nvblox_ros/body_filter.hpp"

namespace nvblox
{

// agentic-humanoid fork, decision 29: what a hand holds, published by the pick
// server that grasped it (g1_definitions.held): a latched MarkerArray with the
// whole state, one CUBE or CYLINDER per holding hand in its palm's frame (the
// marker's frame_id), its pose and size there; empty while nothing is held.

/// The held objects' capsules (heldObjectCapsule, margin_m added) of a
/// body_filter/held_object message. Every marker must be a CUBE or CYLINDER
/// in a frame of links, with a positive, finite size and a finite pose with a
/// non-zero quaternion; otherwise false, error says why and capsules is
/// empty (a publisher bug: never masked half-way).
bool heldObjectsFromMarkers(
  const visualization_msgs::msg::MarkerArray & msg, const std::vector<std::string> & links,
  float margin_m, std::vector<BodyCapsule> * capsules, std::string * error);

}  // namespace nvblox

#endif  // NVBLOX_ROS__HELD_OBJECTS_HPP_
