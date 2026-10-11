// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nvblox_ros/node_params.hpp"

#include <algorithm>

#include "nvblox_ros/mapper_initialization.hpp"

namespace nvblox
{

// Declare a ROS parameter and assign its value to param
template<typename ParamType, typename RosParamType = ParamType>
void initParam(
  rclcpp::Node * node, Param<ParamType> * param,
  std::optional<nvblox::parameters::ParameterTreeNode *> tree = std::nullopt)
{
  const auto desc = param->description();
  rcl_interfaces::msg::ParameterDescriptor ros_desc;
  ros_desc.description = desc.help_string;

  *param = static_cast<ParamType>(node->declare_parameter<RosParamType>(
      desc.name, static_cast<RosParamType>(desc.default_value), ros_desc));

  // Add to the tree, if requested
  if (tree.has_value()) {
    CHECK(tree.value()->children().has_value())
      << "Declare parameter requires an initialized non-leaf ParameterTreeNode.";
    tree.value()->children().value().push_back(
      nvblox::parameters::ParameterTreeNode(desc.name, static_cast<ParamType>(*param)));
  }
}

// Declare a ROS string parameter and assign its value to param
void initStringParam(
  rclcpp::Node * node, StringParam * param,
  std::optional<nvblox::parameters::ParameterTreeNode *> tree = std::nullopt)
{
  const auto desc = param->description();
  rcl_interfaces::msg::ParameterDescriptor ros_desc;
  ros_desc.description = desc.help_string;

  *param = node->declare_parameter<std::string>(desc.name, desc.default_value, ros_desc);

  // Add to the tree, if requested
  if (tree.has_value()) {
    CHECK(tree.value()->children().has_value())
      << "Declare parameter requires an initialized non-leaf ParameterTreeNode.";
    tree.value()->children().value().push_back(
      nvblox::parameters::ParameterTreeNode(desc.name, std::string(*param)));
  }
}

RosDatasetType dataset_type_from_string(const std::string & dataset_str, rclcpp::Node * node)
{
  if (dataset_str == "3dmatch") {
    return RosDatasetType::kThreedMatch;
  } else if (dataset_str == "redwood") {
    return RosDatasetType::kRedwood;
  } else if (dataset_str == "replica") {
    return RosDatasetType::kReplica;
  } else if (dataset_str == "rosbag") {
    return RosDatasetType::kRosbag;
  } else {
    RCLCPP_WARN_STREAM(
      node->get_logger(), "Requested dataset type: \""
        << dataset_str
        << "\" not recognized. Defaulting to: "
        << toString(RosDatasetType::kThreedMatch));
    return RosDatasetType::kThreedMatch;
  }
}

void initializeBaseNodeParams(
  rclcpp::Node * node, BaseNodeParams * params,
  std::optional<nvblox::parameters::ParameterTreeNode *> parameter_tree)
{
  initParam<CudaStreamType, int>(node, &params->cuda_stream_type, parameter_tree);
  initStringParam(node, &params->global_frame, parameter_tree);
  initParam<float>(node, &params->voxel_size, parameter_tree);
  initParam<float>(node, &params->max_back_projection_distance, parameter_tree);
  initParam<float>(node, &params->layer_visualization_min_tsdf_weight, parameter_tree);
  initParam<float>(node, &params->layer_visualization_exclusion_height_m, parameter_tree);
  initParam<float>(node, &params->layer_visualization_exclusion_radius_m, parameter_tree);
  initParam<float>(node, &params->layer_streamer_bandwidth_limit_mbps, parameter_tree);
  initParam<float>(node, &params->distance_map_unknown_value_optimistic, parameter_tree);
  initParam<float>(node, &params->distance_map_unknown_value_pessimistic, parameter_tree);
  initStringParam(node, &params->esdf_mode_str, parameter_tree);
  params->esdf_mode = esdf_mode_from_string(params->esdf_mode_str, node);
  initStringParam(node, &params->mapping_type_str, parameter_tree);
  params->mapping_type = mapping_type_from_string(params->mapping_type_str, node);
}

void initializeNvbloxNodeParams(
  rclcpp::Node * node, NvbloxNodeParams * params,
  std::optional<nvblox::parameters::ParameterTreeNode *> parameter_tree)
{
  initializeBaseNodeParams(node, params, parameter_tree);

  initStringParam(node, &params->pose_frame, parameter_tree);
  initStringParam(node, &params->map_clearing_frame_id, parameter_tree);
  initStringParam(node, &params->after_shutdown_map_save_path, parameter_tree);
  initStringParam(
    node, &params->esdf_slice_bounds_visualization_attachment_frame_id,
    parameter_tree);
  initStringParam(
    node, &params->workspace_height_bounds_visualization_attachment_frame_id,
    parameter_tree);
  initStringParam(
    node, &params->ground_plane_visualization_attachment_frame_id,
    parameter_tree);

  initParam<bool>(node, &params->publish_esdf_distance_slice, parameter_tree);
  initParam<bool>(node, &params->use_color, parameter_tree);
  initParam<bool>(node, &params->use_depth, parameter_tree);
  initParam<bool>(node, &params->use_segmentation, parameter_tree);
  initParam<bool>(node, &params->use_lidar, parameter_tree);
  initParam<bool>(node, &params->use_non_equal_vertical_fov_lidar_params, parameter_tree);
  initParam<bool>(node, &params->print_timings_to_console, parameter_tree);
  initParam<bool>(node, &params->print_rates_to_console, parameter_tree);
  initParam<bool>(node, &params->print_queue_drops_to_console, parameter_tree);
  initParam<bool>(node, &params->print_delays_to_console, parameter_tree);
  initParam<bool>(node, &params->layer_visualization_undo_gamma_correction, parameter_tree);
  initParam<bool>(node, &params->output_pessimistic_distance_map, parameter_tree);

  initParam<int>(node, &params->num_cameras, parameter_tree);
  initParam<int>(node, &params->lidar_width, parameter_tree);
  initParam<int>(node, &params->lidar_height, parameter_tree);
  initParam<int>(node, &params->tick_period_ms, parameter_tree);
  initParam<int>(node, &params->print_statistics_on_console_period_ms, parameter_tree);
  initParam<int>(node, &params->maximum_input_queue_length, parameter_tree);
  initParam<int>(node, &params->back_projection_subsampling, parameter_tree);

  initParam<float>(node, &params->lidar_vertical_fov_rad, parameter_tree);
  initParam<float>(node, &params->lidar_min_valid_range_m, parameter_tree);
  initParam<float>(node, &params->lidar_max_valid_range_m, parameter_tree);
  initParam<float>(node, &params->min_angle_below_zero_elevation_rad, parameter_tree);
  initParam<float>(node, &params->max_angle_above_zero_elevation_rad, parameter_tree);
  initParam<float>(node, &params->esdf_slice_bounds_visualization_side_length, parameter_tree);
  initParam<float>(
    node, &params->workspace_height_bounds_visualization_side_length,
    parameter_tree);
  initParam<float>(
    node, &params->ground_plane_visualization_side_length,
    parameter_tree);
  initParam<float>(node, &params->integrate_depth_rate_hz, parameter_tree);
  initParam<float>(node, &params->integrate_color_rate_hz, parameter_tree);
  initParam<float>(node, &params->integrate_lidar_rate_hz, parameter_tree);
  initParam<float>(node, &params->update_mesh_rate_hz, parameter_tree);
  initParam<float>(node, &params->update_esdf_rate_hz, parameter_tree);
  initParam<float>(node, &params->publish_layer_rate_hz, parameter_tree);
  initParam<float>(node, &params->publish_debug_vis_rate_hz, parameter_tree);
  initParam<float>(node, &params->decay_tsdf_rate_hz, parameter_tree);
  initParam<float>(node, &params->decay_dynamic_occupancy_rate_hz, parameter_tree);
  initParam<float>(node, &params->clear_map_outside_radius_rate_hz, parameter_tree);
  initParam<float>(node, &params->map_clearing_radius_m, parameter_tree);
  initParam<float>(node, &params->esdf_and_gradients_unobserved_value, parameter_tree);
  initParam<bool>(node, &params->body_filter_enable, parameter_tree);
  initParam<float>(node, &params->body_filter_tf_timeout_s, parameter_tree);
  initParam<float>(node, &params->body_filter_margin_m, parameter_tree);
  initParam<bool>(node, &params->body_filter_held_object_enable, parameter_tree);
  initParam<float>(node, &params->body_filter_held_object_margin_m, parameter_tree);
}

std::vector<BodyCapsule> initializeBodyCapsuleParams(rclcpp::Node * node)
{
  const std::string prefix = "body_filter.capsules.";
  const auto names = node->declare_parameter<std::vector<std::string>>(
    prefix + "names", std::vector<std::string>{});
  auto offset = [&](const std::string & name) -> Vector3f {
      const auto xyz =
        node->declare_parameter<std::vector<double>>(name, std::vector<double>{0.0, 0.0, 0.0});
      CHECK_EQ(xyz.size(), 3u) << name << " must be [x, y, z]";
      return Vector3f(xyz[0], xyz[1], xyz[2]);
    };
  std::vector<BodyCapsule> capsules;
  for (const std::string & name : names) {
    const std::string capsule = prefix + name + ".";
    BodyCapsule body_capsule;
    body_capsule.name = name;
    body_capsule.start.link = node->declare_parameter<std::string>(capsule + "start_link", "");
    body_capsule.start.offset = offset(capsule + "start_offset");
    body_capsule.end.link = node->declare_parameter<std::string>(capsule + "end_link", "");
    body_capsule.end.offset = offset(capsule + "end_offset");
    body_capsule.radius = node->declare_parameter<double>(capsule + "radius", 0.0);
    CHECK(!body_capsule.start.link.empty() && !body_capsule.end.link.empty())
      << "Body capsule " << name << " needs start_link and end_link";
    CHECK_GT(body_capsule.radius, 0.0f) << "Body capsule " << name << " needs a radius";
    capsules.push_back(body_capsule);
  }
  // Filtered only while body_filter/holding is true (BodyCapsule::held), with
  // held_only_while_holding; without it (the default) every capsule always.
  const auto held = node->declare_parameter<std::vector<std::string>>(
    "body_filter.held_capsules", std::vector<std::string>{});
  const bool gate = node->declare_parameter<bool>("body_filter.held_only_while_holding", false);
  for (const std::string & name : held) {
    auto it = std::find_if(
      capsules.begin(), capsules.end(),
      [&](const BodyCapsule & capsule) {return capsule.name == name;});
    CHECK(it != capsules.end())
      << "body_filter.held_capsules names " << name << ", which is not in body_filter.capsules";
    it->held = gate;
  }
  return capsules;
}

void initializeFuserNodeParams(
  rclcpp::Node * node, FuserNodeParams * params,
  std::optional<nvblox::parameters::ParameterTreeNode *> parameter_tree)
{
  initializeBaseNodeParams(node, params, parameter_tree);
  initStringParam(node, &params->dataset_type_str, parameter_tree);
  params->dataset_type = dataset_type_from_string(params->dataset_type_str, node);
  initStringParam(node, &params->dataset_path, parameter_tree);
  initParam<int>(node, &params->number_of_frames_to_integrate, parameter_tree);
  initParam<bool>(node, &params->update_on_key, parameter_tree);
  initStringParam(node, &params->depth_topic, parameter_tree);
  initStringParam(node, &params->depth_camera_info_topic, parameter_tree);
  initStringParam(node, &params->color_topic, parameter_tree);
  initStringParam(node, &params->color_camera_info_topic, parameter_tree);
  initParam<float>(node, &params->tf_lead_time_s, parameter_tree);
}

}  // namespace nvblox
