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

#include "nvblox_ros/body_filter.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace nvblox {

std::vector<std::string> bodyLinks(const std::vector<BodyCapsule>& model) {
  std::vector<std::string> links;
  for (const BodyCapsule& capsule : model) {
    for (const CapsuleEnd* end : {&capsule.start, &capsule.end}) {
      if (std::find(links.begin(), links.end(), end->link) == links.end()) {
        links.push_back(end->link);
      }
    }
  }
  return links;
}

std::vector<Capsule> placeCapsules(const std::vector<BodyCapsule>& model,
                                   const std::vector<std::string>& links,
                                   const std::vector<Transform>& T_S_links,
                                   const float margin_m) {
  CHECK_EQ(links.size(), T_S_links.size());
  auto place = [&](const CapsuleEnd& end) -> Vector3f {
    const auto it = std::find(links.begin(), links.end(), end.link);
    CHECK(it != links.end()) << "No pose for body link " << end.link;
    return T_S_links[it - links.begin()] * end.offset;
  };
  std::vector<Capsule> capsules;
  capsules.reserve(model.size());
  for (const BodyCapsule& capsule : model) {
    capsules.push_back(Capsule{place(capsule.start), place(capsule.end),
                               capsule.radius + margin_m});
  }
  return capsules;
}

BodyCapsule heldObjectCapsule(const std::string& link, const HeldShape shape,
                              const Vector3f& size,
                              const Transform& T_link_object,
                              const float margin_m) {
  CHECK(size.allFinite() && (size.array() > 0.0f).all())
      << "A held object needs a positive size, not " << size.transpose();
  int axis = 2;
  float radius = 0.0f;
  if (shape == HeldShape::kBox) {
    size.maxCoeff(&axis);
    const int a = (axis + 1) % 3;
    const int b = (axis + 2) % 3;
    radius = 0.5f * std::hypot(size[a], size[b]);
  } else {
    radius = 0.5f * std::max(size.x(), size.y());
  }
  const Vector3f half = 0.5f * size[axis] * T_link_object.linear().col(axis);
  const Vector3f centre = T_link_object.translation();
  BodyCapsule capsule;
  capsule.name = link + "_held_object";
  capsule.start = CapsuleEnd{link, centre - half};
  capsule.end = CapsuleEnd{link, centre + half};
  capsule.radius = radius + margin_m;
  return capsule;
}

BodyFilter::BodyFilter(std::vector<BodyCapsule> model, const float margin_m,
                       std::shared_ptr<CudaStream> cuda_stream)
    : model_(std::move(model)),
      links_(bodyLinks(model_)),
      margin_m_(margin_m),
      cuda_stream_(std::move(cuda_stream)),
      num_masked_device_(make_unified<int>(MemoryType::kDevice)),
      num_masked_host_(make_unified<int>(MemoryType::kHost)) {
  CHECK(!model_.empty()) << "The body filter needs at least one capsule";
  CHECK_LE(model_.size(), static_cast<size_t>(kMaxBodyCapsules))
      << "The body filter takes at most " << kMaxBodyCapsules << " capsules";
}

void BodyFilter::setLinkPoses(const std::vector<Transform>& T_S_links) {
  const std::vector<Capsule> placed =
      placeCapsules(model_, links_, T_S_links, margin_m_);
  const bool holding = holding_;
  capsules_.num_capsules = 0;
  for (size_t i = 0; i < placed.size(); ++i) {
    if (!model_[i].held || holding) {
      capsules_.capsules[capsules_.num_capsules++] = placed[i];
    }
  }
  std::vector<BodyCapsule> held_objects;
  {
    std::lock_guard<std::mutex> lock(held_objects_mutex_);
    held_objects = held_objects_;
  }
  // setHeldObjects() checked their links and that they fit.
  for (const Capsule& capsule :
       placeCapsules(held_objects, links_, T_S_links, margin_m_)) {
    capsules_.capsules[capsules_.num_capsules++] = capsule;
  }
}

bool BodyFilter::setHeldObjects(std::vector<BodyCapsule> held_objects) {
  bool valid = model_.size() + held_objects.size() <=
               static_cast<size_t>(kMaxBodyCapsules);
  for (const BodyCapsule& capsule : held_objects) {
    for (const CapsuleEnd* end : {&capsule.start, &capsule.end}) {
      valid = valid && std::find(links_.begin(), links_.end(), end->link) !=
                           links_.end();
    }
  }
  std::lock_guard<std::mutex> lock(held_objects_mutex_);
  held_objects_ = valid ? std::move(held_objects) : std::vector<BodyCapsule>{};
  return valid;
}

int BodyFilter::numHeldObjects() const {
  std::lock_guard<std::mutex> lock(held_objects_mutex_);
  return static_cast<int>(held_objects_.size());
}

void BodyFilter::setDepthPixelBoxes(const std::vector<PixelBox>& boxes) {
  CHECK_LE(boxes.size(), static_cast<size_t>(kMaxPixelBoxes))
      << "The body filter takes at most " << kMaxPixelBoxes << " depth pixel boxes";
  pixel_boxes_.num_boxes = 0;
  for (const PixelBox& b : boxes) {
    CHECK(0.0f <= b.u0 && b.u0 < b.u1 && b.u1 <= 1.0f && 0.0f <= b.v0 &&
          b.v0 < b.v1 && b.v1 <= 1.0f)
        << "A depth pixel box needs 0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1, not ["
        << b.u0 << ", " << b.v0 << ", " << b.u1 << ", " << b.v1 << "]";
    pixel_boxes_.boxes[pixel_boxes_.num_boxes++] = b;
  }
}

bool BodyFilter::hasHeldCapsules() const {
  return std::any_of(model_.begin(), model_.end(),
                     [](const BodyCapsule& capsule) { return capsule.held; });
}

namespace {

__global__ void maskDepthImageKernel(const DepthImageConstView depth,
                                     const Camera camera,
                                     const CapsuleSet capsules,
                                     const PixelBoxSet boxes,
                                     MonoImageView mask, int* num_masked) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= depth.cols() || row >= depth.rows()) {
    return;
  }
  const float d = depth(row, col);
  // Invalid (<= 0) and missing (+inf, NaN) returns stay as they are: the
  // integrator ignores them or treats them as it would without the filter.
  bool on_body = false;
  if (d > 0.0f && isfinite(d)) {
    on_body = isPixelInAnyBox(col, row, depth.cols(), depth.rows(), boxes);
    if (!on_body) {
      const Vector3f p_S = camera.unprojectFromPixelIndices(Index2D(col, row), d);
      on_body = isPointInAnyCapsule(p_S, capsules);
    }
  }
  mask(row, col) = on_body ? 0 : image::kMaskedValue;
  if (on_body) {
    atomicAdd(num_masked, 1);
  }
}

}  // namespace

void BodyFilter::maskDepthImage(const DepthImage& depth_image,
                                const Camera& camera, MonoImage* mask) {
  CHECK_NOTNULL(mask);
  CHECK_EQ(depth_image.rows(), camera.rows());
  CHECK_EQ(depth_image.cols(), camera.cols());
  if (mask->rows() != depth_image.rows() ||
      mask->cols() != depth_image.cols()) {
    *mask = MonoImage(depth_image.rows(), depth_image.cols(),
                      MemoryType::kDevice);
  }
  num_masked_device_.setZeroAsync(*cuda_stream_);
  constexpr int kThreadsPerDim = 16;
  const dim3 threads(kThreadsPerDim, kThreadsPerDim);
  const dim3 blocks((depth_image.cols() + kThreadsPerDim - 1) / kThreadsPerDim,
                    (depth_image.rows() + kThreadsPerDim - 1) / kThreadsPerDim);
  maskDepthImageKernel<<<blocks, threads, 0, *cuda_stream_>>>(
      DepthImageConstView(depth_image), camera, capsules_, pixel_boxes_,
      MonoImageView(*mask), num_masked_device_.get());
  checkCudaErrors(cudaPeekAtLastError());
  num_masked_device_.copyToAsync(num_masked_host_, *cuda_stream_);
}

int BodyFilter::numMaskedPixels() {
  cuda_stream_->synchronize();
  return *num_masked_host_;
}

}  // namespace nvblox
