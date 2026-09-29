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
  capsules_.num_capsules = placed.size();
  std::copy(placed.begin(), placed.end(), capsules_.capsules);
}

namespace {

__global__ void maskDepthImageKernel(const DepthImageConstView depth,
                                     const Camera camera,
                                     const CapsuleSet capsules,
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
    const Vector3f p_S = camera.unprojectFromPixelIndices(Index2D(col, row), d);
    on_body = isPointInAnyCapsule(p_S, capsules);
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
      DepthImageConstView(depth_image), camera, capsules_,
      MonoImageView(*mask), num_masked_device_.get());
  checkCudaErrors(cudaPeekAtLastError());
  num_masked_device_.copyToAsync(num_masked_host_, *cuda_stream_);
}

int BodyFilter::numMaskedPixels() {
  cuda_stream_->synchronize();
  return *num_masked_host_;
}

}  // namespace nvblox
