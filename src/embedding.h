#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace embedding {

// Cosine similarity between two vectors. Returns 0 if either is zero-length.
double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

// Top-k indices by cosine similarity to query, sorted descending.
std::vector<std::pair<std::size_t, double>> top_k(
    const std::vector<std::vector<float>>& vectors,
    const std::vector<float>& query,
    std::size_t k);

}  // namespace embedding
