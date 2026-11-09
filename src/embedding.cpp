#include "embedding.h"

#include <algorithm>

using namespace std;

namespace embedding {

double cosine_similarity(const vector<float>& a, const vector<float>& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0;
    }
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (na == 0.0 || nb == 0.0) {
        return 0.0;
    }
    return dot / (sqrt(na) * sqrt(nb));
}

vector<pair<size_t, double>> top_k(
    const vector<vector<float>>& vectors,
    const vector<float>& query,
    size_t k) {
    vector<pair<size_t, double>> scores;
    scores.reserve(vectors.size());
    for (size_t i = 0; i < vectors.size(); ++i) {
        double sim = cosine_similarity(vectors[i], query);
        scores.emplace_back(i, sim);
    }
    partial_sort(scores.begin(), scores.begin() + min(k, scores.size()), scores.end(),
                 [](const pair<size_t, double>& x, const pair<size_t, double>& y) {
                     return x.second > y.second;
                 });
    if (scores.size() > k) {
        scores.resize(k);
    }
    return scores;
}

}  // namespace embedding
