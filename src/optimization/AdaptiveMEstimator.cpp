/**
 * @file      AdaptiveMEstimator.cpp
 * @brief     Implementation of Adaptive M-estimator.
 * @author    Seungwon Choi
 * @date      2025-09-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "AdaptiveMEstimator.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <chrono>
#include <random>
#include "util/LogUtils.h"


#include <iostream>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lidar_odometry {
namespace optimization {

AdaptiveMEstimator::AdaptiveMEstimator(
    bool use_adaptive_m_estimator,
    const std::string& loss_type,
    double min_scale_factor,
    double max_scale_factor,
    int num_alpha_segments,
    double truncated_threshold,
    int gmm_components,
    int gmm_sample_size,
    const std::string& pko_kernel_type)
    : m_last_scale_factor(min_scale_factor)
    , m_alpha_star_ref(max_scale_factor)
    , m_max_density(0.0) {
    
    // Initialize configuration with provided values (PKO only)
    m_config.use_adaptive_m_estimator = use_adaptive_m_estimator;
    m_config.loss_type = loss_type;
    m_config.min_scale_factor = min_scale_factor;
    m_config.max_scale_factor = max_scale_factor;
    m_config.num_alpha_segments = num_alpha_segments;
    m_config.truncated_threshold = truncated_threshold;
    m_config.gmm_components = gmm_components;
    m_config.gmm_sample_size = gmm_sample_size;
    m_config.pko_kernel_type = pko_kernel_type;
    
    // PKO will be initialized lazily when needed
}


const AdaptiveMEstimatorConfig& AdaptiveMEstimator::get_config() const {
    return m_config;
}

double AdaptiveMEstimator::calculate_scale_factor(
    const std::vector<double>& residuals) {
    
    if (residuals.empty()) {
        LOG_WARN("Empty residuals vector, using default scale factor");
        return 1.0;
    }

    double scale_factor = 1.0;

    scale_factor = calculate_pko_scale_factor(residuals);

    // Update last computed scale factor
    m_last_scale_factor = scale_factor;
    
    return scale_factor;
}

double AdaptiveMEstimator::calculate_weight(double residual, double scale_factor) const {
    if (scale_factor <= 0.0) {
        return 1.0;
    }
    
    // Use PKO kernel weight directly
    return pko_kernel_weight(residual, scale_factor);
}



void AdaptiveMEstimator::reset() {
    m_last_scale_factor = m_config.min_scale_factor;
    m_alpha_star_ref = m_config.max_scale_factor;
}


// PKO robust loss functions
double AdaptiveMEstimator::tukey_weight(double residual, double delta) const {
    double abs_residual = std::abs(residual);
    if (abs_residual < delta) {
        double x_over_c = abs_residual / delta;
        double x_over_c2 = x_over_c * x_over_c;
        return (1 - x_over_c2) * (1 - x_over_c2);
    } else {
        return 0.0;
    }
}

double AdaptiveMEstimator::welsch_weight(double residual, double delta) const {
    double e2 = residual * residual;
    double delta2 = delta * delta;
    return std::exp(-e2 / delta2 / 2.0);
}

double AdaptiveMEstimator::geman_mcclure_weight(double residual, double delta) const {
    double e2 = residual * residual;
    double delta2 = delta * delta;
    return residual * delta2 / (delta2 + e2) / (delta2 + e2);
}

double AdaptiveMEstimator::pseudo_huber_weight(double residual, double delta) const {
    double e = residual;
    double delta2 = delta * delta;
    return delta2 / std::pow(delta2 + e * e, 1.5);
}

double AdaptiveMEstimator::pko_kernel_weight(double residual, double delta) const {
    const std::string& kernel_type = m_config.pko_kernel_type;
    
    if (kernel_type == "huber") {
        double abs_residual = std::abs(residual);
        if (abs_residual <= delta) {
            return 1.0;
        } else {
            return delta / abs_residual;
        }
    } else if (kernel_type == "cauchy") {
        double e2 = residual * residual;
        double delta2 = delta * delta;
        return delta2 / (delta2 + e2);
    } else if (kernel_type == "tukey") {
        return tukey_weight(residual, delta);
    } else if (kernel_type == "welsch") {
        return welsch_weight(residual, delta);
    } else if (kernel_type == "gemanMcClure") {
        return geman_mcclure_weight(residual, delta);
    } else if (kernel_type == "pseudoHuber") {
        return pseudo_huber_weight(residual, delta);
    } else {
        // Default to Cauchy
        double e2 = residual * residual;
        double delta2 = delta * delta;
        return delta2 / (delta2 + e2);
    }
}

double AdaptiveMEstimator::calculate_information_matrix_diagonal(const std::vector<double>& residuals, 
                                                               std::vector<double>& information_diagonal) {
    if (residuals.empty()) {
        information_diagonal.clear();
        return m_config.fixed_scale_factor;
    }
    
    // Resize information diagonal vector
    information_diagonal.resize(residuals.size());
    
    // If adaptive M-estimator is disabled, return identity information matrix
    if (!m_config.use_adaptive_m_estimator) {
        std::fill(information_diagonal.begin(), information_diagonal.end(), 1.0);
        return m_last_scale_factor;
    }
    
    // PKO 방법에서는 커널이 직접 가중치를 결정하므로 정보 매트릭스 계산 불필요
    std::fill(information_diagonal.begin(), information_diagonal.end(), 1.0);
    return m_last_scale_factor;
}

double AdaptiveMEstimator::calculate_information_matrix(const std::vector<double>& residuals, 
                                                       std::vector<std::vector<double>>& information_matrix) {
    if (residuals.empty()) {
        information_matrix.clear();
        return m_config.min_scale_factor;
    }
    
    size_t n = residuals.size();
    
    // Initialize information matrix as n x n zero matrix
    information_matrix.assign(n, std::vector<double>(n, 0.0));
    
    // Calculate diagonal elements (off-diagonal remain zero for robust estimation)
    std::vector<double> information_diagonal;
    double scale_factor = calculate_information_matrix_diagonal(residuals, information_diagonal);
    
    // Fill diagonal elements
    for (size_t i = 0; i < n; ++i) {
        information_matrix[i][i] = information_diagonal[i];
    }
    
    return scale_factor;
}

double AdaptiveMEstimator::calculate_information_weight(double residual, double scale_factor) const {
    if (scale_factor <= 0.0) {
        LOG_WARN("[AdaptiveMEstimator] Invalid scale factor for information weight: {}", scale_factor);
        return 1.0;
    }
    
    // Calculate robust weight first
    double weight = calculate_weight(residual, scale_factor);
    
    // Information weight is typically sqrt(information_matrix_element)
    // Since information_matrix_element = weight^2, information_weight = weight
    return weight;
}

// PKO (Probabilistic Kernel Optimization) Implementation
void AdaptiveMEstimator::initialize_pko() {
    // 백업과 동일한 방식으로 alpha candidates와 partition functions를 미리 계산
    m_alpha_candidates.clear();
    m_partition_functions.clear();
    
    m_alpha_candidates.resize(m_config.num_alpha_segments + 1);
    m_partition_functions.resize(m_config.num_alpha_segments + 1);
    
    // 첫 번째 값은 min_scale_factor와 그에 해당하는 partition function
    m_alpha_candidates[0] = m_config.min_scale_factor;
    m_partition_functions[0] = calculate_partition_function(m_config.min_scale_factor);
    
    // 나머지 값들 (log scaling)
    for (int i = 1; i <= m_config.num_alpha_segments; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(m_config.num_alpha_segments);
        double log_scaled_value = (std::pow(100.0, t) - 1.0) / 99.0;  // 백업과 동일한 log scaling
        double alpha = m_config.min_scale_factor + (m_config.max_scale_factor - m_config.min_scale_factor) * log_scaled_value;
        
        double Z = calculate_partition_function(alpha);
        
        m_alpha_candidates[i] = alpha;
        m_partition_functions[i] = Z;
    }
}

double AdaptiveMEstimator::calculate_pko_scale_factor(const std::vector<double>& residuals) {
    if (residuals.empty()) {
        return 1.0;
    }
    
    // Lazy initialization of PKO
    if (m_alpha_candidates.empty()) {
        initialize_pko();
    }
    
    // Fit GMM to residual distribution
    fit_gmm(residuals);
    
    double best_alpha = m_config.min_scale_factor;
    double best_cost = std::numeric_limits<double>::max();
    
    for (size_t i = 1; i < m_alpha_candidates.size(); ++i) {
        double alpha = m_alpha_candidates[i];


        // // To assure graduated non-convexity
        // if(alpha >= m_alpha_star_ref)
        //     continue;
        
        double js_divergence = calculate_js_divergence(residuals, alpha);

        // LOG_INFO("[AdaptiveMEstimator] Alpha: {:.6f}, JS Divergence: {:.6f}", alpha, js_divergence);
        
        if (js_divergence < best_cost) {
            best_cost = js_divergence;
            best_alpha = alpha;
        }
    }

    // LOG_ERROR("Best Alpha: {:.6f}, Best JS Divergence: {:.6f}", best_alpha, best_cost);
    
    // Update reference alpha
    m_alpha_star_ref = best_alpha;
    
    // Update last scale factor BEFORE logging histogram
    m_last_scale_factor = best_alpha;
    
    // Log residual histogram with updated scale factor
    log_residual_histogram(residuals);

    // LOG_INFO("[AdaptiveMEstimator] Selected PKO scale factor (alpha*): {:.6f}, JS Divergence: {:.6f}", best_alpha, best_cost);
    
    return best_alpha;
}


void AdaptiveMEstimator::fit_gmm(const std::vector<double>& residuals) {
    // Remove histogram logging from here - it will be done after best alpha calculation
    
    if (residuals.empty()) {
        return;
    }
    
    // config에서 설정된 비율 또는 고정 샘플 크기 사용
    int n = residuals.size();
    int sample_size;
    
    if (m_config.gmm_sample_size > 0) {
        // config에서 직접 샘플 크기 지정 (고정값)
        sample_size = m_config.gmm_sample_size;
    } else {
        // 기본값: 전체 데이터의 10% (최소 100개, 최대 10000개)
        sample_size = std::max(100, static_cast<int>(n * 0.1));
        sample_size = std::min(sample_size, 10000);
    }
    
    if (sample_size > n) {
        sample_size = n; // 전체 데이터가 샘플 크기보다 작으면 전체 사용
    }
    
    // 2. 데이터에서 임의로 샘플 추출 (백업과 완전히 동일)
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    // Use fixed seed for consistent results
    std::mt19937 g(42); // Fixed seed instead of random_device
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<double> sampled_data(sample_size);
    for (int i = 0; i < sample_size; ++i) {
        sampled_data[i] = residuals[indices[i]];
    }
    
    // 이제 n은 샘플 사이즈를 기준으로 재설정 (백업과 동일)
    n = sample_size;
    
    // 3. 파라미터 초기화 (백업과 완전히 동일)
    
    // K-means 초기화 (fixed seed for consistency)
    std::mt19937 gen(42); // Fixed seed
    std::uniform_int_distribution<> dis(0, sampled_data.size() - 1);

    m_gmm_means.resize(m_config.gmm_components);
    // 첫 번째 component는 항상 mean=0으로 고정 (작은 residual 모델링)
    m_gmm_means[0] = 0.0;
    // 나머지 components는 랜덤 초기화
    for (int i = 1; i < m_config.gmm_components; ++i) {
        m_gmm_means[i] = sampled_data[dis(gen)];
    }

    std::vector<int> clusters(sampled_data.size());
    std::vector<double> new_means(m_config.gmm_components);

    // K-means clustering 수렴까지 반복 (백업과 동일)
    while (true) {
        // Assign points to the nearest cluster
        for (size_t i = 0; i < sampled_data.size(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int cluster_index = 0;
            for (int j = 0; j < m_config.gmm_components; ++j) {
                double dist = std::abs(sampled_data[i] - m_gmm_means[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    cluster_index = j;
                }
            }
            clusters[i] = cluster_index;
        }

        // Calculate new means
        std::fill(new_means.begin(), new_means.end(), 0.0);
        std::vector<int> counts(m_config.gmm_components, 0);
        for (size_t i = 0; i < sampled_data.size(); ++i) {
            new_means[clusters[i]] += sampled_data[i];
            counts[clusters[i]]++;
        }
        for (int j = 0; j < m_config.gmm_components; ++j) {
            if (j == 0) {
                // 첫 번째 component는 항상 mean=0으로 고정
                new_means[j] = 0.0;
            } else if (counts[j] > 0) {
                new_means[j] /= static_cast<double>(counts[j]);
            }
        }

        // Check for convergence
        if (m_gmm_means == new_means) {
            break;
        }
        // 첫 번째 component mean은 항상 0으로 유지
        new_means[0] = 0.0;
        m_gmm_means = new_means;
    }
    
    // 초기 분산 설정 (백업과 동일)
    double mean_of_data = std::accumulate(sampled_data.begin(), sampled_data.end(), 0.0) / sampled_data.size();
    double initial_variance = 0.0;
    for (double x : sampled_data) {
        initial_variance += std::pow(x - mean_of_data, 2);
    }
    initial_variance /= sampled_data.size();
    
    m_gmm_variances.assign(m_config.gmm_components, initial_variance);
    
    // 초기 가중치를 클러스터 크기에 비례하게 설정 (로그 수정)
    std::vector<int> cluster_counts(m_config.gmm_components, 0);
    for (size_t i = 0; i < sampled_data.size(); ++i) {
        cluster_counts[clusters[i]]++;
    }
    
    m_gmm_weights.resize(m_config.gmm_components);
    for (int j = 0; j < m_config.gmm_components; ++j) {
        m_gmm_weights[j] = static_cast<double>(cluster_counts[j]) / static_cast<double>(sampled_data.size());
    }
    
    // EM algorithm for GMM fitting (백업과 동일한 방식)
    const int max_iterations = 100; // 백업과 동일
    const double convergence_threshold = 1e-6; // 백업과 동일
    
    std::vector<std::vector<double>> responsibilities(n, std::vector<double>(m_config.gmm_components));
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // E-step: calculate responsibilities (백업과 동일)
        std::vector<double> sum_responsibilities(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m_config.gmm_components; ++j) {
                responsibilities[i][j] = m_gmm_weights[j] * gaussian_pdf(sampled_data[i], m_gmm_means[j], m_gmm_variances[j]);
                sum_responsibilities[i] += responsibilities[i][j];
            }
            for (int j = 0; j < m_config.gmm_components; ++j) {
                responsibilities[i][j] /= sum_responsibilities[i];
            }
        }
        
        // M-step: update parameters (백업과 동일)
        std::vector<double> N_k(m_config.gmm_components, 0.0);
        for (int j = 0; j < m_config.gmm_components; ++j) {
            for (int i = 0; i < n; ++i) {
                N_k[j] += responsibilities[i][j];
            }
        }
        
        // Update weights, means, variances (백업과 동일)
        std::vector<double> new_weights(m_config.gmm_components);
        std::vector<double> new_means_em(m_config.gmm_components, 0.0);
        std::vector<double> new_variances(m_config.gmm_components, 0.0);
        
        for (int j = 0; j < m_config.gmm_components; ++j) {
            new_weights[j] = N_k[j] / static_cast<double>(n);
            
            // Update mean
            if (j == 0) {
                // 첫 번째 component는 항상 mean=0으로 고정
                new_means_em[j] = 0.0;
            } else {
                for (int i = 0; i < n; ++i) {
                    new_means_em[j] += responsibilities[i][j] * sampled_data[i];
                }
                new_means_em[j] /= N_k[j];
            }
            
            // Update variance (첫 번째 component는 mean=0 기준으로 계산)
            for (int i = 0; i < n; ++i) {
                double diff = sampled_data[i] - new_means_em[j];
                new_variances[j] += responsibilities[i][j] * diff * diff;
            }
            new_variances[j] /= N_k[j];
            
            // Prevent variance collapse (백업과 동일)
            new_variances[j] = std::max(new_variances[j], 1e-6);
        }
        
        // Check convergence (백업과 동일 - means 변화량 기준, 첫 번째는 제외)
        double param_change = 0.0;
        for (int j = 1; j < m_config.gmm_components; ++j) {
            param_change += std::abs(new_means_em[j] - m_gmm_means[j]);
        }
        
        m_gmm_weights = new_weights;
        // 첫 번째 component mean은 항상 0으로 유지
        new_means_em[0] = 0.0;
        m_gmm_means = new_means_em;
        m_gmm_variances = new_variances;
        
        if (param_change < convergence_threshold) {
            break;
        }
    }
}

void AdaptiveMEstimator::kmeans_init(const std::vector<double>& residuals) {
    // Safety check
    if (residuals.empty()) {
        return;
    }
    
    // config에서 설정된 샘플 크기 사용
    int n = residuals.size();
    int sample_size = m_config.gmm_sample_size;
    if (sample_size > n) {
        sample_size = n;
    }
    
    // 데이터에서 임의로 샘플 추출
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 g(42); // Fixed seed for consistency
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<double> sampled_data(sample_size);
    for (int i = 0; i < sample_size; ++i) {
        sampled_data[i] = residuals[indices[i]];
    }
    
    // Initialize GMM parameters
    m_gmm_weights.assign(m_config.gmm_components, 1.0 / m_config.gmm_components);
    m_gmm_means.resize(m_config.gmm_components);
    m_gmm_variances.resize(m_config.gmm_components);
    
    // Pick 기반 초기화 시도
    std::vector<double> picks = detect_picks_for_init(sampled_data);
    
    if (picks.size() >= static_cast<size_t>(m_config.gmm_components)) {
        // Pick들이 충분히 있으면 사용
        for (int i = 0; i < m_config.gmm_components; ++i) {
            m_gmm_means[i] = picks[i];
        }
        LOG_DEBUG("[AdaptiveMEstimator] Using {} picks for GMM initialization", picks.size());
    } else {
        // Pick이 부족하면 기존 K-means++ 방식 사용
        LOG_DEBUG("[AdaptiveMEstimator] Insufficient picks ({}), using K-means++ initialization", picks.size());
        
        std::mt19937 gen(42); // Fixed seed for consistency
        std::uniform_int_distribution<> dis(0, sampled_data.size() - 1);

        for (int i = 0; i < m_config.gmm_components; ++i) {
            m_gmm_means[i] = sampled_data[dis(gen)];
        }
    }

    std::vector<int> clusters(sampled_data.size());
    std::vector<double> new_means(m_config.gmm_components);

    // K-means clustering 수렴까지 반복 (백업과 동일)
    while (true) {
        // Assign points to the nearest cluster
        for (size_t i = 0; i < sampled_data.size(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int cluster_index = 0;
            for (int j = 0; j < m_config.gmm_components; ++j) {
                double dist = std::abs(sampled_data[i] - m_gmm_means[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    cluster_index = j;
                }
            }
            clusters[i] = cluster_index;
        }

        // Calculate new means
        std::fill(new_means.begin(), new_means.end(), 0.0);
        std::vector<int> counts(m_config.gmm_components, 0);
        for (size_t i = 0; i < sampled_data.size(); ++i) {
            new_means[clusters[i]] += sampled_data[i];
            counts[clusters[i]]++;
        }
        for (int j = 0; j < m_config.gmm_components; ++j) {
            if (counts[j] > 0) {
                new_means[j] /= static_cast<double>(counts[j]);
            }
        }

        // Check for convergence
        if (m_gmm_means == new_means) {
            break;
        }
        m_gmm_means = new_means;
    }
    
    // 초기 분산 설정 (백업과 동일)
    double mean_of_data = std::accumulate(sampled_data.begin(), sampled_data.end(), 0.0) / sampled_data.size();
    double initial_variance = 0.0;
    for (double x : sampled_data) {
        initial_variance += std::pow(x - mean_of_data, 2);
    }
    initial_variance /= sampled_data.size();
    
    m_gmm_variances.assign(m_config.gmm_components, initial_variance);
}

std::vector<double> AdaptiveMEstimator::detect_picks_for_init(const std::vector<double>& residuals) {
    std::vector<double> picks;
    
    if (residuals.size() < 5) {
        return picks;  // 데이터가 너무 적으면 pick detection 불가
    }
    
    // 데이터를 정렬하여 히스토그램 생성
    std::vector<double> sorted_data = residuals;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    // 히스토그램 bin 수 결정 (Sturges' rule 사용)
    int num_bins = std::min(50, static_cast<int>(std::ceil(std::log2(sorted_data.size())) + 1));
    
    double min_val = sorted_data.front();
    double max_val = sorted_data.back();
    double bin_width = (max_val - min_val) / num_bins;
    
    if (bin_width <= 0.0) {
        return picks;  // 모든 값이 동일하면 pick 없음
    }
    
    // 히스토그램 생성
    std::vector<int> histogram(num_bins, 0);
    std::vector<double> bin_centers(num_bins);
    
    for (int i = 0; i < num_bins; ++i) {
        bin_centers[i] = min_val + (i + 0.5) * bin_width;
    }
    
    for (double val : sorted_data) {
        int bin_idx = static_cast<int>((val - min_val) / bin_width);
        bin_idx = std::min(bin_idx, num_bins - 1);  // 경계값 처리
        histogram[bin_idx]++;
    }
    
    // Pick detection: 주변보다 현저히 낮은 bin들을 찾기
    for (int i = 1; i < num_bins - 1; ++i) {
        double current = histogram[i];
        double left = histogram[i - 1];
        double right = histogram[i + 1];
        
        // Pick 조건: 현재 bin이 주변 bin들보다 현저히 낮음
        double threshold_ratio = 0.3;  // 주변의 30% 이하면 pick으로 간주
        
        if (current < threshold_ratio * std::max(left, right) && 
            current > 0 &&  // 완전히 비어있지 않음
            std::max(left, right) > 2) {  // 주변에 충분한 데이터 있음
            
            picks.push_back(bin_centers[i]);
        }
    }
    
    // Pick들을 값 순으로 정렬
    std::sort(picks.begin(), picks.end());
    
    // 너무 가까운 pick들 제거 (bin_width의 2배 이내면 제거)
    std::vector<double> filtered_picks;
    for (size_t i = 0; i < picks.size(); ++i) {
        bool too_close = false;
        for (const double& existing_pick : filtered_picks) {
            if (std::abs(picks[i] - existing_pick) < 2.0 * bin_width) {
                too_close = true;
                break;
            }
        }
        if (!too_close) {
            filtered_picks.push_back(picks[i]);
        }
    }
    
    // Create a string representation of first few picks for logging
    std::string picks_str;
    size_t max_picks_to_show = std::min(filtered_picks.size(), size_t(5));
    for (size_t i = 0; i < max_picks_to_show; ++i) {
        if (i > 0) picks_str += ", ";
        picks_str += std::to_string(filtered_picks[i]);
    }
    if (filtered_picks.size() > 5) {
        picks_str += "...";
    }
    
    LOG_DEBUG("[AdaptiveMEstimator] Detected {} picks from {} bins: [{}]", 
                  filtered_picks.size(), num_bins, picks_str);
    
    return filtered_picks;
}

double AdaptiveMEstimator::gaussian_pdf(double x, double mean, double variance) const {
    if (variance <= 0.0) {
        return 0.0;
    }
    
    double diff = x - mean;
    double exponent = -0.5 * (diff * diff) / variance;
    double normalization = 1.0 / std::sqrt(2.0 * M_PI * variance);
    
    return normalization * std::exp(exponent);
}

double AdaptiveMEstimator::calculate_partition_function(double alpha) const {
    // 백업과 동일한 numerical integration 방식 사용
    return calculate_partition_function_integration(alpha);
}

double AdaptiveMEstimator::calculate_partition_function_integration(double alpha) const {
    // 백업과 정확히 동일한 numerical integration 방식
    // Z(alpha) = integral of kernel_weight(x, alpha) dx from 0 to truncated_threshold
    
    const double integration_bound = m_config.truncated_threshold;
    const double integration_step = 0.01; // 백업에서 사용하는 step size
    
    double integral = 0.0;
    
    // 백업과 동일: 0부터 integration_bound까지 적분 (대칭성 가정 안함)
    for (double x = 0.0; x <= integration_bound; x += integration_step) {
        double kernel_value = pko_kernel_weight(x, alpha);
        integral += kernel_value * integration_step;
    }
    
    return std::max(integral, 1e-10); // 백업과 동일한 최소값
}

double AdaptiveMEstimator::calculate_js_divergence(const std::vector<double>& /* residuals */, double alpha) {
    // 백업과 동일한 방식으로 JS divergence 계산
    
    // 백업에서 사용하는 방식: 이산화된 residual 범위에서 P(r)과 Q(r) 비교
    int num_segments = 100;
    double dr = m_config.truncated_threshold / static_cast<double>(num_segments);
    
    std::vector<double> resvec(num_segments);
    for (int i = 0; i < num_segments; ++i) {
        resvec[i] = dr * (1 + static_cast<double>(i)); // 0~truncated_threshold
    }
    
    double cost = 0.0;
    // 백업과 같이 미리 계산된 partition function 사용 (동적 계산 대신)
    double partition_func = 0.0;
    
    // Find corresponding partition function for this alpha
    for (size_t j = 0; j < m_alpha_candidates.size(); ++j) {
        if (std::abs(m_alpha_candidates[j] - alpha) < 1e-10) {
            partition_func = m_partition_functions[j];
            break;
        }
    }
    
    // 만약 정확히 일치하지 않으면 동적 계산
    if (partition_func == 0.0) {
        partition_func = calculate_partition_function(alpha);
    }
    
    if (partition_func < 1e-10) {
        return std::numeric_limits<double>::max();
    }
    
    double total_Pr = 0.0;
    double total_Q = 0.0;
    double cnt = 0.0; // 백업과 동일한 변수명 사용
    
    for (double r : resvec) {
        // P(r): Empirical distribution approximated by GMM
        double Pr = 0.0;
        if (!m_gmm_weights.empty() && !m_gmm_means.empty() && !m_gmm_variances.empty()) {
            for (int m = 0; m < m_config.gmm_components && m < static_cast<int>(m_gmm_weights.size()); ++m) {
                double component_pdf = gaussian_pdf(r, m_gmm_means[m], m_gmm_variances[m]);
                Pr += m_gmm_weights[m] * component_pdf;
            }
        }
        Pr += 1e-10; // 백업과 완전히 동일한 epsilon 처리
        
        // Q(r): Kernel distribution (백업과 완전히 동일한 방식)
        double kernel_val = pko_kernel_weight(r, alpha);
        double Q = kernel_val / (partition_func + 1e-10) + 1e-10; // 백업과 동일
        
        total_Pr += Pr;
        total_Q += Q;
        
        // Jensen-Shannon divergence (백업과 정확히 동일한 계산)
        double M = 0.5 * (Pr + Q);
        
        // 백업과 동일한 JS divergence 계산
        double jsd = 0.5 * (Pr * std::log(Pr / M) + Q * std::log(Q / M));
        
        // 백업과 동일한 NaN 처리
        if (std::isnan(jsd)) {
            continue; // 백업과 동일: NaN이면 continue
        }
        
        cost += jsd; // 백업과 동일: 모든 valid한 jsd를 누적
        cnt += 1.0;  // 백업과 동일: 유효한 샘플 카운트
    }
    
    if (cnt == 0) { // 백업과 동일한 조건
        return std::numeric_limits<double>::max();
    }
    
    double avg_js_divergence = cost / cnt; // 백업과 동일: cost/cnt
    
    return avg_js_divergence;
}

double AdaptiveMEstimator::calculate_adaptive_weight(double residual, double alpha) const {
    // Calculate adaptive weight based on PKO kernel
    double abs_residual = std::abs(residual);
    return std::exp(-alpha * abs_residual);
}

void AdaptiveMEstimator::log_residual_histogram(const std::vector<double>& residuals) const {
    if (residuals.empty()) return;
    
    // Create histogram with 15 bins (for detailed distribution analysis)
    const int num_bins = 15;
    auto minmax = std::minmax_element(residuals.begin(), residuals.end());
    double min_val = *minmax.first;
    double max_val = *minmax.second;
    double range = max_val - min_val;
    
    if (range < 1e-10) {
        LOG_INFO("Residual histogram: All values are ~{:.6f}", min_val);
        return;
    }
    
    double bin_width = range / num_bins;
    std::vector<int> bins(num_bins, 0);
    std::vector<double> weight_sums(num_bins, 0.0);
    std::vector<int> weight_counts(num_bins, 0);
    std::vector<double> gmm_densities(num_bins, 0.0);
    
    // Fill histogram and calculate weights for each bin
    for (double residual : residuals) {
        int bin_idx = static_cast<int>((residual - min_val) / bin_width);
        if (bin_idx >= num_bins) bin_idx = num_bins - 1;
        if (bin_idx < 0) bin_idx = 0;
        
        bins[bin_idx]++;
        
        // Calculate weight for this residual using current scale factor
        if (m_last_scale_factor > 0) {
            double weight = calculate_weight(residual, m_last_scale_factor);
            weight_sums[bin_idx] += weight;
            weight_counts[bin_idx]++;
        }
    }
    
    // Calculate GMM densities for each bin
    int max_residual_count = 0;
    if (!m_gmm_weights.empty() && !m_gmm_means.empty() && !m_gmm_variances.empty()) {
        max_residual_count = *std::max_element(bins.begin(), bins.end());
        
        for (int i = 0; i < num_bins; ++i) {
            double bin_center = min_val + (i + 0.5) * bin_width;
            double gmm_prob = 0.0;
            
            for (int k = 0; k < m_config.gmm_components && k < static_cast<int>(m_gmm_weights.size()); ++k) {
                gmm_prob += m_gmm_weights[k] * gaussian_pdf(bin_center, m_gmm_means[k], m_gmm_variances[k]);
            }
            gmm_densities[i] = gmm_prob;
        }
        
        // Normalize GMM densities to match residual count scale
        double max_gmm = *std::max_element(gmm_densities.begin(), gmm_densities.end());
        
        if (max_gmm > 0) {
            double scale_factor_norm = static_cast<double>(max_residual_count) / max_gmm;
            for (double& density : gmm_densities) {
                density *= scale_factor_norm;
            }
        }
    }
    /*
    // Log histogram with three distributions

    LOG_INFO("                                | [Residual Distribution] | [   P_data(inlier|r)  ] |         [P_model(inlier|r,c)]        |");
    LOG_INFO("                                |-------------------------|-------------------------|--------------------------------------|");

    // Find maximum count for proper normalization
    int max_count = *std::max_element(bins.begin(), bins.end());
    
    for (int i = 0; i < num_bins; ++i) {
        double bin_start = min_val + i * bin_width;
        double bin_end = min_val + (i + 1) * bin_width;
        int count = bins[i];
        double percentage = 100.0 * count / residuals.size();
        
        // Calculate average weight for this bin
        double avg_weight = 0.0;
        if (weight_counts[i] > 0) {
            avg_weight = weight_sums[i] / weight_counts[i];
        }
        
        // Create residual distribution bar (max 25 chars, normalized to max count)
        const int max_bar_width = 25;
        int residual_bar_length = 0;
        if (max_count > 0) {
            residual_bar_length = static_cast<int>((static_cast<double>(count) / max_count) * max_bar_width);
        }
        residual_bar_length = std::min(residual_bar_length, max_bar_width);
        std::string residual_bar(residual_bar_length, '*');
        residual_bar.resize(max_bar_width, ' ');
        
        // Create GMM distribution bar (scale to match residual bar)
        int gmm_bar_length = 0;
        if (!gmm_densities.empty() && max_residual_count > 0) {
            gmm_bar_length = static_cast<int>((gmm_densities[i] / max_residual_count) * max_bar_width);
            gmm_bar_length = std::min(gmm_bar_length, max_bar_width);
        }
        std::string gmm_bar(gmm_bar_length, '+');
        gmm_bar.resize(max_bar_width, ' ');
        
        // Create weight bar (scale to 0-max_bar_width range)
        int weight_bar_length = static_cast<int>(avg_weight * max_bar_width);
        weight_bar_length = std::min(weight_bar_length, max_bar_width);
        std::string weight_bar(weight_bar_length, '=');
        
        // Check if scale factor falls in this bin
        std::string scale_indicator = "";
        if (m_last_scale_factor >= bin_start && m_last_scale_factor < bin_end) {
            scale_indicator = " <-- SCALE FACTOR";
        }
        
        LOG_INFO("  [{:7.4f},{:7.4f}): {:3d}({:4.1f}%) |{}|{}|{}| w={:.3f}{}", 
                   bin_start, bin_end, count, percentage, 
                   residual_bar, gmm_bar, weight_bar, avg_weight, scale_indicator);
    }
    */
    
}

} // namespace optimization
} // namespace lidar_odometry
