/**
 * @file      AdaptiveMEstimator.h
 * @brief     Adaptive M-estimator for robust loss functions.
 * @author    Seungwon Choi
 * @date      2025-09-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef LIDAR_ODOMETRY_ADAPTIVE_M_ESTIMATOR_H
#define LIDAR_ODOMETRY_ADAPTIVE_M_ESTIMATOR_H

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>

namespace lidar_slam {
namespace optimization {

/**
 * @brief Configuration for AdaptiveMEstimator
 */
struct AdaptiveMEstimatorConfig {
    bool use_adaptive_m_estimator = true;     ///< Enable adaptive M-estimator
    std::string loss_type = "cauchy";          ///< Loss function type: "cauchy", "huber"
    std::string scale_method = "PKO";         
    double fixed_scale_factor = 1.0;          ///< Fixed scale factor when scale_method is "fixed"
    double min_scale_factor = 0.01;           ///< Minimum allowed scale factor (also PKO alpha lower bound)
    double max_scale_factor = 10.0;           ///< Maximum allowed scale factor (also PKO alpha upper bound)
    
    // PKO (Probabilistic Kernel Optimization) parameters
    int num_alpha_segments = 1000;            ///< Number of alpha segments for PKO
    double truncated_threshold = 10.0;        ///< Truncated threshold for integration
    int gmm_components = 3;                   ///< Number of GMM components
    int gmm_sample_size = 100;                ///< Sample size for GMM fitting
    std::string pko_kernel_type = "cauchy";   ///< PKO kernel type: "huber", "cauchy", "tukey", "welsch", "gemanMcClure", "pseudoHuber"
};


class AdaptiveMEstimator {
public:
    /**
     * @brief Constructor with PKO configuration parameters
     * @param use_adaptive_m_estimator Enable adaptive M-estimator
     * @param loss_type Loss function type ("cauchy", "huber")
     * @param min_scale_factor Minimum allowed scale factor
     * @param max_scale_factor Maximum allowed scale factor
     * @param num_alpha_segments Number of alpha candidates for PKO
     * @param truncated_threshold Truncation threshold for PKO integration
     * @param gmm_components Number of GMM components
     * @param gmm_sample_size Sample size for GMM fitting
     * @param pko_kernel_type PKO kernel type
     */
    explicit AdaptiveMEstimator(
        bool use_adaptive_m_estimator = true,
        const std::string& loss_type = "cauchy",
        double min_scale_factor = 0.0001,
        double max_scale_factor = 0.1,
        int num_alpha_segments = 1000,
        double truncated_threshold = 10.0,
        int gmm_components = 1,
        int gmm_sample_size = 500,
        const std::string& pko_kernel_type = "cauchy"
    );
    
    /**
     * @brief Destructor
     */
    ~AdaptiveMEstimator() = default;
    
    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void set_config(const AdaptiveMEstimatorConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const AdaptiveMEstimatorConfig& get_config() const;
    
    /**
     * @brief Calculate scale factor from residuals
     * @param residuals Vector of residuals
     * @return Calculated scale factor (delta)
     */
    double calculate_scale_factor(const std::vector<double>& residuals);
    
    /**
     * @brief Calculate robust weight for a given residual
     * @param residual Input residual
     * @param scale_factor Scale factor (delta)
     * @return Robust weight [0, 1]
     */
    double calculate_weight(double residual, double scale_factor) const;
    

    
    /**
     * @brief Calculate information matrix diagonal from residuals
     * @param residuals Input residuals
     * @param information_diagonal Output information matrix diagonal (will be resized)
     * @return Scale factor used for calculation
     * 
     * Information matrix is diagonal with elements = weight^2 or weight depending on formulation
     */
    double calculate_information_matrix_diagonal(const std::vector<double>& residuals, 
                                                std::vector<double>& information_diagonal);
    
    /**
     * @brief Calculate full information matrix from residuals
     * @param residuals Input residuals
     * @param information_matrix Output information matrix (n x n, will be resized)
     * @return Scale factor used for calculation
     * 
     * For robust estimation, this is typically a diagonal matrix
     */
    double calculate_information_matrix(const std::vector<double>& residuals, 
                                       std::vector<std::vector<double>>& information_matrix);
    
    /**
     * @brief Calculate information weight (sqrt of information matrix diagonal)
     * @param residual Input residual  
     * @param scale_factor Scale factor (delta)
     * @return Information weight (typically sqrt(weight))
     */
    double calculate_information_weight(double residual, double scale_factor) const;
    
    /**
     * @brief Get the last computed scale factor
     * @return Last scale factor
     */
    double get_last_scale_factor() const { return m_last_scale_factor; }
    
    /**
     * @brief Reset internal state
     */
    void reset();
    
    /**
     * @brief PKO (Probabilistic Kernel Optimization) method to calculate optimal scale factor
     * @param residuals Input residuals
     * @return Optimal scale factor using JS divergence minimization
     */
    double calculate_pko_scale_factor(const std::vector<double>& residuals);

private:
    /**
     * @brief Initialize PKO (Probabilistic Kernel Optimization)
     */
    void initialize_pko();
    
    // PKO robust loss functions (used as kernels)
    /**
     * @brief Tukey weight function for PKO
     * @param residual Residual value
     * @param delta Scale parameter
     * @return Weight value
     */
    double tukey_weight(double residual, double delta) const;
    
    /**
     * @brief Welsch weight function for PKO
     * @param residual Residual value
     * @param delta Scale parameter
     * @return Weight value
     */
    double welsch_weight(double residual, double delta) const;
    
    /**
     * @brief Geman-McClure weight function for PKO
     * @param residual Residual value
     * @param delta Scale parameter
     * @return Weight value
     */
    double geman_mcclure_weight(double residual, double delta) const;
    
    /**
     * @brief Pseudo-Huber weight function for PKO
     * @param residual Residual value
     * @param delta Scale parameter
     * @return Weight value
     */
    double pseudo_huber_weight(double residual, double delta) const;
    
    /**
     * @brief PKO kernel weight function (dispatch to appropriate robust loss)
     * @param residual Residual value
     * @param delta Scale parameter (alpha in PKO context)
     * @return Weight value
     */
    double pko_kernel_weight(double residual, double delta) const;
    
    /**
     * @brief Gaussian Mixture Model fitting using EM algorithm
     * @param residuals Input residuals
     */
    void fit_gmm(const std::vector<double>& residuals);
    
    /**
     * @brief K-means initialization for GMM
     * @param residuals Input residuals
     */
    void kmeans_init(const std::vector<double>& residuals);
    
    /**
     * @brief Detect picks (local minima) in residual distribution for GMM initialization
     * @param residuals Input residuals
     * @return Vector of pick values to use as initial means
     */
    std::vector<double> detect_picks_for_init(const std::vector<double>& residuals);
    
    /**
     * @brief Calculate Jensen-Shannon divergence
     * @param residuals Input residuals
     * @param alpha Alpha parameter
     * @return JS divergence value
     */
    double calculate_js_divergence(const std::vector<double>& residuals, double alpha);
    
    /**
     * @brief Gaussian probability density function
     * @param x Input value
     * @param mean Mean
     * @param variance Variance
     * @return PDF value
     */
    double gaussian_pdf(double x, double mean, double variance) const;
    
    /**
     * @brief Calculate variance of data
     * @param data Input data
     * @param mean Mean of data
     * @return Variance
     */
    double calculate_variance(const std::vector<double>& data, double mean) const;
    
    /**
     * @brief Calculate partition function for given alpha
     * @param alpha Scale parameter
     * @return Partition function value
     */
    double calculate_partition_function(double alpha) const;
    
    /**
     * @brief Calculate partition function using numerical integration (backup style)
     * @param alpha Alpha parameter
     * @return Partition function value
     */
    double calculate_partition_function_integration(double alpha) const;
    
    /**
     * @brief Calculate adaptive weight for residual
     * @param residual Input residual
     * @param alpha Scale parameter
     * @return Adaptive weight
     */
    double calculate_adaptive_weight(double residual, double alpha) const;
    
    /**
     * @brief Log residual histogram for debugging
     * @param residuals Input residuals
     */
    void log_residual_histogram(const std::vector<double>& residuals) const;

private:
    AdaptiveMEstimatorConfig m_config;        ///< Configuration
    mutable double m_last_scale_factor;       ///< Last computed scale factor
    
    // PKO related members
    std::vector<double> m_alpha_candidates;   ///< Pre-computed alpha candidates for PKO
    std::vector<double> m_partition_functions; ///< Pre-computed partition functions
    std::vector<double> m_gmm_weights;        ///< GMM weights for current residuals
    std::vector<double> m_gmm_means;          ///< GMM means for current residuals
    std::vector<double> m_gmm_variances;      ///< GMM variances for current residuals
    double m_alpha_star_ref;                  ///< Reference alpha star value
    double m_max_density;                     ///< Maximum density encountered
};

} // namespace optimization
} // namespace lidar_slam

#endif // LIDAR_ODOMETRY_ADAPTIVE_M_ESTIMATOR_H
