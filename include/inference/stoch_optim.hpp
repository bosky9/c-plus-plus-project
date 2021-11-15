/**
 * @file stoch_optim.hpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#pragma once

#include "Eigen/Core"

/**
 * @class StochOptim stoch_optim.hpp
 * @brief Parent class for optimizers
 */
class StochOptim {
public:
    /**
     * @brief Constructor for StochOptim
     * @param starting_parameters Starting parameters
     * @param starting_variances Starting variances
     * @param learning_rate Learning rate
     */
    StochOptim(Eigen::VectorXd starting_parameters, Eigen::VectorXd starting_variances, double learning_rate);

    /**
     * @brief Update variances and parameters given a vector of gradients
     * @param gradient Vector of gradients
     * @return Vector of updated parameters
     */
    virtual Eigen::VectorXd update(Eigen::VectorXd& gradient);

    /**
     * @brief Returns optimizer's parameters
     * @return Parameters
     */
    [[nodiscard]] Eigen::VectorXd get_parameters() const;

protected:
    Eigen::VectorXd _parameters; ///< Parameters
    Eigen::VectorXd _variances;  ///< Variances
    double _learning_rate;       ///< Learning rate
    static double _epsilon;      ///< Epsilon
    int _t = 1;
};

/**
 * @class RMSProp stoch_optim.hpp
 * @brief Root Mean Square Propagation
 *
 * @details Computes adaptive learning rates for each parameter. Has an EWMA of squared gradients.
 */
class RMSProp final : public StochOptim {
public:
    /**
     * @brief RMSProp constructor
     * @param starting_parameters Starting parameters
     * @param starting_variance Starting variances
     * @param learning_rate Learning rate
     * @param ewma Exponentially-Weighted Moving Average
     */
    RMSProp(const Eigen::VectorXd& starting_parameters, const Eigen::VectorXd& starting_variances, double learning_rate,
            double ewma);

    /**
     * @brief Update variances and parameters given a vector of gradients
     * @param gradient Vector of gradients
     * @return Vector of updated parameters
     */
    Eigen::VectorXd update(Eigen::VectorXd& gradient) override;

private:
    double _ewma; ///< Exponentially Weighted Moving Average
};

/**
 * @class ADAM stoch_optim.hpp
 * @brief Adaptive Moment Estimation
 *
 * @details Computes adaptive learning rates for each parameter. Has an EWMA of past gradients and squared gradients.
 */
class ADAM final : public StochOptim {
public:
    /**
     * @brief ADAM constructor
     * @param starting_parameters Starting parameters
     * @param starting_variance Starting variances
     * @param learning_rate Learning rate
     * @param ewma1 Exponentially Weighted Moving Average
     * @param ewma2 Exponentially Weighted Moving Average
     */
    ADAM(const Eigen::VectorXd& starting_parameters, const Eigen::VectorXd& starting_variances, double learning_rate,
         double ewma1, double ewma2);

    /**
     * @brief Update variances and parameters given a vector of gradients
     * @param gradient Vector of gradients
     * @return Vector of updated parameters
     */
    Eigen::VectorXd update(Eigen::VectorXd& gradient) override;

private:
    Eigen::VectorXd _f_gradient; ///<
    double _ewma_1;              ///< Exponentially Weighted Moving Average
    double _ewma_2;              ///< Exponentially Weighted Moving Average
};
