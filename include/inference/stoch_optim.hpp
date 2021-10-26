#pragma once

#include "headers.hpp"

// If defined as a static it gives multiple definition error
// const double EPSILON = 1e-7;

class StochOptim {
protected:
    Eigen::VectorXd _parameters;
    Eigen::VectorXd _variances;
    double _learning_rate;
    static double _epsilon;
    int _t = 1;

public:
    StochOptim(const Eigen::VectorXd& starting_parameters, const Eigen::VectorXd& starting_variances,
               double learning_rate);

    /**
     * @brief Copy constructor for StochOptim
     * @param stochOptim The StochOptim object
     */
    StochOptim(const StochOptim& stochOptim);

    /**
     * @brief Move constructor for StochOptim
     * @param stochOptim A StochOptim object
     */
    StochOptim(StochOptim&& stochOptim) noexcept;

    /**
     * @brief Assignment operator for StochOptim
     * @param stochOptim A StochOptim object
     */
    StochOptim& operator=(const StochOptim& stochOptim);

    /**
     * @brief Move assignment operator for StochOptim
     * @param stochOptim A StochOptim object
     */
    StochOptim& operator=(StochOptim&& stochOptim) noexcept;

    /**
     * @brief
     * @param gradient
     * @return
     */
    virtual Eigen::VectorXd update(Eigen::VectorXd& gradient);

    /**
     * @brief
     * @return
     */
    [[nodiscard]] Eigen::VectorXd get_parameters() const;

    /**
     * @brief Get the name of the optimizer
     * @return Optimizer's name
     */
    virtual std::string get_name() const;
};

/**
 * @brief Computes adaptive learning rates for each parameter. Has an EWMA of squared gradients.
 */
class RMSProp final : public StochOptim {
private:
    double _ewma;

public:
    /**
     * @brief RMSProp constructor
     * @param starting_parameters
     * @param starting_variance
     * @param learning_rate
     * @param ewma Exponentially-Weighted Moving Average
     */
    RMSProp(const Eigen::VectorXd& starting_parameters, const Eigen::VectorXd& starting_variances, double learning_rate,
            double ewma);

    /**
     * @brief Copy constructor for RMSProp
     * @param rmsprop The RMSProp object
     */
    RMSProp(const RMSProp& rmsprop);

    /**
     * @brief Move constructor for RMSProp
     * @param rmsprop A RMSProp object
     */
    RMSProp(RMSProp&& rmsprop) noexcept;

    /**
     * @brief Assignment operator for RMSProp
     * @param rmsprop A RMSProp object
     */
    RMSProp& operator=(const RMSProp& rmsprop);

    /**
     * @brief Move assignment operator for RMSProp
     * @param rmsprop A RMSProp object
     */
    RMSProp& operator=(RMSProp&& rmsprop) noexcept;

    /**
     * @brief
     * @param gradient
     * @return
     */
    Eigen::VectorXd update(Eigen::VectorXd& gradient) override;

    /**
     * @brief Get the name of the optimizer
     * @return Optimizer's name
     */
    std::string get_name() const override;
};

/**
 * @brief Adaptive Moment Estimation.
 * @brief Computes adaptive learning rates for each parameter.
 * @brief Has an EWMA of past gradients and squared gradients.
 */
class ADAM final : public StochOptim {
private:
    Eigen::VectorXd _f_gradient;
    double _ewma_1;
    double _ewma_2;

public:
    /**
     * @brief ADAM constructor
     * @param starting_parameters
     * @param starting_variance
     * @param learning_rate
     * @param ewma1 Exponentially-Weighted Moving Average
     * @param ewma2 Exponentially-Weighted Moving Average
     */
    ADAM(const Eigen::VectorXd& starting_parameters, const Eigen::VectorXd& starting_variances, double learning_rate,
         double ewma1, double ewma2);

    /**
     * @brief Copy constructor for ADAM
     * @param adam The ADAM object
     */
    ADAM(const ADAM& adam);

    /**
     * @brief Move constructor for ADAM
     * @param adam A ADAM object
     */
    ADAM(ADAM&& adam) noexcept;

    /**
     * @brief Assignment operator for ADAM
     * @param adam A ADAM object
     */
    ADAM& operator=(const ADAM& adam);

    /**
     * @brief Move assignment operator for ADAM
     * @param adam A ADAM object
     */
    ADAM& operator=(ADAM&& adam) noexcept;

    /**
     * @brief
     * @param gradient
     * @return
     */
    Eigen::VectorXd update(Eigen::VectorXd& gradient) override;

    /**
     * @brief Get the name of the optimizer
     * @return Optimizer's name
     */
    std::string get_name() const override;
};

inline double StochOptim::_epsilon = pow(10, -8);