/**
 * @file bbvi.hpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#pragma once

#include "Eigen/Core"                // Eigen::VectorXd, Eigen::MatrixXd
#include "families/family.hpp"       // Family
#include "inference/stoch_optim.hpp" // StochOptim

#include <memory> // std::unique_ptr, std::shared_ptr
#include <optional>

/**
 * @struct BBVIReturnData bbvi.hpp
 * @brief Data structure return by run in BBVI class
 */
struct BBVIReturnData {
    std::vector<std::unique_ptr<Family>> q;
    Eigen::VectorXd final_means;
    Eigen::VectorXd final_ses;
    Eigen::VectorXd elbo_records;
    Eigen::MatrixXd stored_means                 = Eigen::MatrixXd();
    Eigen::VectorXd stored_predictive_likelihood = Eigen::MatrixXd();
};

/**
 * @class BBVI bbvi.hpp
 * @brief Black Box Variational Inference class
 */
class BBVI {
public:
    /**
     * @brief Constructor for BBVI
     * @param neg_posterior Posterior function
     * @param q List holding distribution objects
     * @param sims Number of Monte Carlo sims for the gradient
     * @param optimizer Name of the optmizer
     * @param iterations How many iterations to run
     * @param learning_rate Learning rate
     * @param record_elbo Wheter to record the ELBO at every iteration
     * @param quiet_progress Wheter to print progress or stay quiet
     */
    BBVI(std::function<double(const Eigen::VectorXd&, std::optional<size_t>)> neg_posterior,
         const std::vector<std::unique_ptr<Family>>& q, size_t sims, std::string optimizer = "RMSProp",
         size_t iterations = 1000, double learning_rate = 0.001, bool record_elbo = false, bool quiet_progress = false);

    /**
     * @brief Copy constructor for BBVI
     * @param bbvi The BBVI object
     */
    BBVI(const BBVI& bbvi);

    /**
     * @brief Move constructor for BBVI
     * @param bbvi A BBVI object
     */
    BBVI(BBVI&& bbvi) noexcept;

    /**
     * @brief Assignment operator for BBVI
     * @param bbvi A BBVI object
     */
    BBVI& operator=(const BBVI& bbvi);

    /**
     * @brief Move assignment operator for BBVI
     * @param bbvi A BBVI object
     */
    BBVI& operator=(BBVI&& bbvi) noexcept;

    /**
     * @brief Destructor for BBVI
     */
    virtual ~BBVI();

    /**
     * @brief Equality operator for BBVI
     * @param bbvi1 First BBVI object
     * @param bbvi2 Second BBVI object
     * @return If the two objects are equal
     */
    friend bool operator==(const BBVI& bbvi1, const BBVI& bbvi2);

    /**
     * @brief Utility function for changing the approximate distribution parameters
     * @param params Vector of parameters to change to (i.e., mean and distribution)
     *
     * @details Please notice that the function wil iterate over the number of parameters
     *          of the distribution, changing every one of them.
     */
    void change_parameters(const Eigen::VectorXd& params);

    /**
     * @brief Create logq components for mean-field normal family (the entropy estimate)
     * @param z Vector of variables
     * @return The sum over all the logpdf of the z variables.
     *
     * @details Each z variable has its own mean and variance.
     */
    [[nodiscard]] double create_normal_logq(const Eigen::VectorXd& z) const;

    /**
     * @brief Obtains an array with the current parameters
     * @return A vector of parameters
     *
     * @details Inside the function, values are appended to a std::vector, which is then converted to a Eigen::VectorXd,
     *          by means of the Map(pointer, size) function.
     */
    [[nodiscard]] Eigen::VectorXd current_parameters() const;

    /**
     * @brief Return the control variate augmented Monte Carlo gradient estimate
     * @param z Vector of variables
     * @param initial cv_gradient or cv_gradient_initial
     * @return Vector of gradients
     *
     * @details In Python, there are two identical functions, differing only on the log_q initializations.
     *          Using the "initial" parameter allows to write a single function.
     */
    [[nodiscard]] virtual Eigen::VectorXd cv_gradient(const Eigen::MatrixXd& z, bool initial) const;

    /**
     * @brief Draw parameters from a mean-field normal family
     * @return Matrix of parameters
     */
    [[nodiscard]] Eigen::MatrixXd draw_normal(bool initial = false) const;

    /**
     * @brief Draw parameters from the approximating distributions
     * @return Matrix of parameters
     */
    [[nodiscard]] Eigen::MatrixXd draw_variables() const;

    /**
     * @brief Gets the mean and scales for normal approximating parameters
     * @return Vectors of means and scales
     */
    [[nodiscard]] std::pair<Eigen::VectorXd, Eigen::VectorXd> get_means_and_scales_from_q() const;

    /**
     * @brief Gets the mean and scales for normal approximating parameters
     * @return Vectors of means and scales
     */
    [[nodiscard]] std::pair<Eigen::VectorXd, Eigen::VectorXd> get_means_and_scales() const;

    /**
     * @brief Returns the gradients of the approximating distributions
     * @param z Matrix of variables
     * @return Matrix of gradients
     */
    [[nodiscard]] Eigen::MatrixXd grad_log_q(const Eigen::MatrixXd& z) const;

    /**
     * @brief Returns the unnormalized log posterior components (the quantity we want to approximate)
     * @param z Matrix of variables
     * @return Unnormalized log posterior components
     */
    [[nodiscard]] virtual Eigen::MatrixXd log_p(const Eigen::MatrixXd& z) const;

    /**
     * @brief Returns the mean-field normal log posterior components (the quantity we want to approximate)
     * @param z Matrix of variables
     * @return Mean-field normal log posterior components
     */
    [[nodiscard]] virtual Eigen::MatrixXd normal_log_q(const Eigen::MatrixXd& z, bool initial) const;

    /**
     * @brief Prints the current ELBO at every decile of total iterations
     * @param i Iteration
     * @param current_params Current set of parameters
     */
    virtual void print_progress(size_t i, const Eigen::VectorXd& current_params) const;

    /**
     * @brief Obtains the ELBO for the current set of parameters
     * @param current_params Current set of parameters
     * @return ELBO
     */
    [[nodiscard]] virtual double get_elbo(const Eigen::VectorXd& current_params) const;

    /**
     * @brief The core BBVI routine - Draws Monte Carlo gradients and uses a stochastic optimizer
     * @param store If true, stores rgw history of updates for the benefit of a pretty animation
     * @return Results in BBVIReturnData structure
     * @details Using store, this function translates both "run" and "run_and_store".
     *          This "run" actually calls another function, run_with; this is to avoid troubles with the optional
     *          parameters of neg_posterior.
     */
    virtual BBVIReturnData run(bool store);

protected:
    std::function<double(const Eigen::VectorXd&, std::optional<size_t>)> _neg_posterior; ///< Posterior function
    size_t _sims;                                ///< Number of Monte Carlo sims for the gradient
    bool _printer;                               ///< True if printing is enabled
    std::string _optimizer;                      ///< Name of the optimizer
    size_t _iterations;                          ///< How many iterations to run
    double _learning_rate;                       ///< Learning rate
    bool _record_elbo;                           ///< Whether to record the ELBO at every iteration
    bool _quiet_progress;                        ///< Whether to print progress or stay quiet
    std::vector<std::unique_ptr<Family>> _q;     ///< List holding the distribution objects
    Eigen::VectorXd _approx_param_no;            ///< Number of parameters for approximations
    std::unique_ptr<StochOptim> _optim{nullptr}; ///< Optimizer

    /**
     * @brief Internal method called by run with selected negative posterior function
     * @param store If true, stores raw history of updates for the benefit of a pretty animation
     * @param neg_posterior Negative posterior function
     * @return Results in BBVIReturnData structure
     */
    BBVIReturnData run_with(bool store, const std::function<double(const Eigen::VectorXd&)>& neg_posterior);
};

/**
 * @class CBBVI bbvi.hpp
 * @brief Black Box Variational Inference
 */
class CBBVI final : public BBVI {
public:
    /**
     * @brief Constructor for CBBVI
     * @param neg_posterior Posterior function
     * @param log_p_blanket Log posterior with Bayesian Markov Blanket estimation
     * @param sims Number of Monte Carlo sims for the gradient
     * @param optimizer Name of the optmizer
     * @param iterations How many iterations to run
     * @param learning_rate Learning rate
     * @param record_elbo Wheter to record the ELBO at every iteration
     * @param quiet_progress Wheter to print progress or stay quiet
     */
    CBBVI(const std::function<double(const Eigen::VectorXd&, std::optional<size_t>)>& neg_posterior,
          std::function<Eigen::VectorXd(const Eigen::VectorXd&)> log_p_blanket, std::vector<std::unique_ptr<Family>>& q,
          size_t sims, const std::string& optimizer = "RMSProp", size_t iterations = 300000,
          double learning_rate = 0.001, bool record_elbo = false, bool quiet_progress = false);

    /**
     * @brief Copy constructor for CBBVI
     * @param cbbvi The CBBVI object
     */
    CBBVI(const CBBVI& cbbvi);

    /**
     * @brief Move constructor for CBBVI
     * @param cbbvi A CBBVI object
     */
    CBBVI(CBBVI&& cbbvi) noexcept;

    /**
     * @brief Assignment operator for CBBVI
     * @param cbbvi A CBBVI object
     */
    CBBVI& operator=(const CBBVI& cbbvi);

    /**
     * @brief Move assignment operator for CBBVI
     * @param cbbvi A CBBVI object
     */
    CBBVI& operator=(CBBVI&& cbbvi) noexcept;

    /**
     * @brief Equality operator for CBBVI
     * @param cbbvi1 First CBBVI object
     * @param cbbvi2 Second CBBVI object
     * @return If the two objects are equal
     */
    friend bool operator==(const CBBVI& cbbvi1, const CBBVI& cbbvi2);

    /**
     * @brief Returns the unnormalized log posterior components (the quantity we want to approximate)
     * @param z Matrix of variables
     * @return Unnormalized log posterior components
     */
    [[nodiscard]] Eigen::MatrixXd log_p(const Eigen::MatrixXd& z) const override;

    /**
     * @brief Returns the mean-field normal log posterior components (the quantity we want to approximate)
     * @param z Matrix of variables
     * @return Mean-field normal log posterior components
     */
    [[nodiscard]] Eigen::MatrixXd normal_log_q(const Eigen::MatrixXd& z, bool initial) const override;

    /**
     * @brief Return the control variate augmented Monte Carlo gradient estimate
     * @param z Vector of variables
     * @param initial cv_gradient or cv_gradient_initial
     * @return Vector of gradients
     */
    [[nodiscard]] Eigen::VectorXd cv_gradient(const Eigen::MatrixXd& z, bool initial) const override;

private:
    std::function<Eigen::VectorXd(Eigen::VectorXd)>
            _log_p_blanket; ///< Computes the log posterior with Bayesian Markov Blanket estimation
};

/**
 * @class BBVIM bbvi.hpp
 * @brief Black Box Variational Inference with mini batches
 */
class BBVIM final : public BBVI {
public:
    /**
     * @brief Contructor for BBCVIM
     * @param neg_posterior Posterior function
     * @param full_neg_posterior Posterior function
     * @param sims Number of Monte Carlo sims for the gradient
     * @param optimizer Name of the optimizer
     * @param iterations How many iterations to run
     * @param learning_rate Learning rate
     * @param mini_batch Mini batch size
     * @param record_elbo Whether to record the ELBO
     * @param quiet_progress Whether to print progress or stay quiet
     */
    BBVIM(const std::function<double(const Eigen::VectorXd&, std::optional<size_t>)>& neg_posterior,
          std::function<double(const Eigen::VectorXd&)> full_neg_posterior,
          const std::vector<std::unique_ptr<Family>>& q, size_t sims, const std::string& optimizer = "RMSProp",
          size_t iterations = 1000, double learning_rate = 0.001, size_t mini_batch = 2, bool record_elbo = false,
          bool quiet_progress = false);

    /**
     * @brief Copy constructor for BBVIM
     * @param bbvim The BBVIM object
     */
    BBVIM(const BBVIM& bbvim);

    /**
     * @brief Move constructor for BBVIM
     * @param bbvim A BBVIM object
     */
    BBVIM(BBVIM&& bbvim) noexcept;

    /**
     * @brief Assignment operator for BBVIM
     * @param bbvim A BBVIM object
     */
    BBVIM& operator=(const BBVIM& bbvim);

    /**
     * @brief Move assignment operator for BBVIM
     * @param bbvim A BBVIM object
     */
    BBVIM& operator=(BBVIM&& bbvim) noexcept;

    /**
     * @brief Equality operator for BBVIM
     * @param bbvim1 First BBVIM object
     * @param bbvim2 Second BBVIM object
     * @return If the two objects are equal
     */
    friend bool operator==(const BBVIM& bbvim1, const BBVIM& bbvim2);

    /**
     * @brief Returns the unnormalized log posterior components (the quantity we want to approximate)
     * @param z Matrix of variables
     * @return Unnormalized log posterior components
     */
    [[nodiscard]] Eigen::MatrixXd log_p(const Eigen::MatrixXd& z) const override;

    /**
     * @brief Obtains the ELBO for the current set of parameters
     * @param current_params Current set of parameters
     * @return ELBO
     */
    [[nodiscard]] double get_elbo(const Eigen::VectorXd& current_params) const override;

    /**
     * @brief Prints the current ELBO at every decile of total iterations
     * @param i Iteration
     * @param current_params Current set of parameters
     */
    void print_progress(size_t i, const Eigen::VectorXd& current_params) const override;

    /**
     * @brief The core BBVI routine - Draws Monte Carlo gradients and uses a stochastic optimizer
     * @param store If true, stores rgw history of updates for the benefit of a pretty animation
     * @return Results in BBVIReturnData structure
     */
    BBVIReturnData run(bool store) override;

private:
    std::function<double(const Eigen::VectorXd&)> _full_neg_posterior; ///< Posterior function
    size_t _mini_batch;                                                ///< Number of mini batches
};