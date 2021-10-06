#pragma once

#include "families/normal.hpp"
#include "headers.hpp"
#include "inference/bbvi_routines.hpp"
#include "inference/stoch_optim.hpp"
#include "multivariate_normal.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

struct BBVIReturnData {
    std::vector<Normal*> q;
    Eigen::VectorXd final_means;
    Eigen::VectorXd final_ses;
    Eigen::VectorXd elbo_records;
    Eigen::MatrixXd stored_means                 = Eigen::MatrixXd();
    Eigen::VectorXd stored_predictive_likelihood = Eigen::MatrixXd();
};

/**
 * @brief Black Box Variational Inference
 */
class BBVI {
protected:
    std::function<double(Eigen::VectorXd, std::optional<size_t>)> _neg_posterior; ///< Posterior function
    std::vector<Normal*> _q;          ///< List holding the distribution objects
    size_t _sims;                     ///< Number of Monte Carlo sims for the gradient
    bool _printer;                    ///<
    std::string _optimizer;           ///<
    size_t _iterations;               ///< How many iterations to run
    double _learning_rate;            ///<
    bool _record_elbo;                ///< Whether to record the ELBO at every iteration
    bool _quiet_progress;             ///< Whether to print progress or stay quiet
    Eigen::VectorXd _approx_param_no; ///<

    /**
     * @brief Internal method called by run with selected negative posterior function
     * @param store If true, stores rgw history of updates for the benefit of a pretty animation
     * @param neg_posterior Negative posterior function
     * @return
     */
    BBVIReturnData run_with(bool store, const std::function<double(Eigen::VectorXd)>& neg_posterior);

public:
    // @TODO: chiedere a Busato (e considera unique pointer)
    std::unique_ptr<StochOptim> _optim{nullptr}; ///<
    // new StochOptim(Eigen::Vector<double, 1>{3.0}, Eigen::Vector<double, 1>{1.0}, 0)

    /**
     * @brief Base constructor for BBVI
     */
    BBVI();

    /**
     * @brief Constructor for BBVI
     * @param neg_posterior Posterior function
     * @param q List holding distribution objects
     * @param sims Number of Monte Carlo sims for the gradient
     * @param optimizer
     * @param iterations How many iterations to run
     * @param learning_rate
     * @param record_elbo
     * @param quiet_progress
     */
    BBVI(std::function<double(Eigen::VectorXd, std::optional<size_t>)> neg_posterior, std::vector<Normal*>& q,
         size_t sims, std::string optimizer = "RMSProp", size_t iterations = 1000, double learning_rate = 0.001,
         bool record_elbo = false, bool quiet_progress = false);

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
     * @brief Equality operator for BBVI
     * @param bbvi1 First BBVI object
     * @param bbvi2 Second BBVI object
     * @return If the two objects are equal
     */
    friend bool operator==(const BBVI& bbvi1, const BBVI& bbvi2);

    /**
     * @brief Utility function for changing the approximate distribution parameters
     * @param params
     */
    void change_parameters(Eigen::VectorXd& params);

    /**
     * @brief Create logq components for mean-field normal family (the entropy estimate)
     * @param z
     * @return
     */
    double create_normal_logq(Eigen::VectorXd& z) const;

    /**
     * @brief Obtains an array with the current parameters
     * @return An array of parameters
     */
    [[nodiscard]] Eigen::VectorXd current_parameters() const;

    /**
     * @brief The control variate augmented Monte Carlo gradient estimate
     * @param z
     * @return
     */
    virtual Eigen::VectorXd cv_gradient(Eigen::MatrixXd& z, bool initial);

    /**
     * @brief Draw parameters from a mean-field normal family
     * @return
     */
    Eigen::MatrixXd draw_normal(bool initial = false);

    /**
     * @brief Draw parameters from the approximating distributions
     * @return
     */
    Eigen::MatrixXd draw_variables();

    /**
     * @brief Returns the number of parameters for every function (Normal) in q
     * @return Number of parameters
     */
    Eigen::VectorXd get_approx_param_no();

    /**
     * @brief Returns the number of iterations
     * @return Number of iterations
     */
    [[nodiscard]] size_t get_iterations() const;

    /**
     * @brief Returns the learning rate
     * @return Learning rate
     */
    [[nodiscard]] double get_learning_rate() const;

    /**
     * @brief Gets the mean and scales for normal approximating parameters
     * @return
     */
    std::pair<Eigen::VectorXd, Eigen::VectorXd> get_means_and_scales_from_q();

    /**
     * @brief Gets the mean and scales for normal approximating parameters
     * @return
     */
    [[nodiscard]] std::pair<Eigen::VectorXd, Eigen::VectorXd> get_means_and_scales() const;

    /**
     * @brief The gradients of the approximating distributions
     * @param z
     * @return
     */
    // In Python ritorna una matrice, ma poi assegna un double ad ogni riga (?)
    Eigen::MatrixXd grad_log_q(Eigen::MatrixXd& z);

    /**
     * @brief The unnormalized log posterior components (the quantity we want to approximate)
     * @param z
     * @return
     */
    virtual Eigen::MatrixXd log_p(Eigen::MatrixXd& z);

    /**
     * @brief The mean-field normal log posterior components (the quantity we want to approximate)
     * @param z
     * @return
     */
    virtual Eigen::MatrixXd normal_log_q(Eigen::MatrixXd& z, bool initial);

    /**
     * @brief Prints the current ELBO at every decile of total iterations
     * @param i
     * @param current_params
     */
    virtual void print_progress(double i, Eigen::VectorXd& current_params);

    /**
     * @brief Obtains the ELBO for the current set of parameters
     * @param current_params
     * @return
     */
    virtual double get_elbo(Eigen::VectorXd& current_params);

    /**
     * @brief The core BBVI routine - Draws Monte Carlo gradients and uses a stochastic optimizer
     * @param store If true, stores rgw history of updates for the benefit of a pretty animation
     * @return
     */
    virtual BBVIReturnData run(bool store);
};

class CBBVI final : public BBVI {
private:
    std::function<Eigen::VectorXd(Eigen::VectorXd)> _log_p_blanket;

public:
    /**
     * @brief Constructor for CBBVI
     * @param neg_posterior Posterior function
     * @param log_p_blanket
     * @param sims Number of Monte Carlo sims for the gradient
     * @param optimizer
     * @param iterations How many iterations to run
     * @param learning_rate
     * @param record_elbo
     * @param quiet_progress
     */
    CBBVI(std::function<double(Eigen::VectorXd, std::optional<size_t>)> neg_posterior,
          std::function<Eigen::VectorXd(Eigen::VectorXd)> log_p_blanket, std::vector<Normal*>& q, size_t sims,
          std::string optimizer = "RMSProp", size_t iterations = 300000, double learning_rate = 0.001,
          bool record_elbo = false, bool quiet_progress = false);

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
     * @param z
     * @return
     */
    Eigen::MatrixXd log_p(Eigen::MatrixXd& z) override;

    /**
     * @brief The unnormalized log posterior components for mean-field normal family (the quantity we want to
     * approximate)
     * @param z
     * @return
     */
    Eigen::MatrixXd normal_log_q(Eigen::MatrixXd& z, bool initial) override;

    /**
     * @brief The control variate augmented Monte Carlo gradient estimate
     * @param z
     * @return
     */
    Eigen::VectorXd cv_gradient(Eigen::MatrixXd& z, bool initial) override;
};

/**
 * @brief Black Box Variational Inference - Minibatch
 */
class BBVIM final : public BBVI {
private:
    std::function<double(Eigen::VectorXd)> _full_neg_posterior;
    size_t _mini_batch;

public:
    /**
     * @brief Contructor for BBCVIM
     * @param neg_posterior Posterior function
     * @param full_neg_posterior Posterior function
     * @param sims Number of Monte Carlo sims for the gradient
     * @param optimizer
     * @param iterations How many iterations to run
     * @param learning_rate
     * @param mini_batch Mini batch size
     * @param record_elbo
     * @param quiet_progress
     */
    BBVIM(std::function<double(Eigen::VectorXd, std::optional<size_t>)> neg_posterior,
          std::function<double(Eigen::VectorXd)> full_neg_posterior, std::vector<Normal*>& q, size_t sims,
          std::string optimizer = "RMSProp", size_t iterations = 1000, double learning_rate = 0.001,
          size_t mini_batch = 2, bool record_elbo = false, bool quiet_progress = false);

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
     * @brief The unnormalized log posterior components (the quantity we want to approximate)
     * @param z
     * @return
     */
    Eigen::MatrixXd log_p(Eigen::MatrixXd& z) override;

    /**
     * @brief Obtains the ELBO for the current set of parameters
     * @param current_params
     * @return
     */
    double get_elbo(Eigen::VectorXd& current_params) override;

    /**
     * @brief Prints the current ELBO at every decile of total iterations
     * @param i
     * @param current_params
     */
    void print_progress(double i, Eigen::VectorXd& current_params) override;

    /**
     * @brief The core BBVI routine - Draws Monte Carlo gradients and uses a stochastic optimizer
     * @param store If true, stores rgw history of updates for the benefit of a pretty animation
     * @return
     */
    BBVIReturnData run(bool store) override;
};