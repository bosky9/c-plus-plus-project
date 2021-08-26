#pragma once

#include "families/normal.hpp"
#include "headers.hpp"
#include "inference/bbvi_routines.hpp"
#include "inference/stoch_optim.hpp"
#include "multivariate_normal.hpp"

struct BBVIReturnData {
    std::vector<Normal> q;
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
private:
    std::function<double(Eigen::VectorXd)> _neg_posterior; ///< Posterior function
    std::vector<Normal> _q;                                ///< List holding the distribution objects
    int _sims;                                             ///< Number of Monte Carlo sims for the gradient
    Eigen::VectorXd _approx_param_no;                      ///<
    bool _printer;                                         ///<
    int _iterations;                                       ///< How many iterations to run
    bool _record_elbo;                                     ///< Whether to record the ELBO at every iteration
    bool _quiet_progress;                                  ///< Whether to print progress or stay quiet
    std::string _optimizer;                                ///<
    double _learning_rate;                                 ///<
    // @TODO: chiedere a Busato (e considera unique pointer)
    StochOptim* _optim = new StochOptim(Eigen::Vector<double, 1>{3.0}, 0, 0); ///<

public:
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
    BBVI(std::function<double(Eigen::VectorXd)> neg_posterior, std::vector<Normal>& q, int sims,
         std::string optimizer = "RMSProp", int iterations = 1000, double learning_rate = 0.001,
         bool record_elbo = false, bool quiet_progress = false);


    virtual ~BBVI();

    /**
     * @brief Utility function for changing the approximate distribution parameters
     * @param params
     */
    void change_parameters(std::vector<double>& params);

    /**
     * @brief Create logq components for mean-field normal family (the entropy estimate)
     * @param z
     * @return
     */
    double create_normal_logq(Eigen::VectorXd& z);

    /**
     * @brief Obtains an array with the current parameters
     * @return An array of parameters
     */
    std::vector<double> current_parameters();

    /**
     * @brief The control variate augmented Monte Carlo gradient estimate
     * @param z
     * @return
     */
    virtual Eigen::VectorXd cv_gradient(Eigen::MatrixXd& z, bool initial = false);

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
     * @brief Gets the mean and scales for normal approximating parameters
     * @return
     */
    std::pair<Eigen::VectorXd, Eigen::VectorXd> get_means_and_scales_from_q();

    /**
     * @brief Gets the mean and scales for normal approximating parameters
     * @return
     */
    std::pair<Eigen::VectorXd, Eigen::VectorXd> get_means_and_scales();

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
    virtual Eigen::VectorXd log_p(Eigen::MatrixXd& z);

    /**
     * @brief The mean-field normal log posterior components (the quantity we want to approximate)
     * @param z
     * @return
     */
    virtual Eigen::VectorXd normal_log_q(Eigen::MatrixXd& z, bool initial = false);

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
     * @return
     */
    virtual BBVIReturnData run();

    /**
     * @brief The core BBVI routine - Draws Monte Carlo gradients and uses a stochastic optimizer
     * Stores rgw history of updates for the benefit of a pretty animation.
     * @return
     */
    virtual BBVIReturnData run_and_store();

    std::vector<Normal> get_q() const;
};

class CBBVI : BBVI {
private:
    std::function<double(double)> _log_p_blanket;

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
    CBBVI(std::function<double(Eigen::VectorXd)> neg_posterior, std::function<double(double)> log_p_blanket, int sims,
          std::string optimizer = "RMSProp", int iterations = 300000, double learning_rate = 0.001,
          bool record_elbo = false, bool quiet_progress = false);

    /**
     * @brief Returns the unnormalized log posterior components (the quantity we want to approximate)
     * @param z
     * @return
     */
    Eigen::VectorXd log_p(Eigen::VectorXd& z) override;

    /**
     * @brief The unnormalized log posterior components for mean-field normal family (the quantity we want to
     * approximate)
     * @param z
     * @return
     */
    Eigen::VectorXd normal_log_q(Eigen::VectorXd& z, bool initial = false) override;

    /**
     * @brief The control variate augmented Monte Carlo gradient estimate
     * @param z
     * @return
     */
    double cv_gradient(Eigen::MatrixXd& z, bool initial = false) override;
};

/**
 * @brief Black Box Variational Inference - Minibatch
 */
class BBVIM : BBVI {
private:
    std::function<double(Eigen::VectorXd)> _full_neg_posterior;
    std::vector<int> _approx_param_no;
    bool _printer;
    int _mini_batch;

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
    BBVIM(std::function<double(Eigen::VectorXd)> neg_posterior,
          std::function<double(Eigen::VectorXd)> full_neg_posterior, int sims, std::string optimizer = "RMSProp",
          int iterations = 1000, double learning_rate = 0.001, int mini_batch = 2, bool record_elbo = false,
          bool quiet_progress = false);

    /**
     * @brief The unnormalized log posterior components (the quantity we want to approximate)
     * @param z
     * @return
     */
    Eigen::VectorXd log_p(Eigen::VectorXd& z) override;

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
     * @return
     */
    BBVIReturnData run() override;

    /**
     * @brief The core BBVI routine - Draws Monte Carlo gradients and uses a stochastic optimizer
     * Stores rgw history of updates for the benefit of a pretty animation.
     * @return
     */
    BBVIReturnData run_and_store() override;
};