#pragma once

#include "latent_variables.hpp"
#include "output/tableprinter.hpp"
#include "tests/nhst.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

class Results {
protected:
    std::vector<std::string> _x_names;
    std::string _model_name;
    std::string _model_type;
    LatentVariables _z;
    Eigen::VectorXd _z_values;
    Eigen::VectorXd _results; // FIXME: OptimizeResult type in Python (da scipy) ma viene utilizzato solo l'array
                              // non gli altri oggetti al suo interno
    Eigen::MatrixXd _data;    ///< Predicted values for the time series and length-adjusted time series
    std::vector<size_t> _index;
    bool _multivariate_model;
    std::function<double(Eigen::VectorXd)> _objective_object; ///< Likelihood or posterior
    std::string _method;
    uint8_t _z_hide;
    int _max_lag;
    Eigen::MatrixXd _ihessian;
    Eigen::VectorXd _signal;
    // FIXME: _scores, _states e _states_var non sono mai usate con i modelli ARMA
    Eigen::VectorXd _scores;
    Eigen::VectorXd _states;
    Eigen::VectorXd _states_var;
    size_t _data_length;
    std::string _data_name;
    uint8_t _rounding_points;
    double _aic;
    double _bic;

    /**
     * @brief Constructor for Results
     * @param data_name
     * @param X_names
     * @param model_name
     * @param model_type
     * @param latent_variables
     * @param results
     * @param data
     * @param index
     * @param multivariate_model
     * @param objective_object
     * @param method
     * @param z_hide
     * @param max_lag
     * @param ihessian
     * @param scores
     * @param states
     * @param states_var
     */
    Results(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
            const std::string& model_type, const LatentVariables& latent_variables, Eigen::VectorXd results,
            Eigen::MatrixXd data, std::vector<size_t> index, bool multivariate_model,
            std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide, int max_lag,
            Eigen::MatrixXd ihessian = Eigen::MatrixXd::Zero(0, 0), Eigen::VectorXd signal = Eigen::VectorXd::Zero(0),
            Eigen::VectorXd scores = Eigen::VectorXd::Zero(0), Eigen::VectorXd states = Eigen::VectorXd::Zero(0),
            Eigen::VectorXd states_var = Eigen::VectorXd::Zero(0));

    [[nodiscard]] static double round_to(double x, uint8_t rounding_points);

public:
    virtual void summary(bool transformed) = 0;
};

class MLEResults : Results {
private:
    double _loglik;

    /**
     * @brief Prints results with hessian
     * @param transformed
     */
    void summary_with_hessian(bool transformed = true) const;

    /**
     * @brief Prints results without hessian
     */
    void summary_without_hessian() const;

public:
    /**
     * @brief Constructor for MLEResults
     * @param data_name
     * @param X_names
     * @param model_name
     * @param model_type
     * @param latent_variables
     * @param results
     * @param data
     * @param index
     * @param multivariate_model
     * @param objective_object
     * @param method
     * @param z_hide
     * @param max_lag
     * @param ihessian
     * @param scores
     * @param states
     * @param states_var
     */
    MLEResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
               const std::string& model_type, const LatentVariables& latent_variables, Eigen::VectorXd results,
               Eigen::MatrixXd data, std::vector<size_t> index, bool multivariate_model,
               std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide, int max_lag,
               Eigen::MatrixXd ihessian = Eigen::VectorXd::Zero(0), Eigen::VectorXd signal = Eigen::VectorXd::Zero(0),
               Eigen::VectorXd scores = Eigen::VectorXd::Zero(0), Eigen::VectorXd states = Eigen::VectorXd::Zero(0),
               Eigen::VectorXd states_var = Eigen::VectorXd::Zero(0));

    /**
     * @brief Stream operator for MLEResults
     * @param stream Output stream
     * @param mleresults MLEResults object
     * @return Output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const MLEResults& mleresults);

    /**
     * @brief Prints results
     * @param transformed
     */
    void summary(bool transformed) override;
};

class BBVIResults : Results {
private:
    Eigen::VectorXd _elbo_records;
    Eigen::VectorXd _ses;
    Eigen::MatrixXd _chains;

public:
    /**
     * @brief Constructor for BBVIResults
     * @param data_name
     * @param X_names
     * @param model_name
     * @param model_type
     * @param latent_variables
     * @param data
     * @param index
     * @param multivariate_model
     * @param objective_object
     * @param method
     * @param z_hide
     * @param max_lag
     * @param ses
     * @param signal
     * @param scores
     * @param elbo_records
     * @param states
     * @param states_var
     */
    BBVIResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                const std::string& model_type, LatentVariables latent_variables, Eigen::MatrixXd data, int index,
                bool multivariate_model, std::function<double(Eigen::VectorXd)> objective_object, std::string method,
                bool z_hide, int max_lag, Eigen::VectorXd ses, Eigen::VectorXd signal = Eigen::VectorXd::Zero(0),
                Eigen::VectorXd scores       = Eigen::VectorXd::Zero(0),
                Eigen::VectorXd elbo_records = Eigen::VectorXd::Zero(0),
                Eigen::VectorXd states       = Eigen::VectorXd::Zero(0),
                Eigen::VectorXd states_var   = Eigen::VectorXd::Zero(0));

    /**
     * @brief Stream operator for BBVIResults
     * @param stream Output stream
     * @param results BBVIResults object
     * @return Output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const BBVIResults& results);

    /**
     * @brief Plots the ELBO progress (if present)
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_elbo(size_t width = 15, size_t height = 7) const;

    /**
     * @brief Prints results
     * @param transformed
     */
    void summary(bool transformed) override;
};

class BBVISSResults : Results {
private:
    Eigen::VectorXd _elbo_records;
    Eigen::VectorXd _ses;
    Eigen::MatrixXd _chains;

public:
    /**
     * @brief Constructor for BBVISSResults
     * @param data_name
     * @param X_names
     * @param model_name
     * @param model_type
     * @param latent_variables
     * @param data
     * @param index
     * @param multivariate_model
     * @param objective_object
     * @param method
     * @param z_hide
     * @param max_lag
     * @param ses
     * @param signal
     * @param scores
     * @param elbo_records
     * @param states
     * @param states_var
     */
    BBVISSResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                  const std::string& model_type, LatentVariables latent_variables, Eigen::MatrixXd data, int index,
                  bool multivariate_model, std::function<double(Eigen::VectorXd)> objective_object, std::string method,
                  bool z_hide, int max_lag, Eigen::VectorXd ses, Eigen::VectorXd signal = Eigen::VectorXd::Zero(0),
                  Eigen::VectorXd scores       = Eigen::VectorXd::Zero(0),
                  Eigen::VectorXd elbo_records = Eigen::VectorXd::Zero(0),
                  Eigen::VectorXd states       = Eigen::VectorXd::Zero(0),
                  Eigen::VectorXd states_var   = Eigen::VectorXd::Zero(0));

    /**
     * @brief Stream operator for BBVISSResults
     * @param stream Output stream
     * @param results BBVISSResults object
     * @return Output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const BBVISSResults& results);

    /**
     * @brief Plots the ELBO progress (if present)
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_elbo(size_t width = 15, size_t height = 7) const;

    /**
     * @brief Prints results
     * @param transformed
     */
    void summary(bool transformed) override;
};

class LaplaceResults : Results {
private:
    Eigen::MatrixXd _chains;

public:
    /**
     * Constructor for LaplaceResults
     * @param data_name
     * @param X_names
     * @param model_name
     * @param model_type
     * @param latent_variables
     * @param data
     * @param index
     * @param multivariate_model
     * @param objective_object
     * @param method
     * @param z_hide
     * @param max_lag
     * @param ihessian
     * @param signal
     * @param scores
     * @param states
     * @param states_var
     */
    LaplaceResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                   const std::string& model_type, LatentVariables latent_variables, Eigen::MatrixXd data, int index,
                   bool multivariate_model, std::function<double(Eigen::VectorXd)> objective_object, std::string method,
                   bool z_hide, int max_lag, Eigen::MatrixXd ihessian,
                   Eigen::VectorXd signal = Eigen::VectorXd::Zero(0), Eigen::VectorXd scores = Eigen::VectorXd::Zero(0),
                   Eigen::VectorXd states     = Eigen::VectorXd::Zero(0),
                   Eigen::VectorXd states_var = Eigen::VectorXd::Zero(0));

    /**
     * @brief Stream operator for LaplaceResults
     * @param stream Output stream
     * @param results LaplaceResults object
     * @return Output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const LaplaceResults& results);

    /**
     * @brief Prints results
     * @param transformed
     */
    void summary(bool transformed) override;
};

class MCMCResults : Results {
private:
    Eigen::MatrixXd _samples;
    Eigen::VectorXd _mean_est;
    Eigen::VectorXd _median_est;
    Eigen::VectorXd _lower_95_est;
    Eigen::VectorXd _upper_95_est;

public:
    /**
     * @brief Constructor for MCMCResults
     * @param data_name
     * @param X_names
     * @param model_name
     * @param model_type
     * @param latent_variables
     * @param data
     * @param index
     * @param multivariate_model
     * @param objective_object
     * @param method
     * @param z_hide
     * @param max_lag
     * @param samples
     * @param mean_est
     * @param median_est
     * @param lower_95_est
     * @param upper_95_est
     * @param signal
     * @param scores
     * @param states
     * @param states_var
     */
    MCMCResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                const std::string& model_type, LatentVariables latent_variables, Eigen::MatrixXd data, int index,
                bool multivariate_model, std::function<double(Eigen::VectorXd)> objective_object, std::string method,
                bool z_hide, int max_lag, Eigen::MatrixXd samples, Eigen::VectorXd mean_est, Eigen::VectorXd median_est,
                Eigen::VectorXd lower_95_est, Eigen::VectorXd upper_95_est,
                Eigen::VectorXd signal = Eigen::VectorXd::Zero(0), Eigen::VectorXd scores = Eigen::VectorXd::Zero(0),
                Eigen::VectorXd states     = Eigen::VectorXd::Zero(0),
                Eigen::VectorXd states_var = Eigen::VectorXd::Zero(0));

    /**
     * @brief Stream operator for MCMCResults
     * @param stream Output stream
     * @param results MCMCResults object
     * @return Output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const MCMCResults& results);

    /**
     * @brief Prints results
     * @param transformed
     */
    void summary(bool transformed) override;
};