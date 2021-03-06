/**
 * @file results.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "results.hpp"

#include "Eigen/Core"                  // Eigen::Index, Eigen::VectorXd, Eigen::MatrixXd
#include "inference/norm_post_sim.hpp" // Sample, nps::norm_post_sim
#include "latent_variables.hpp"        // LatentVariables
#include "output/tableprinter.hpp"     // TablePrinter
#include "sciplot/sciplot.hpp"         // sciplot::Plot
#include "tests/nhst.hpp"              // find_p_value

#include <algorithm>  // std::transform
#include <cmath>      // pow, log
#include <functional> // std::function
#include <iostream>   // std::cout
#include <iterator>   // std::ostream_iterator
#include <list>       // std::list
#include <map>        // std::map
#include <memory>     // std::unique_ptr
#include <optional>   // std::optional
#include <ostream>    // std::ostream
#include <string>     // std::string, std::to_string
#include <tuple>      // std::tuple
#include <utility>    // std::move
#include <vector>     // std::vector

Results::Results(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                 const std::string& model_type, const LatentVariables& latent_variables, Eigen::MatrixXd data,
                 std::vector<double> index, bool multivariate_model,
                 std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide, int max_lag,
                 Eigen::VectorXd signal, std::optional<Eigen::VectorXd> scores, std::optional<Eigen::VectorXd> states,
                 std::optional<Eigen::VectorXd> states_var)
    : _x_names{std::move(X_names)}, _model_name{std::move(model_name)}, _model_type{model_type}, _z{latent_variables},
      _z_values{latent_variables.get_z_values()}, _data{std::move(data)}, _index{std::move(index)},
      _multivariate_model{multivariate_model}, _objective_object{std::move(objective_object)},
      _method{std::move(method)}, _z_hide{static_cast<uint8_t>(z_hide)}, _max_lag{max_lag}, _signal{std::move(signal)},
      _scores{std::move(scores)}, _states{std::move(states)}, _states_var{std::move(states_var)} {
    if (_multivariate_model) {
        _data_length = _data.row(0).size();
    } else {
        _data_length = _data.rows();
    }

    std::ostringstream oss;
    std::copy(data_name.begin(), data_name.end(), std::ostream_iterator<std::string>(oss, ","));
    _data_name = oss.str();

    if (_model_type == "LLT" || model_type == "LLEV")
        _rounding_points = 10;
    else
        _rounding_points = 4;
}

double Results::round_to(double x, uint8_t rounding_points) {
    const double shift = pow(10.0, rounding_points);
    return round(x * shift) / shift;
}

LatentVariables Results::get_z() const {
    return _z;
}

MLEResults::MLEResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                       const std::string& model_type, const LatentVariables& latent_variables, Eigen::VectorXd results,
                       Eigen::MatrixXd data, std::vector<double> index, bool multivariate_model,
                       std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide,
                       int max_lag, Eigen::MatrixXd ihessian, Eigen::VectorXd signal,
                       std::optional<Eigen::VectorXd> scores, std::optional<Eigen::VectorXd> states,
                       std::optional<Eigen::VectorXd> states_var)
    : Results{std::move(data_name),
              std::move(X_names),
              std::move(model_name),
              model_type,
              latent_variables,
              std::move(data),
              std::move(index),
              multivariate_model,
              std::move(objective_object),
              std::move(method),
              z_hide,
              max_lag,
              std::move(signal),
              std::move(scores),
              std::move(states),
              std::move(states_var)},
      _results{std::move(results)}, _ihessian{std::move(ihessian)} {
    if (_method == "MLE" || _method == "OLS") {
        _loglik = -_objective_object(_z_values);
        _aic    = 2 * static_cast<double>(_z_values.size()) + 2 * _objective_object(_z_values);
        _bic    = 2 * _objective_object(_z_values) +
               static_cast<double>(_z_values.size()) * log(static_cast<double>(_data_length));
    } else if (_method == "PML") {
        _aic = 2 * static_cast<double>(_z_values.size()) + 2 * _objective_object(_z_values);
        _bic = 2 * _objective_object(_z_values) +
               static_cast<double>(_z_values.size()) * log(static_cast<double>(_data_length));
    }
}

std::ostream& operator<<(std::ostream& stream, const MLEResults& results) {
    if (results._method == "MLE")
        stream << "MLE Results Object";
    else if (results._method == "OLS")
        stream << "OLS Results Object";
    else
        stream << "PML Results Object";
    stream << "\n=========================="
              "\nDependent variable: "
           << results._data_name << "\nRegressors: ";
    for (const std::string& s : results._x_names)
        stream << s << " ";
    stream << "\n=========================="
              "\nLatent Variable Attributes: ";
    if (results._ihessian.size() > 0)
        stream << "\n.ihessian: Inverse Hessian";
    stream << "\n.z : LatentVariables() object";
    if (results._results.size() > 0)
        stream << "\n.results : optimizer results";
    stream << "\n\nImplied Model Attributes: "
              "\n.aic: Akaike Information Criterion"
              "\n.bic: Bayesian Information Criterion"
              "\n.data: Model Data"
              "\n.index: Model Index";
    if (results._method == "MLE" || results._method == "OLS")
        stream << "\n.loglik: Loglikelihood";
    if (results._scores.has_value())
        stream << "\n.scores: Model Scores";
    if (results._signal.size() > 0)
        stream << "\n.signal: Model Signal";
    if (results._states.has_value())
        stream << "\n.states: Model States";
    if (results._states_var.has_value())
        stream << "\n.states_var: Model State Variances";
    stream << "\n.results : optimizer results"
              "\n\nMethods: "
              "\n.summary() : printed results";
    return stream;
}

void MLEResults::summary(bool transformed) {
    return (_ihessian.size() > 0) ? summary_with_hessian(transformed) : summary_without_hessian();
}

void MLEResults::summary_with_hessian(bool transformed) const {
    Eigen::VectorXd ses{_ihessian.diagonal().cwiseAbs().array().pow(0.5)};
    Eigen::VectorXd t_z{_z.get_z_values(true)};
    Eigen::VectorXd t_p_std{ses}; // Vector for transformed standard errors

    // Create transformed variables (commented also in Python)
    /*for (size_t k = 0; k < t_z.size() - _z_hide; k++) {
        double z_temp = _z_values[k] / ses[k];
        t_p_std[k]    = t_z[k] / z_temp;
    }*/

    std::list<std::map<std::string, std::string>> data;
    std::vector<std::string> z_names{_z.get_z_names()};
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(z_names.size() - _z_hide); ++i) {
        if (_z.get_z_list()[i].get_prior_clone()->get_transform_name().empty())
            data.push_back(
                    {{"z_name", z_names[i]},
                     {"z_value",
                      std::to_string(round_to(_z.get_z_list()[i].get_prior_clone()->get_transform()(_z_values[i]),
                                              _rounding_points))},
                     {"z_std", std::to_string(round_to(t_p_std[i], _rounding_points))},
                     {"z_z", std::to_string(round_to(t_z[i] / t_p_std[i], _rounding_points))},
                     {"z_p", std::to_string(round_to(find_p_value(t_z[i] / t_p_std[i]), _rounding_points))},
                     {"ci", "(" + std::to_string(round_to(t_z[i] - t_p_std[i] * 1.96, _rounding_points)) + " | " +
                                    std::to_string(round_to(t_z[i] + t_p_std[i] * 1.96, _rounding_points)) + ")"}});
        else if (transformed)
            data.push_back({{"z_name", _z.get_z_list()[i].get_name()},
                            {"z_value", std::to_string(round_to(
                                                _z.get_z_list()[i].get_prior_clone()->get_transform()(_z_values[i]),
                                                _rounding_points))}});
        else
            data.push_back(
                    {{"z_name", (_z.get_z_list()[i].get_prior_clone()->get_itransform_name() + "(" +
                                 _z.get_z_list()[i].get_name() + ")")},
                     {"z_value", (std::to_string(round_to(_z_values[i], _rounding_points)))},
                     {"z_std", (std::to_string(round_to(t_p_std[i], _rounding_points)))},
                     {"z_z", std::to_string(round_to(t_z[i] / t_p_std[i], _rounding_points))},
                     {"z_p", std::to_string(round_to(find_p_value(t_z[i] / t_p_std[i]), _rounding_points))},
                     {"ci", "(" + std::to_string(round_to(t_z[i] - t_p_std[i] * 1.96, _rounding_points)) + " | " +
                                    std::to_string(round_to(t_z[i] + t_p_std[i] * 1.96, _rounding_points)) + ")"}

                    });
    }

    std::vector<std::tuple<std::string, std::string, int>> fmt{{"Latent Variable", "z_name", 40},
                                                               {"Estimate", "z_value", 10},
                                                               {"Std Error", "z_std", 10},
                                                               {"z", "z_z", 8},
                                                               {"P>|z|", "z_p", 8},
                                                               {"95% C.I.", "ci", 25}};
    std::vector<std::tuple<std::string, std::string, int>> model_fmt{{_model_name, "model_details", 55},
                                                                     {"", "model_results", 50}};

    std::list<std::map<std::string, std::string>> model_details;
    std::string obj_desc;
    if (_method == "MLE" || _method == "OLS")
        obj_desc = "Log Likelihood" + std::to_string(round_to(-_objective_object(_z_values), 4));
    else
        obj_desc = "Unnormalized Log Posterior" + std::to_string(round_to(-_objective_object(_z_values), 4));
    model_details.push_back(
            {{"model_details", "Dependent Variable: " + _data_name}, {"model_results", "Method: " + _method}});
    model_details.push_back(
            {{"model_details", "Start Date: " + std::to_string(_index.at(_max_lag))}, {"model_results", obj_desc}});
    model_details.push_back(
            {{"model_details", "End Date: " + std::to_string(_index.back())},
             {"model_results",
              "AIC: " + std::to_string(round_to(
                                2 * static_cast<double>(_z_values.size()) + 2 * _objective_object(_z_values), 4))}});
    model_details.push_back(
            {{"model_details", "Number of observations: " + std::to_string(_data_length)},
             {"model_results", "BIC: " + std::to_string(round_to(2 * _objective_object(_z_values) +
                                                                         static_cast<double>(_z_values.size()) *
                                                                                 log(static_cast<double>(_data_length)),
                                                                 4))}});

    std::cout << TablePrinter{model_fmt, " ", "="}(model_details) << "\n";
    std::cout << std::string(106, '=') << "\n";
    std::cout << TablePrinter{fmt, " ", "="}(data) << "\n";
    std::cout << std::string(106, '=') << "\n";
    if (_model_name.find("Skewt") != std::string::npos) {
        std::cout << "WARNING: Skew t distribution is not well-suited for MLE or MAP inference\n";
        std::cout << "Workaround 1: Use a t-distribution instead for MLE/MAP\n";
        std::cout << "Workaround 2: Use M-H or BBVI inference for Skew t distribution\n";
    }
}

void MLEResults::summary_without_hessian() const {
    Eigen::VectorXd t_z{_z.get_z_values(true)}; // Never used
    std::cout << "\nHessian not invertible! Consider a different model specification.\n";

    // Initialize data
    std::list<std::map<std::string, std::string>> data;
    std::vector<std::string> z_names{_z.get_z_names()};
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(z_names.size()); ++i)
        data.push_back({{"z_name", z_names[i]},
                        {"z_value", std::to_string(round_to(
                                            _z.get_z_list()[i].get_prior_clone()->get_transform()(_results[i]), 4))}});

    // Create fmts
    std::vector<std::tuple<std::string, std::string, int>> fmt{{"Latent Variable", "z_name", 40},
                                                               {"Estimate", "z_value", 10}};
    std::vector<std::tuple<std::string, std::string, int>> model_fmt{{_model_name, "model_details", 55},
                                                                     {"", "model_results", 50}};
    // Initialize model_details
    std::list<std::map<std::string, std::string>> model_details;
    std::string obj_desc = (_method == "MLE") ? "Log Likelihood: " : "Unnormalized Log Posterior: ";
    obj_desc += std::to_string(round_to(-_objective_object(_results), 4));
    model_details.push_back(
            {{"model_details", "Dependent Variable: " + _data_name}, {"model_results", "Method: " + _method}});
    model_details.push_back(
            {{"model_details", "Start Date: " + std::to_string(_index.at(_max_lag))}, {"model_results", obj_desc}});
    model_details.push_back({{"model_details", "End Date: " + std::to_string(_index.back())},
                             {"model_results", "AIC: " + std::to_string(_aic)}});
    model_details.push_back({{"model_details", "Number of observations: " + std::to_string(_data_length)},
                             {"model_results", "BIC: " + std::to_string(_bic)}});

    std::cout << TablePrinter{model_fmt, " ", "="}(model_details) << "\n";
    std::cout << std::string(106, '=') << "\n";
    std::cout << TablePrinter{fmt, " ", "="}(data) << "\n";
    std::cout << std::string(106, '=') << "\n";
    if (_model_name.find("Skewt") != std::string::npos) {
        std::cout << "WARNING: Skew t distribution is not well-suited for MLE or MAP inference\n";
        std::cout << "Workaround 1: Use a t-distribution instead for MLE/MAP\n";
        std::cout << "Workaround 2: Use M-H or BBVI inference for Skew t distribution\n";
    }
}

Eigen::MatrixXd MLEResults::get_ihessian() const {
    return _ihessian;
}

BBVIResults::BBVIResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                         const std::string& model_type, const LatentVariables& latent_variables, Eigen::MatrixXd data,
                         std::vector<double> index, bool multivariate_model,
                         std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide,
                         int max_lag, Eigen::VectorXd ses, Eigen::VectorXd signal,
                         std::optional<Eigen::VectorXd> scores, Eigen::VectorXd elbo_records,
                         std::optional<Eigen::VectorXd> states, std::optional<Eigen::VectorXd> states_var)
    : Results{std::move(data_name),
              std::move(X_names),
              std::move(model_name),
              model_type,
              latent_variables,
              std::move(data),
              std::move(index),
              multivariate_model,
              std::move(objective_object),
              std::move(method),
              z_hide,
              max_lag,
              std::move(signal),
              std::move(scores),
              std::move(states),
              std::move(states_var)},
      _ses{std::move(ses)}, _elbo_records{std::move(elbo_records)} {
    _ihessian = Eigen::MatrixXd(static_cast<Eigen::VectorXd>(_ses.array().exp().pow(2)).asDiagonal());
    _aic      = 2 * static_cast<double>(_z_values.size()) + 2 * _objective_object(_z_values);
    _bic      = 2 * _objective_object(_z_values) +
           static_cast<double>(_z_values.size()) * log(static_cast<double>(_data_length));

    Sample samp                                   = nps::norm_post_sim(_z_values, _ihessian);
    _chain                                        = samp.chain;
    _mean_est                                     = samp.mean_est;
    _median_est                                   = samp.median_est;
    _upper_95_est                                 = samp.upper_95_est;
    _lower_5_est                                  = samp.lower_95_est;
    _t_chain                                      = Eigen::MatrixXd(_chain.rows(), _chain.cols());
    _t_mean_est                                   = Eigen::VectorXd(_chain.rows());
    _t_median_est                                 = Eigen::VectorXd(_chain.rows());
    _t_upper_95_est                               = Eigen::VectorXd(_chain.rows());
    _t_lower_5_est                                = Eigen::VectorXd(_chain.rows());
    std::vector<std::unique_ptr<Family>> z_priors = _z.get_z_priors();
    for (Eigen::Index k{0}; k < _chain.rows(); k++) {
        auto transform{z_priors[k]->get_transform()};
        std::transform(_chain.row(k).begin(), _chain.row(k).end(), _t_chain.row(k).begin(),
                       [transform](double x) { return transform(x); });
        _t_mean_est(k)     = transform(_mean_est(k));
        _t_median_est(k)   = transform(_median_est(k));
        _t_upper_95_est(k) = transform(_upper_95_est(k));
        _t_lower_5_est(k)  = transform(_lower_5_est(k));
    }
}

std::ostream& operator<<(std::ostream& stream, const BBVIResults& results) {
    stream << "BBVI Results Object"
              "\n=========================="
              "\nDependent variable: "
           << results._data_name << "\nRegressors: ";
    for (const std::string& s : results._x_names)
        stream << s << " ";
    stream << "\n=========================="
              "\nLatent Variable Attributes: "
              "\n.z : LatentVariables() object"
              "\n.results : optimizer results"
              "\n\nImplied Model Attributes: "
              "\n.aic: Akaike Information Criterion"
              "\n.bic: Bayesian Information Criterion"
              "\n.data: Model Data"
              "\n.index: Model Index";
    if (results._scores.has_value())
        stream << "\n.scores: Model Scores";
    if (results._signal.size() > 0)
        stream << "\n.signal: Model Signal";
    if (results._states.has_value())
        stream << "\n.states: Model States";
    if (results._states_var.has_value())
        stream << "\n.states_var: Model State Variances";
    stream << "\n\nMethods: "
              "\n.summary() : printed results";
    return stream;
}

void BBVIResults::plot_elbo(size_t width, size_t height) const {
    sciplot::Plot plot;
    plot.size(width, height);
    std::vector<double> elbo_records{&_elbo_records[0], _elbo_records.data() + _elbo_records.size()};
    plot.drawCurve(std::vector<double>(), elbo_records);
    plot.xlabel("Iterations");
    plot.ylabel("ELBO");
    plot.save("../data/BBVIResults_plots/plot_elbo.pdf");
    plot.show();
}

void BBVIResults::summary(bool transformed) {
    // Initialize data
    std::list<std::map<std::string, std::string>> data;
    std::vector<std::string> z_names = _z.get_z_names();
    if (transformed) {
        for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(z_names.size() - _z_hide); ++i) {
            data.push_back({{"z_name", z_names[i]},
                            {"z_mean", std::to_string(round_to(_t_mean_est[i], _rounding_points))},
                            {"z_median", std::to_string(round_to(_t_median_est[i], _rounding_points))},
                            {"ci", "(" + std::to_string(round_to(_t_lower_5_est[i], _rounding_points)) + "|" +
                                           std::to_string(round_to(_t_upper_95_est[i], _rounding_points)) + ")"}});
        }
    } else {
        for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(z_names.size() - _z_hide); ++i) {
            data.push_back({{"z_name", z_names[i]},
                            {"z_mean", std::to_string(round_to(_mean_est[i], _rounding_points))},
                            {"z_median", std::to_string(round_to(_median_est[i], _rounding_points))},
                            {"ci", "(" + std::to_string(round_to(_lower_5_est[i], _rounding_points)) + "|" +
                                           std::to_string(round_to(_upper_95_est[i], _rounding_points)) + ")"}});
        }
    }
    // Create fmts
    std::vector<std::tuple<std::string, std::string, int>> fmt{{"Latent Variable", "z_name", 40},
                                                               {"Median", "z_median", 18},
                                                               {"Mean", "z_mean", 18},
                                                               {"95% Credibility Interval", "ci", 25}};
    std::vector<std::tuple<std::string, std::string, int>> model_fmt{{_model_name, "model_details", 55},
                                                                     {"", "model_results", 50}};
    // Initialize model_details
    std::list<std::map<std::string, std::string>> model_details;
    std::string obj_desc = (_method == "MLE") ? "Log Likelihood: " : "Unnormalized Log Posterior: ";
    obj_desc += std::to_string(round_to(-_objective_object(_z_values), 4));
    model_details.push_back(
            {{"model_details", "Dependent Variable: " + _data_name}, {"model_results", "Method: " + _method}});
    model_details.push_back(
            {{"model_details", "Start Date: " + std::to_string(_index.at(_max_lag))}, {"model_results", obj_desc}});
    model_details.push_back({{"model_details", "End Date: " + std::to_string(_index.back())},
                             {"model_results", "AIC: " + std::to_string(_aic)}});
    model_details.push_back({{"model_details", "Number of observations: " + std::to_string(_data_length)},
                             {"model_results", "BIC: " + std::to_string(_bic)}});
    // Print the summary
    std::cout << TablePrinter{model_fmt, " ", "="}(model_details) << "\n";
    std::cout << std::string(106, '=') << "\n";
    std::cout << TablePrinter{fmt, " ", "="}(data) << "\n";
    std::cout << std::string(106, '=') << "\n";
}

BBVISSResults::BBVISSResults(std::vector<std::string> data_name, std::vector<std::string> X_names,
                             std::string model_name, const std::string& model_type,
                             const LatentVariables& latent_variables, Eigen::MatrixXd data, std::vector<double> index,
                             bool multivariate_model, double objective_value, std::string method, bool z_hide,
                             int max_lag, Eigen::VectorXd ses, Eigen::VectorXd signal,
                             std::optional<Eigen::VectorXd> scores, Eigen::VectorXd elbo_records,
                             std::optional<Eigen::VectorXd> states, std::optional<Eigen::VectorXd> states_var)
    : Results{std::move(data_name),
              std::move(X_names),
              std::move(model_name),
              model_type,
              latent_variables,
              std::move(data),
              std::move(index),
              multivariate_model,
              {},
              std::move(method),
              z_hide,
              max_lag,
              std::move(signal),
              std::move(scores),
              std::move(states),
              std::move(states_var)},
      _objective_value{objective_value}, _ses{std::move(ses)}, _elbo_records{std::move(elbo_records)} {
    _ihessian = ses.array().exp().pow(2).matrix().diagonal();
    _aic      = 2 * static_cast<double>(_z_values.size()) + 2 * _objective_value;
    _bic      = 2 * _objective_value + static_cast<double>(_z_values.size()) * log(static_cast<double>(_data_length));

    Sample samp                                   = nps::norm_post_sim(_z_values, _ihessian);
    _chain                                        = samp.chain;
    _mean_est                                     = samp.mean_est;
    _median_est                                   = samp.median_est;
    _upper_95_est                                 = samp.upper_95_est;
    _lower_5_est                                  = samp.lower_95_est;
    _t_chain                                      = _chain;
    std::vector<std::unique_ptr<Family>> z_priors = _z.get_z_priors();
    for (Eigen::Index k{0}; k < _mean_est.size(); k++) {
        //_t_chain(k) = z_priors[k]->get_transform()(_chain(k));
        _t_mean_est(k)     = z_priors[k]->get_transform()(_mean_est(k));
        _t_median_est(k)   = z_priors[k]->get_transform()(_median_est(k));
        _t_upper_95_est(k) = z_priors[k]->get_transform()(_upper_95_est(k));
        _t_lower_5_est(k)  = z_priors[k]->get_transform()(_lower_5_est(k));
    }
}

std::ostream& operator<<(std::ostream& stream, const BBVISSResults& results) {
    stream << "BBVI Results Object"
              "\n=========================="
              "\nDependent variable: "
           << results._data_name << "\nRegressors: ";
    for (const std::string& s : results._x_names)
        stream << s << " ";
    stream << "\n=========================="
              "\nLatent Variable Attributes: "
              "\n.z : LatentVariables() object"
              "\n\nImplied Model Attributes: "
              "\n.aic: Akaike Information Criterion"
              "\n.bic: Bayesian Information Criterion"
              "\n.data: Model Data"
              "\n.index: Model Index"
              "\n.objective: Unnormalized Log Posterior";
    if (results._scores.has_value())
        stream << "\n.scores: Model Scores";
    if (results._signal.size() > 0)
        stream << "\n.signal: Model Signal";
    if (results._states.has_value())
        stream << "\n.states: Model States";
    if (results._states_var.has_value())
        stream << "\n.states_var: Model State Variances";
    stream << "\n\nMethods: "
              "\n.summary() : printed results";
    return stream;
}

void BBVISSResults::plot_elbo(size_t width, size_t height) const {
    sciplot::Plot plot;
    plot.size(width, height);
    std::vector<double> elbo_records{&_elbo_records[0], _elbo_records.data() + _elbo_records.size()};
    plot.drawCurve(std::vector<double>(), elbo_records);
    plot.xlabel("Iterations");
    plot.ylabel("ELBO");
    plot.save("../data/BBVISResults_plots/plot_elbo.pdf");
    plot.show();
}

void BBVISSResults::summary(bool transformed) {
    // Initialize data
    std::list<std::map<std::string, std::string>> data;
    std::vector<std::string> z_names = _z.get_z_names();
    if (transformed) {
        for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(z_names.size() - _z_hide); ++i) {
            data.push_back({{"z_name", z_names[i]},
                            {"z_mean", std::to_string(round_to(_t_mean_est[i], _rounding_points))},
                            {"z_median", std::to_string(round_to(_t_median_est[i], _rounding_points))},
                            {"ci", "(" + std::to_string(round_to(_t_lower_5_est[i], _rounding_points)) + "|" +
                                           std::to_string(round_to(_t_upper_95_est[i], _rounding_points)) + ")"}});
        }
    } else {
        for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(z_names.size() - _z_hide); ++i) {
            data.push_back({{"z_name", z_names[i]},
                            {"z_mean", std::to_string(round_to(_mean_est[i], _rounding_points))},
                            {"z_median", std::to_string(round_to(_median_est[i], _rounding_points))},
                            {"ci", "(" + std::to_string(round_to(_lower_5_est[i], _rounding_points)) + "|" +
                                           std::to_string(round_to(_upper_95_est[i], _rounding_points)) + ")"}});
        }
    }
    // Create fmts
    std::vector<std::tuple<std::string, std::string, int>> fmt{{"Latent Variable", "z_name", 40},
                                                               {"Median", "z_median", 18},
                                                               {"Mean", "z_mean", 18},
                                                               {"95% Credibility Interval", "ci", 25}};
    std::vector<std::tuple<std::string, std::string, int>> model_fmt{{_model_name, "model_details", 55},
                                                                     {"", "model_results", 50}};
    // Initialize model_details
    std::list<std::map<std::string, std::string>> model_details;
    std::string obj_desc = "Unnormalized Log Posterior: " + std::to_string(round_to(-_objective_value, 4));
    model_details.push_back(
            {{"model_details", "Dependent Variable: " + _data_name}, {"model_results", "Method: " + _method}});
    model_details.push_back(
            {{"model_details", "Start Date: " + std::to_string(_index.at(_max_lag))}, {"model_results", obj_desc}});
    model_details.push_back({{"model_details", "End Date: " + std::to_string(_index.back())},
                             {"model_results", "AIC: " + std::to_string(_aic)}});
    model_details.push_back({{"model_details", "Number of observations: " + std::to_string(_data_length)},
                             {"model_results", "BIC: " + std::to_string(_bic)}});
    // Print the summary
    std::cout << TablePrinter{model_fmt, " ", "="}(model_details) << "\n";
    std::cout << std::string(106, '=') << "\n";
    std::cout << TablePrinter{fmt, " ", "="}(data) << "\n";
    std::cout << std::string(106, '=') << "\n";
}

LaplaceResults::LaplaceResults(std::vector<std::string> data_name, std::vector<std::string> X_names,
                               std::string model_name, const std::string& model_type,
                               const LatentVariables& latent_variables, Eigen::MatrixXd data, std::vector<double> index,
                               bool multivariate_model, std::function<double(Eigen::VectorXd)> objective_object,
                               std::string method, bool z_hide, int max_lag, Eigen::MatrixXd ihessian,
                               Eigen::VectorXd signal, std::optional<Eigen::VectorXd> scores,
                               std::optional<Eigen::VectorXd> states, std::optional<Eigen::VectorXd> states_var)
    : Results{std::move(data_name),
              std::move(X_names),
              std::move(model_name),
              model_type,
              latent_variables,
              std::move(data),
              std::move(index),
              multivariate_model,
              std::move(objective_object),
              std::move(method),
              z_hide,
              max_lag,
              std::move(signal),
              std::move(scores),
              std::move(states),
              std::move(states_var)},
      _ihessian{std::move(ihessian)} {
    if (_multivariate_model) {
        _data_length = _data.rows();
        _data_name.push_back(',');
    } else
        _data_length = _data.size();

    _z_values = _z.get_z_values(false);
    _aic      = 2 * static_cast<double>(_z_values.size()) + 2 * _objective_object(_z_values);
    _bic      = 2 * _objective_object(_z_values) +
           static_cast<double>(_z_values.size()) * log(static_cast<double>(_data_length));

    if (_model_type == "LLT" || _model_type == "LLEV")
        _rounding_points = 10;
    else
        _rounding_points = 4;

    Sample samp                                   = nps::norm_post_sim(_z_values, _ihessian);
    _chain                                        = samp.chain;
    _mean_est                                     = samp.mean_est;
    _median_est                                   = samp.median_est;
    _upper_95_est                                 = samp.upper_95_est;
    _lower_5_est                                  = samp.lower_95_est;
    _t_chain                                      = _chain;
    _t_mean_est                                   = Eigen::VectorXd(_mean_est.size());
    _t_median_est                                 = Eigen::VectorXd(_median_est.size());
    _t_upper_95_est                               = Eigen::VectorXd(_upper_95_est.size());
    _t_lower_5_est                                = Eigen::VectorXd(_lower_5_est.size());
    std::vector<std::unique_ptr<Family>> z_priors = _z.get_z_priors();
    for (Eigen::Index k{0}; k < _mean_est.size(); k++) {
        //_t_chain(k) = z_priors[k]->get_transform()(_chain(k));
        _t_mean_est(k)     = z_priors[k]->get_transform()(_mean_est(k));
        _t_median_est(k)   = z_priors[k]->get_transform()(_median_est(k));
        _t_upper_95_est(k) = z_priors[k]->get_transform()(_upper_95_est(k));
        _t_lower_5_est(k)  = z_priors[k]->get_transform()(_lower_5_est(k));
    }
}

std::ostream& operator<<(std::ostream& stream, const LaplaceResults& results) {
    stream << "Laplace Results Object";
    stream << "\n=========================="
              "\nDependent variable: "
           << results._data_name << "\nRegressors: ";
    for (const std::string& s : results._x_names)
        stream << s << " ";
    stream << "\n=========================="
              "\nLatent Variable Attributes: ";
    if (results._ihessian.size() > 0)
        stream << "\n.ihessian: Inverse Hessian";
    stream << "\n.z : LatentVariables() object";
    stream << "\n.results : optimizer results";
    stream << "\n\nImplied Model Attributes: "
              "\n.aic: Akaike Information Criterion"
              "\n.bic: Bayesian Information Criterion"
              "\n.data: Model Data"
              "\n.index: Model Index";
    if (results._scores.has_value())
        stream << "\n.scores: Model Scores";
    if (results._signal.size() > 0)
        stream << "\n.signal: Model Signal";
    if (results._states.has_value())
        stream << "\n.states: Model States";
    if (results._states_var.has_value())
        stream << "\n.states_var: Model State Variances";
    stream << "\n\nMethods: "
              "\n.summary() : printed results";
    return stream;
}

void LaplaceResults::summary(bool transformed) {
    //_z_values = _z.get_z_values(false);
    std::list<std::map<std::string, std::string>> data;
    std::vector<std::string> z_names = _z.get_z_names();
    if (transformed) {
        for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(z_names.size() - _z_hide); ++i) {
            data.push_back({{"z_name", z_names[i]},
                            {"z_mean", std::to_string(round_to(_t_mean_est[i], _rounding_points))},
                            {"z_median", std::to_string(round_to(_t_median_est[i], _rounding_points))},
                            {"ci", "(" + std::to_string(round_to(_t_lower_5_est[i], _rounding_points)) + "|" +
                                           std::to_string(round_to(_t_upper_95_est[i], _rounding_points)) + ")"}});
        }
    } else {
        for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(z_names.size() - _z_hide); ++i) {
            data.push_back({{"z_name", z_names[i]},
                            {"z_mean", std::to_string(round_to(_mean_est[i], _rounding_points))},
                            {"z_median", std::to_string(round_to(_median_est[i], _rounding_points))},
                            {"ci", "(" + std::to_string(round_to(_lower_5_est[i], _rounding_points)) + "|" +
                                           std::to_string(round_to(_upper_95_est[i], _rounding_points)) + ")"}});
        }
    }

    // Create fmts
    std::vector<std::tuple<std::string, std::string, int>> fmt{{"Latent Variable", "z_name", 40},
                                                               {"Median", "z_median", 18},
                                                               {"Mean", "z_mean", 18},
                                                               {"95% Credibility Interval", "ci", 25}};
    std::vector<std::tuple<std::string, std::string, int>> model_fmt{{_model_name, "model_details", 55},
                                                                     {"", "model_results", 50}};
    // Initialize model_details
    std::list<std::map<std::string, std::string>> model_details;

    std::string obj_desc;
    if (_method == "MLE")
        obj_desc = ("Log Likelihood: " + std::to_string(round_to(-1 * _objective_object(_z_values), 4)));
    else
        obj_desc = "Unnormalized Log Posterior: " + std::to_string(round_to(_objective_object(_z_values), 4));

    model_details.push_back(
            {{"model_details", "Dependent Variable: " + _data_name}, {"model_results", "Method: " + _method}});
    model_details.push_back(
            {{"model_details", "Start Date: " + std::to_string(_index.at(_max_lag))}, {"model_results", obj_desc}});
    model_details.push_back({{"model_details", "End Date: " + std::to_string(_index.back())},
                             {"model_results", "AIC: " + std::to_string(_aic)}});
    model_details.push_back({{"model_details", "Number of observations: " + std::to_string(_data_length)},
                             {"model_results", "BIC: " + std::to_string(_bic)}});

    std::cout << TablePrinter{model_fmt, " ", "="}(model_details) << "\n";
    std::cout << std::string(106, '=') << "\n";
    std::cout << TablePrinter{fmt, " ", "="}(data) << "\n";
    std::cout << std::string(106, '=') << "\n";
}

MCMCResults::MCMCResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                         const std::string& model_type, const LatentVariables& latent_variables, Eigen::MatrixXd data,
                         std::vector<double> index, bool multivariate_model,
                         std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide,
                         int max_lag, Eigen::MatrixXd samples, Eigen::VectorXd mean_est, Eigen::VectorXd median_est,
                         Eigen::VectorXd upper_95_est, Eigen::VectorXd lower_95_est, Eigen::VectorXd signal,
                         std::optional<Eigen::VectorXd> scores, std::optional<Eigen::VectorXd> states,
                         std::optional<Eigen::VectorXd> states_var)
    : Results{std::move(data_name),
              std::move(X_names),
              std::move(model_name),
              model_type,
              latent_variables,
              std::move(data),
              std::move(index),
              multivariate_model,
              std::move(objective_object),
              std::move(method),
              z_hide,
              max_lag,
              std::move(signal),
              std::move(scores),
              std::move(states),
              std::move(states_var)},
      _samples{std::move(samples)}, _mean_est{std::move(mean_est)}, _median_est{std::move(median_est)},
      _lower_95_est{std::move(lower_95_est)}, _upper_95_est{std::move(upper_95_est)} {
    _aic = 2 * static_cast<double>(_z_values.size()) + 2 * _objective_object(_z_values);
    _bic = 2 * _objective_object(_z_values) +
           static_cast<double>(_z_values.size()) * log(static_cast<double>(_data.size()));
}

std::ostream& operator<<(std::ostream& stream, const MCMCResults& results) {
    stream << "Metropolis Hastings Results Object"
              "\n=========================="
              "\nDependent variable: "
           << results._data_name << "\nRegressors: ";
    for (const std::string& s : results._x_names)
        stream << s << " ";
    stream << "\n=========================="
              "\nLatent Variable Attributes: "
              "\n.z : LatentVariables() object";
    if (results._samples.size() > 0)
        stream << "\n.samples: MCMC samples";
    stream << "\n\n.results : optimizer results"
              "\n\nImplied Model Attributes: "
              "\n.aic: Akaike Information Criterion"
              "\n.bic: Bayesian Information Criterion"
              "\n.data: Model Data"
              "\n.index: Model Index";
    if (results._scores.has_value())
        stream << "\n.scores: Model Scores";
    if (results._signal.size() > 0)
        stream << "\n.signal: Model Signal";
    if (results._states.has_value())
        stream << "\n.states: Model States";
    if (results._states_var.has_value())
        stream << "\n.states_var: Model State Variances";
    stream << "\n\nMethods: "
              "\n.summary() : printed results";
    return stream;
}

void MCMCResults::summary([[maybe_unused]] bool transformed) {
    _z_values = _z.get_z_values(false);
    std::list<std::map<std::string, std::string>> data;
    std::vector<std::string> z_names = _z.get_z_names();
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(z_names.size() - _z_hide); ++i)
        data.push_back({{"z_name", z_names[i]},
                        {"z_mean", std::to_string(round_to(_mean_est[i], _rounding_points))},
                        {"z_median", std::to_string(round_to(_median_est[i], _rounding_points))},
                        {"ci", "(" + std::to_string(round_to(_lower_95_est[i], _rounding_points)) + "|" +
                                       std::to_string(round_to(_upper_95_est[i], _rounding_points)) + ")"}});

    // Create fmts
    std::vector<std::tuple<std::string, std::string, int>> fmt{{"Latent Variable", "z_name", 40},
                                                               {"Median", "z_median", 18},
                                                               {"Mean", "z_mean", 18},
                                                               {"95% Credibility Interval", "ci", 25}};
    std::vector<std::tuple<std::string, std::string, int>> model_fmt{{_model_name, "model_details", 55},
                                                                     {"", "model_results", 50}};
    // Initialize model_details
    std::list<std::map<std::string, std::string>> model_details;

    std::string obj_desc;
    if (_method == "MLE")
        obj_desc = ("Log Likelihood: " + std::to_string(round_to(-1 * _objective_object(_z_values), 4)));
    else
        obj_desc = "Unnormalized Log Posterior: " + std::to_string(round_to(_objective_object(_z_values), 4));

    model_details.push_back(
            {{"model_details", "Dependent Variable: " + _data_name}, {"model_results", "Method: " + _method}});
    model_details.push_back(
            {{"model_details", "Start Date: " + std::to_string(_index.at(_max_lag))}, {"model_results", obj_desc}});
    model_details.push_back({{"model_details", "End Date: " + std::to_string(_index.back())},
                             {"model_results", "AIC: " + std::to_string(_aic)}});
    model_details.push_back({{"model_details", "Number of observations: " + std::to_string(_data_length)},
                             {"model_results", "BIC: " + std::to_string(_bic)}});

    std::cout << TablePrinter{model_fmt, " ", "="}(model_details) << "\n";
    std::cout << std::string(106, '=') << "\n";
    std::cout << TablePrinter{fmt, " ", "="}(data) << "\n";
    std::cout << std::string(106, '=') << "\n";
}