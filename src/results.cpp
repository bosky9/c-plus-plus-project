#include "results.hpp"

#include "inference/sample.hpp"
#include "inference/norm_post_sim.hpp"

#include <algorithm>

Results::Results(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                 const std::string& model_type, const LatentVariables& latent_variables, Eigen::MatrixXd data,
                 std::vector<size_t> index, bool multivariate_model,
                 std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide, int max_lag,
                 Eigen::VectorXd signal, Eigen::VectorXd scores, Eigen::VectorXd states, Eigen::VectorXd states_var)
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

MLEResults::MLEResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                       const std::string& model_type, const LatentVariables& latent_variables, Eigen::VectorXd results,
                       Eigen::MatrixXd data, std::vector<size_t> index, bool multivariate_model,
                       std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide,
                       int max_lag, Eigen::MatrixXd ihessian, Eigen::VectorXd signal, Eigen::VectorXd scores,
                       Eigen::VectorXd states, Eigen::VectorXd states_var)
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
        _bic    = 2 * _objective_object(_z_values) + static_cast<double>(_z_values.size()) * log(_data_length);
    } else if (_method == "PML") {
        _aic = 2 * static_cast<double>(_z_values.size()) + 2 * _objective_object(_z_values);
        _bic = 2 * _objective_object(_z_values) + static_cast<double>(_z_values.size()) * log(_data_length);
    }
}

inline std::ostream& operator<<(std::ostream& stream, const MLEResults& mle_results) {
    if (mle_results._method == "MLE")
        stream << "MLE Results Object";
    else if (mle_results._method == "OLS")
        stream << "OLS Results Object";
    else
        stream << "PML Results Object";
    stream << "\n=========================="
              "\nDependent variable: "
           << mle_results._data_name << "\nRegressors: ";
    for (const std::string& s : mle_results._x_names)
        stream << s << " ";
    stream << "\n=========================="
              "\nLatent Variable Attributes: ";
    if (mle_results._ihessian.size() > 0)
        stream << "\n.ihessian: Inverse Hessian";
    stream << "\n.z : LatentVariables() object";
    if (mle_results._results.size() > 0)
        stream << "\n.results : optimizer results";
    stream << "\n\nImplied Model Attributes: "
              "\n.aic: Akaike Information Criterion"
              "\n.bic: Bayesian Information Criterion"
              "\n.data: Model Data"
              "\n.index: Model Index";
    if (mle_results._method == "MLE" || mle_results._method == "OLS")
        stream << "\n.loglik: Loglikelihood";
    if (mle_results._scores.size() > 0)
        stream << "\n.scores: Model Scores";
    if (mle_results._signal.size() > 0)
        stream << "\n.signal: Model Signal";
    if (mle_results._states.size() > 0)
        stream << "\n.states: Model States";
    if (mle_results._states_var.size() > 0)
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
    for (Eigen::Index i{0}; i < z_names.size() - _z_hide; i++) {
        if (_z.get_z_list()[i].get_prior()->get_transform_name().empty())
            data.push_back(
                    {{"z_name", z_names[i]},
                     {"z_value", std::to_string(round_to(_z.get_z_list()[i].get_prior()->get_transform()(_z_values[i]),
                                                         _rounding_points))},
                     {"z_std", std::to_string(round_to(t_p_std[i], _rounding_points))},
                     {"z_z", std::to_string(round_to(t_z[i] / t_p_std[i], _rounding_points))},
                     {"z_p", std::to_string(round_to(find_p_value(t_z[i] / t_p_std[i]), _rounding_points))},
                     {"ci", "(" + std::to_string(round_to(t_z[i] - t_p_std[i] * 1.96, _rounding_points)) + " | " +
                                    std::to_string(round_to(t_z[i] + t_p_std[i] * 1.96, _rounding_points)) + ")"}});
        else if (transformed)
            data.push_back(
                    {{"z_name", _z.get_z_list()[i].get_name()},
                     {"z_value", std::to_string(round_to(_z.get_z_list()[i].get_prior()->get_transform()(_z_values[i]),
                                                         _rounding_points))}});
        else
            data.push_back(
                    {{"z_name", (_z.get_z_list()[i].get_prior()->get_itransform_name() + "(" +
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
            {{"model_details", "Start Date: " + std::to_string(_index[_max_lag])}, {"model_results", obj_desc}});
    model_details.push_back(
            {{"model_details", "End Date: " + std::to_string(_index[-1])},
             {"model_results",
              "AIC: " + std::to_string(round_to(
                                2 * static_cast<double>(_z_values.size()) + 2 * _objective_object(_z_values), 4))}});
    model_details.push_back(
            {{"model_details", "Number of observations: " + std::to_string(_data_length)},
             {"model_results",
              "BIC: " + std::to_string(round_to(2 * _objective_object(_z_values) +
                                                        static_cast<double>(_z_values.size()) * log(_data_length),
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
    for (Eigen::Index i{0}; i < z_names.size(); i++)
        data.push_back({{"z_name", z_names[i]},
                        {"z_value",
                         std::to_string(round_to(_z.get_z_list()[i].get_prior()->get_transform()(_results[i]), 4))}});

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
            {{"model_details", "Start Date: " + std::to_string(_index[_max_lag])}, {"model_results", obj_desc}});
    model_details.push_back({{"model_details", "End Date: " + std::to_string(_index[-1])},
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

BBVIResults::BBVIResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                         const std::string& model_type, const LatentVariables& latent_variables, Eigen::MatrixXd data,
                         std::vector<size_t> index, bool multivariate_model,
                         std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide,
                         int max_lag, Eigen::VectorXd ses, Eigen::VectorXd signal, Eigen::VectorXd scores,
                         std::vector<double> elbo_records, Eigen::VectorXd states, Eigen::VectorXd states_var)
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
    _aic = 2 * static_cast<double>(_z_values.size()) + 2 * _objective_object(_z_values);
    _bic = 2 * _objective_object(_z_values) + static_cast<double>(_z_values.size()) * log(_data_length);

    Sample samp = norm_post_sim(_z_values, _ihessian);
    _chain = samp.chain;
    _mean_est = samp.mean_est;
    _median_est = samp.median_est;
    _upper_95_est = samp.upper_95_est;
    _lower_5_est = samp.lower_5_est;
    _t_chain = _chain;
    std::vector<Family*> z_priors = _z.get_z_priors();
    for (size_t k{0}; k < _mean_est.size(); k++) {
        //_t_chain(k) = z_priors.at(k)->get_transform()(_chain(k));
        _t_mean_est(k) = z_priors.at(k)->get_transform()(_mean_est(k));
        _t_median_est(k) = z_priors.at(k)->get_transform()(_median_est(k));
        _t_upper_95_est(k) = z_priors.at(k)->get_transform()(_upper_95_est(k));
        _t_lower_5_est(k) = z_priors.at(k)->get_transform()(_lower_5_est(k));
    }
}

inline std::ostream& operator<<(std::ostream& stream, const BBVIResults& results) {
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
    if (results._scores.size() > 0)
        stream << "\n.scores: Model Scores";
    if (results._signal.size() > 0)
        stream << "\n.signal: Model Signal";
    if (results._states.size() > 0)
        stream << "\n.states: Model States";
    if (results._states_var.size() > 0)
        stream << "\n.states_var: Model State Variances";
    stream << "\n\nMethods: "
              "\n.summary() : printed results";
    return stream;
}

void BBVIResults::plot_elbo(size_t width, size_t height) const {
    plt::figure_size(width, height);
    std::vector<double> elbo_records{&_elbo_records[0], _elbo_records.data()};
    plt::plot(elbo_records);
    plt::xlabel("Iterations");
    plt::ylabel("ELBO");
    plt::save("../data/BBVIResults_plot_elbo.png");
    // plt::show();
}

void BBVIResults::summary(bool transformed) {
    // Initialize data
    std::list<std::map<std::string, std::string>> data;
    std::vector<std::string> z_names = _z.get_z_names();
    if (transformed) {
        for (Eigen::Index i{0}; i < z_names.size() - _z_hide; i++) {
            data.push_back(
                    {{"z_name", z_names[i]},
                     {"z_mean", std::to_string(round_to(_t_mean_est(i), _rounding_points))},
                     {"z_median", std::to_string(round_to(_t_median_est(i), _rounding_points))},
                     {"ci", "(" + std::to_string(round_to(_t_lower_5_est(i), _rounding_points)) + "|"
                     + std::to_string(round_to(_t_upper_95_est(i), _rounding_points)) + ")"}
                    });
        }
    } else {
        for (Eigen::Index i{0}; i < z_names.size() - _z_hide; i++) {
            data.push_back(
                    {{"z_name", z_names[i]},
                     {"z_mean", std::to_string(round_to(_mean_est(i), _rounding_points))},
                     {"z_median", std::to_string(round_to(_median_est(i), _rounding_points))},
                     {"ci", "(" + std::to_string(round_to(_lower_5_est(i), _rounding_points)) + "|"
                     + std::to_string(round_to(_upper_95_est(i), _rounding_points)) + ")"}
                    });
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
            {{"model_details", "Start Date: " + std::to_string(_index[_max_lag])}, {"model_results", obj_desc}});
    model_details.push_back({{"model_details", "End Date: " + std::to_string(_index[-1])},
                             {"model_results", "AIC: " + std::to_string(_aic)}});
    model_details.push_back({{"model_details", "Number of observations: " + std::to_string(_data_length)},
                             {"model_results", "BIC: " + std::to_string(_bic)}});
    // Print the summary
    std::cout << TablePrinter{model_fmt, " ", "="}(model_details) << "\n";
    std::cout << std::string(106, '=') << "\n";
    std::cout << TablePrinter{fmt, " ", "="}(data) << "\n";
    std::cout << std::string(106, '=') << "\n";
}

std::ostream& operator<<(std::ostream& stream, const BBVIResults& results) {
    stream << "BBVI Results Object";
    stream << "\n=========================="
              "\nDependent variable: "
           << results._data_name << "\nRegressors: ";
    for (const std::string& s : results._x_names)
        stream << s << " ";
    stream << "\n=========================="
              "\nLatent Variable Attributes: ";
    stream << "\n.z : LatentVariables() object";
    stream << "\n.results : optimizer results";
    stream << "\n\nImplied Model Attributes: "
              "\n.aic: Akaike Information Criterion"
              "\n.bic: Bayesian Information Criterion"
              "\n.data: Model Data"
              "\n.index: Model Index";
    if (results._scores.size() > 0)
        stream << "\n.scores: Model Scores";
    if (results._signal.size() > 0)
        stream << "\n.signal: Model Signal";
    if (results._states.size() > 0)
        stream << "\n.states: Model States";
    if (results._states_var.size() > 0)
        stream << "\n.states_var: Model State Variances";
    stream << "\n\nMethods: "
              "\n.summary() : printed results";
    return stream;
}