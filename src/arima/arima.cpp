/**
 * @file arima.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#define _USE_MATH_DEFINES

#include "arima/arima.hpp"

#include "Eigen/Core" // Eigen::VectorXd, Eigen::MatrixXd, Eigen::Index, Eigen::last, Eigen::all, Eigen::seq
#include "arima/arima_recursion.hpp" // arima_recursion, arima_recursion_normal
#include "data_check.hpp"            // data_check
#include "families/family.hpp"       // Family, FamilyAttributes, lv_to_build
#include "families/normal.hpp"       // Normal
#include "latent_variables.hpp"      // LatentVariables
#include "multivariate_normal.hpp"   // Mvn::logpdf
#include "results.hpp"               // Results
#include "sciplot/sciplot.hpp"       // sciplot::Plot
#include "tsm.hpp"                   // TSM, SingleDataFrame, draw_latent_variables
#include "utilities.hpp"             // utils::diff, utils::DataFrame, utils::percentile, utils::mean

#include <algorithm> // std::max, std::min, std::copy, std::transform, std::max_element, std::min_element
#include <cmath>     // std::sqrt, std::tgamma, M_PI
#include <limits>    // std::numeric_limits
#include <numeric>   // std::reduce, std::iota
#include <optional>  // std::optional
#include <random>    // std::random_device, std::default_random_engine, std::uniform_int_distribution
#include <string>    // std::string, std::to_string
#include <tuple>     // std::get, std::tuple, std::make_tuple
#include <utility>   // std::pair, std::move
#include <vector>    // std::vector

template<typename T>
ARIMA::ARIMA(const T& data, size_t ar, size_t ma, size_t integ, const std::optional<std::string>& target,
             const Family& family)
    : TSM("ARIMA") {
    // Latent Variable information
    _ar                 = ar;
    _ma                 = ma;
    _integ              = integ;
    _z_no               = _ar + _ma + 2;
    _max_lag            = std::max(static_cast<int>(_ar), static_cast<int>(_ma));
    _z_hide             = false; // Whether to cutoff latent variables from results table
    _supported_methods  = {"MLE", "PML", "Laplace", "M-H", "BBVI"};
    _default_method     = "MLE";
    _multivariate_model = false;

    // Format the data
    utils::SingleDataFrame checked_data = data_check(data, _data_original, target);
    _data_frame.data                    = checked_data.data;
    _data_frame.data_name               = checked_data.data_name;
    _data_frame.index                   = checked_data.index;

    // Difference data
    for (size_t order{0}; order < _integ; order++)
        _data_frame.data = utils::diff(_data_frame.data);
    _data_frame.data_name = "Differenced " + _data_frame.data_name;
    _data_length          = _data_frame.data.size();

    _x = ar_matrix();
    create_latent_variables();

    _family.reset();
    _family             = (family.clone());
    FamilyAttributes fa = family.setup();
    _model_name2        = fa.name;
    _link               = fa.link;
    _scale              = fa.scale;
    _shape              = fa.shape;
    _skewness           = fa.skewness;
    _mean_transform     = fa.mean_transform;
    _model_name         = _model_name2 + " ARIMA(" + std::to_string(_ar) + "," + std::to_string(_integ) + "," +
                  std::to_string(_ma) + ")";

    // Build any remaining latent variables that are specific to the family chosen
    std::vector<lv_to_build> lvs = _family->build_latent_variables();
    for (size_t no{0}; no < lvs.size(); no++) {
        lv_to_build lv = lvs.at(no);
        Family* prior  = std::get<1>(lv);
        Family* q      = std::get<2>(lv);
        _latent_variables.add_z(std::get<0>(lv), *prior, *q);
        _latent_variables.set_z_starting_value(1 + no + _ar + _ma, std::get<3>(lv));
        delete prior;
        delete q;
    }
    _latent_variables.set_z_starting_value(
            0, _mean_transform(static_cast<double>(std::reduce(_data_frame.data.begin(), _data_frame.data.end())) /
                               static_cast<double>(_data_frame.data.size())));

    _family_z_no = lvs.size();
    _z_no        = _latent_variables.get_z_list().size();

    // If Normal family is selected, we use faster likelihood functions
    if ("Normal" == _family->get_name()) {
        _model         = {[this](const Eigen::VectorXd& x) { return normal_model(x); }};
        _mb_model      = {[this](const Eigen::VectorXd& x, size_t mb) { return mb_normal_model(x, mb); }};
        _neg_loglik    = {[this](const Eigen::VectorXd& x) { return normal_neg_loglik(x); }};
        _mb_neg_loglik = {[this](const Eigen::VectorXd& x, size_t mb) { return normal_mb_neg_loglik(x, mb); }};
    }
    // else if (...) with missing models
    else {
        _model         = {[this](const Eigen::VectorXd& x) { return non_normal_model(x); }};
        _mb_model      = {[this](const Eigen::VectorXd& x, size_t mb) { return mb_non_normal_model(x, mb); }};
        _neg_loglik    = {[this](const Eigen::VectorXd& x) { return non_normal_neg_loglik(x); }};
        _mb_neg_loglik = {[this](const Eigen::VectorXd& x, size_t mb) { return non_normal_mb_neg_loglik(x, mb); }};
    }
}

template ARIMA::ARIMA(const std::vector<double>&, size_t, size_t, size_t, const std::optional<std::string>&,
                      const Family&);
template ARIMA::ARIMA(const utils::DataFrame&, size_t, size_t, size_t, const std::optional<std::string>&,
                      const Family&);

Eigen::MatrixXd ARIMA::ar_matrix() const {
    Eigen::MatrixXd X{Eigen::MatrixXd::Ones(static_cast<Eigen::Index>(_ar + 1),
                                            static_cast<Eigen::Index>(_data_length - _max_lag))};

    if (_ar != 0) {
        for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(_ar); ++i)
            std::copy(_data_frame.data.begin() + _max_lag - i - 1, _data_frame.data.end() - i - 1,
                      X.row(i + 1).begin());
    }

    return X;
}

ModelOutput ARIMA::categorize_model_output(const Eigen::VectorXd& z) const {
    auto mu_Y{_model(z)};
    return {mu_Y.first, mu_Y.second, {}, {}, {}, {}};
}

void ARIMA::create_latent_variables() {
    Normal n1{Normal(0, 3)};
    _latent_variables.add_z("Constant", n1, n1);

    n1 = Normal{0, 0.5};
    Normal n2{Normal(0.3)};
    for (size_t ar_terms{0}; ar_terms < _ar; ar_terms++)
        _latent_variables.add_z("AR(" + std::to_string(ar_terms + 1) + ")", n1, n2);

    for (size_t ma_terms{0}; ma_terms < _ma; ma_terms++)
        _latent_variables.add_z("MA(" + std::to_string(ma_terms + 1) + ")", n1, n2);
}

std::tuple<double, double, double> ARIMA::get_scale_and_shape(const Eigen::VectorXd& transformed_lvs) const {

    double model_shape{0}, model_scale{0}, model_skewness{0};

    if (_scale) {
        if (_shape) {
            model_shape = transformed_lvs(Eigen::last);
            model_scale = transformed_lvs(Eigen::last - 1);
        } else
            model_scale = transformed_lvs(Eigen::last);
        // if std::exp approximated the scale to zero we have to change it
        if (model_scale <= 0)
            model_scale = std::numeric_limits<double>::min();
    }

    if (_skewness)
        model_skewness = transformed_lvs(Eigen::last - 2);

    return std::make_tuple(model_scale, model_shape, model_skewness);
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
ARIMA::get_scale_and_shape_sim(const Eigen::MatrixXd& transformed_lvs) const {
    Eigen::VectorXd model_shape{Eigen::VectorXd::Zero(transformed_lvs.cols())},
            model_scale{Eigen::VectorXd::Zero(transformed_lvs.cols())},
            model_skewness{Eigen::VectorXd::Zero(transformed_lvs.cols())};

    if (_scale) {
        if (_shape) {
            // Apply trasform() to every element inside the matrix last row
            model_shape = transformed_lvs(Eigen::last, Eigen::all);
            std::transform(model_shape.begin(), model_shape.end(), model_shape.begin(),
                           [this](double n) { return _latent_variables.get_z_list().back().get_prior_transform()(n); });
            // Second last row for scale
            model_scale = transformed_lvs(Eigen::last - 1, Eigen::all);
            std::transform(model_scale.begin(), model_scale.end(), model_scale.begin(),
                           [this](double n) { return _latent_variables.get_z_list().at(-2).get_prior_transform()(n); });
        } else {
            // Last row for scale
            model_scale = transformed_lvs(Eigen::last, Eigen::all);
            std::transform(model_scale.begin(), model_scale.end(), model_scale.begin(),
                           [this](double n) { return _latent_variables.get_z_list().back().get_prior_transform()(n); });
        }
    }

    if (_skewness) {
        model_skewness = transformed_lvs(Eigen::last - 2, Eigen::all);
        std::transform(model_skewness.begin(), model_skewness.end(), model_skewness.begin(),
                       [this](double n) { return _latent_variables.get_z_list().at(-3).get_prior_transform()(n); });
    }

    return std::make_tuple(model_scale, model_shape, model_skewness);
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> ARIMA::normal_model(const Eigen::VectorXd& beta) const {
    Eigen::VectorXd Y(_data_length - _max_lag);
    std::copy(_data_frame.data.begin() + _max_lag, _data_frame.data.end(), Y.begin());

    // Transform latent variables
    Eigen::VectorXd z(beta.size());
    const std::vector<LatentVariable>& temp_z_list = _latent_variables.get_z_list();
    for (Eigen::Index i{0}; i < beta.size(); ++i) {
        z[i] = temp_z_list.at(i).get_prior_transform()(beta[i]);
    }

    // Constant and AR terms
    Eigen::VectorXd mu(_data_length - _max_lag);
    if (_ar != 0) {
        // We are not adding one to the Eigen index because seq includes the last element
        mu = _x.transpose() * z(Eigen::seq(0, Eigen::last - static_cast<Eigen::Index>(_family_z_no + _ma)));
    } else
        mu = Eigen::VectorXd::Ones(Y.size()) * z[0];

    // MA terms
    if (_ma != 0)
        arima_recursion_normal(z, mu, Y, _max_lag, Y.size(), _ar, _ma);

    return {mu, Y};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> ARIMA::non_normal_model(const Eigen::VectorXd& beta) const {
    std::vector<double> data;
    std::copy(_data_frame.data.begin() + _max_lag, _data_frame.data.end(), std::back_inserter(data));
    Eigen::VectorXd Y{Eigen::VectorXd::Map(data.data(), static_cast<Eigen::Index>(data.size()))};

    // Transform latent variables
    Eigen::VectorXd z(beta.size());
    const std::vector<LatentVariable>& temp_z_list = _latent_variables.get_z_list();
    for (Eigen::Index i{0}; i < beta.size(); ++i) {
        z[i] = temp_z_list.at(i).get_prior_transform()(beta[i]);
    }

    // Constant and AR terms
    Eigen::VectorXd mu;
    if (_ar != 0)
        // We are not adding one to the eigen index because seq includes the last element
        mu = _x.transpose() * z(Eigen::seq(0, Eigen::last - static_cast<Eigen::Index>(_family_z_no + _ma)));
    else
        mu = Eigen::VectorXd::Zero(Y.size()) * z[0];

    // MA terms
    if (_ma != 0) {
        Eigen::VectorXd link_mu(mu.size());
        std::transform(mu.begin(), mu.end(), link_mu.begin(), _link);
        arima_recursion(z, mu, link_mu, Y, _max_lag, Y.size(), _ar, _ma);
    }

    return {mu, Y};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> ARIMA::mb_normal_model(const Eigen::VectorXd& beta,
                                                                   size_t mini_batch) const {
    std::random_device r; // Seed with a real random value, if available
    std::default_random_engine e(r());
    std::uniform_int_distribution<size_t> uniform_dist(0, _data_frame.data.size() - mini_batch - _max_lag + 1);
    size_t rand_int = uniform_dist(e);
    std::vector<double> sample(mini_batch);
    std::iota(sample.begin(), sample.end(), rand_int);

    std::vector<double> data(_data_length - _max_lag);
    std::copy(_data_frame.data.begin() + _max_lag, _data_frame.data.end(), std::back_inserter(data));
    Eigen::VectorXd Y{Eigen::VectorXd::Map(data.data(), static_cast<Eigen::Index>(data.size()))};
    Y = Y(Eigen::all, sample);
    Eigen::MatrixXd X(_x.rows(), sample.size());
    for (Eigen::Index i{0}; i < _x.rows(); ++i)
        X.row(i) = _x.row(i)(Eigen::all, sample);

    // Transform latent variables
    Eigen::VectorXd z(beta.size());
    const std::vector<LatentVariable>& temp_z_list = _latent_variables.get_z_list();
    for (Eigen::Index i{0}; i < beta.size(); ++i) {
        z[i] = temp_z_list.at(i).get_prior_transform()(beta[i]);
    }

    // Constant and AR terms
    Eigen::VectorXd mu;
    if (_ar != 0)
        // We are not adding one to the eigen index because seq includes the last element
        mu = _x.transpose() * z(Eigen::seq(0, Eigen::last - static_cast<Eigen::Index>(_family_z_no + _ma)));
    else
        mu = Eigen::VectorXd::Zero(Y.size()) * z[0];

    // MA terms
    if (_ma != 0)
        arima_recursion_normal(z, mu, Y, _max_lag, Y.size(), _ar, _ma);

    return {mu, Y};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> ARIMA::mb_non_normal_model(const Eigen::VectorXd& beta,
                                                                       size_t mini_batch) const {
    std::random_device r; // Seed with a real random value, if available
    std::default_random_engine e(r());
    std::uniform_int_distribution<size_t> uniform_dist(0, _data_frame.data.size() - mini_batch - _max_lag + 1);
    size_t rand_int = uniform_dist(e);
    std::vector<double> sample(mini_batch);
    std::iota(sample.begin(), sample.end(), rand_int);

    std::vector<double> data(_data_length - _max_lag);
    std::copy(_data_frame.data.begin() + _max_lag, _data_frame.data.end(), std::back_inserter(data));
    Eigen::VectorXd Y{Eigen::VectorXd::Map(data.data(), static_cast<Eigen::Index>(data.size()))};
    Y = Y(Eigen::all, sample);
    Eigen::MatrixXd X(_x.rows(), sample.size());
    for (Eigen::Index i{0}; i < _x.rows(); ++i)
        X.row(i) = _x.row(i)(Eigen::all, sample);

    // Transform latent variables
    Eigen::VectorXd z(beta.size());
    const std::vector<LatentVariable>& temp_z_list = _latent_variables.get_z_list();
    for (Eigen::Index i{0}; i < beta.size(); ++i) {
        z[i] = temp_z_list.at(i).get_prior_transform()(beta[i]);
    }
    // Constant and AR terms
    Eigen::VectorXd mu;
    if (_ar != 0)
        // We are not adding one to the eigen index because seq includes the last element
        mu = _x.transpose() * z(Eigen::seq(0, Eigen::last - static_cast<Eigen::Index>(_family_z_no + _ma)));
    else
        mu = Eigen::VectorXd::Zero(Y.size()) * z[0];

    // MA terms
    if (_ma != 0) {
        Eigen::VectorXd link_mu(mu.size());
        std::transform(mu.begin(), mu.end(), link_mu.begin(), _link);
        arima_recursion(z, mu, link_mu, Y, _max_lag, Y.size(), _ar, _ma);
    }

    return {mu, Y};
}

double ARIMA::normal_neg_loglik(const Eigen::VectorXd& beta) const {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> mu_y = _model(beta);
    Eigen::VectorXd scale{{_latent_variables.get_prior_transform_back()(beta(Eigen::last))}};
    return -Mvn::logpdf(mu_y.second, mu_y.first, scale).sum();
}

double ARIMA::normal_mb_neg_loglik(const Eigen::VectorXd& beta, size_t mini_batch) const {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> mu_Y = _mb_model(beta, mini_batch);
    Eigen::VectorXd scale{{_latent_variables.get_prior_transform_back()(beta(Eigen::last))}};
    return -Mvn::logpdf(mu_Y.second, mu_Y.first, scale).sum();
}

double ARIMA::non_normal_neg_loglik(const Eigen::VectorXd& beta) const {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> mu_Y = _model(beta);
    Eigen::VectorXd transformed_parameters(beta.size());
    for (Eigen::Index k{0}; k < beta.size(); k++)
        transformed_parameters(k) = _latent_variables.get_prior_transform_at(k)(beta(k));
    auto scale_shape_skew = get_scale_and_shape(transformed_parameters);
    Eigen::VectorXd link_mu(mu_Y.first.size());
    std::transform(mu_Y.first.begin(), mu_Y.first.end(), link_mu.begin(), _link);
    return _family->neg_loglikelihood(mu_Y.second, link_mu, std::get<0>(scale_shape_skew));
}

double ARIMA::non_normal_mb_neg_loglik(const Eigen::VectorXd& beta, size_t mini_batch) const {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> mu_Y = _mb_model(beta, mini_batch);
    Eigen::VectorXd transformed_parameters(beta.size());
    for (Eigen::Index k{0}; k < beta.size(); k++)
        transformed_parameters(k) = _latent_variables.get_prior_transform_at(k)(beta(k));
    auto scale_shape_skew = get_scale_and_shape(transformed_parameters);
    Eigen::VectorXd link_mu(mu_Y.first.size());
    std::transform(mu_Y.first.begin(), mu_Y.first.end(), link_mu.begin(), _link);
    return _family->neg_loglikelihood(mu_Y.second, link_mu, std::get<0>(scale_shape_skew));
}

Eigen::VectorXd ARIMA::mean_prediction(const Eigen::VectorXd& mu, const Eigen::VectorXd& Y, size_t h,
                                       const Eigen::VectorXd& t_z) const {
    // Create arrays to iterate over
    Eigen::VectorXd Y_exp{Y};
    Eigen::VectorXd mu_exp{mu};

    // Loop over h time periods
    for (size_t t{0}; t < h; ++t) {
        double new_value = t_z[0];

        if (_ar != 0) {
            for (Eigen::Index i{1}; i <= static_cast<Eigen::Index>(_ar); ++i) {
                new_value += t_z[i] * Y_exp(Eigen::last - i + 1);
            }
        }

        if (_ma != 0) {
            for (Eigen::Index i{1}; i <= static_cast<Eigen::Index>(_ma); ++i) {
                if (i - 1 >= static_cast<Eigen::Index>(t))
                    new_value += t_z[i + static_cast<Eigen::Index>(_ar)] *
                                 (Y_exp(Eigen::last - i + 1) - _link(mu_exp(Eigen::last - i + 1)));
            }
        }

        std::vector<double> Y_exp_v(&Y_exp[0], Y_exp.data() + Y_exp.size());
        if (_model_name2 == "Exponential")
            Y_exp_v.push_back(1.0 / _link(new_value));
        else
            Y_exp_v.push_back(_link(new_value));
        Y_exp = Eigen::VectorXd::Map(Y_exp_v.data(), static_cast<Eigen::Index>(Y_exp_v.size()));

        // For indexing consistency
        std::vector<double> mu_exp_v(&mu_exp[0], mu_exp.data() + mu_exp.size());
        mu_exp_v.push_back(0.0);
        mu_exp = Eigen::VectorXd::Map(mu_exp_v.data(), static_cast<Eigen::Index>(mu_exp_v.size()));
    }

    // mu_exp = Eigen::VectorXd::Zero(mu_exp.size());

    return Y_exp;
}

Eigen::MatrixXd ARIMA::sim_prediction(const Eigen::VectorXd& mu, const Eigen::VectorXd& Y, size_t h,
                                      const Eigen::VectorXd& t_params, size_t simulations) const {
    auto scale_shape_skew{get_scale_and_shape(t_params)};

    Eigen::MatrixXd sim_vector{
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(simulations), static_cast<Eigen::Index>(h))};

    for (Eigen::Index n{0}; n < static_cast<Eigen::Index>(simulations); n++) {
        // Create arrays to iterate over
        Eigen::VectorXd Y_exp{Y};
        Eigen::VectorXd mu_exp{mu};

        // Loop over h time periods
        for (Eigen::Index t{0}; t < static_cast<Eigen::Index>(h); t++) {
            double new_value = t_params[0];

            if (_ar != 0) {
                for (Eigen::Index i{1}; i <= static_cast<Eigen::Index>(_ar); ++i)
                    new_value += t_params[i] * Y_exp(Eigen::last - i + 1);
            }

            if (_ma != 0) {
                for (Eigen::Index i{1}; i <= static_cast<Eigen::Index>(_ma); ++i) {
                    if (i - 1 >= t)
                        new_value += t_params[i + static_cast<Eigen::Index>(_ar)] *
                                     (Y_exp(Eigen::last - i + 1) - mu_exp(Eigen::last - i + 1));
                }
            }

            double rnd_value;
            if (_model_name2 == "Exponential")
                rnd_value = _family->draw_variable(1.0 / _link(new_value), std::get<0>(scale_shape_skew),
                                                   std::get<1>(scale_shape_skew), std::get<2>(scale_shape_skew), 1)[0];
            else
                rnd_value = _family->draw_variable(_link(new_value), std::get<0>(scale_shape_skew),
                                                   std::get<1>(scale_shape_skew), std::get<2>(scale_shape_skew), 1)[0];

            // Append rnd_value to Y_exp
            Eigen::VectorXd new_Y_exp(Y_exp.size() + 1);
            new_Y_exp << Y_exp, rnd_value;
            Y_exp = new_Y_exp;

            // Append 0.0 to mu_exp (for indexing consistency)
            Eigen::VectorXd new_mu_exp(mu_exp.size() + 1);
            new_mu_exp << mu_exp, 0.0;
            mu_exp = new_mu_exp;

            sim_vector.row(n) = Y_exp(Eigen::lastN(static_cast<Eigen::Index>(h)));
        }
    }

    return sim_vector.transpose();
}

Eigen::MatrixXd ARIMA::sim_prediction_bayes(size_t h, size_t simulations) const {
    Eigen::MatrixXd sim_vector{
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(simulations), static_cast<Eigen::Index>(h))};

    for (Eigen::Index n{0}; n < static_cast<Eigen::Index>(simulations); ++n) {
        Eigen::VectorXd t_z{draw_latent_variables(1).transpose().row(0)};

        /*
        bool keep_drawing = true;
        while (keep_drawing) {
            if (std::abs(t_z[0]) > std::abs(t_z[1] * 1e10))
                t_z = draw_latent_variables(1).transpose().row(0);
            else
                keep_drawing = false;
        }*/

        Eigen::VectorXd tz_copy{t_z};
        auto mu_Y{_model(t_z)};
        for (Eigen::Index i{0}; i < t_z.size(); ++i)
            t_z[i] = _latent_variables.get_z_list()[i].get_prior_transform()(t_z[i]);
        auto scale_shape_skew{get_scale_and_shape(t_z)};

        // Create arrays to iterate over
        Eigen::VectorXd Y_exp{mu_Y.second};
        Eigen::VectorXd mu_exp{mu_Y.first};

        // Loop over h time periods
        for (Eigen::Index t{0}; t < static_cast<Eigen::Index>(h); ++t) {
            double new_value = t_z[0];

            if (_ar != 0) {
                for (Eigen::Index i{1}; i <= static_cast<Eigen::Index>(_ar); ++i)
                    new_value += t_z[i] * Y_exp(Eigen::last - i + 1);
            }

            if (_ma != 0) {
                for (Eigen::Index i{1}; i <= static_cast<Eigen::Index>(_ma); ++i) {
                    if (i - 1 >= t)
                        new_value += t_z[i + static_cast<Eigen::Index>(_ar)] *
                                     (Y_exp(Eigen::last - i + 1) - mu_exp(Eigen::last - i + 1));
                }
            }

            std::vector<double> Y_exp_v(&Y_exp[0], Y_exp.data() + Y_exp.size());
            if (_model_name2 == "Exponential")
                Y_exp_v.push_back(_family->draw_variable(1.0 / _link(new_value), std::get<0>(scale_shape_skew),
                                                         std::get<1>(scale_shape_skew), std::get<2>(scale_shape_skew),
                                                         1)[0]);
            else
                Y_exp_v.push_back(_family->draw_variable(_link(new_value), std::get<0>(scale_shape_skew),
                                                         std::get<1>(scale_shape_skew), std::get<2>(scale_shape_skew),
                                                         1)[0]);

            std::vector<double> mu_exp_v(&mu_exp[0], mu_exp.data() + mu_exp.size());
            mu_exp_v.push_back(0.0);
            mu_exp = Eigen::VectorXd::Map(mu_exp_v.data(), static_cast<Eigen::Index>(mu_exp_v.size()));

            // For indexing consistency
            std::vector<double> Y_exp_h;
            std::copy(Y_exp_v.end() - static_cast<long>(h), Y_exp_v.end(), std::back_inserter(Y_exp_h));
            /*
             * FOR DEBUGGING PURPOSES
            double max_val = std::abs(Y_exp_h[std::distance(Y_exp_h.begin() , std::max_element(Y_exp_h.begin(),
            Y_exp_h.end(), utils::abs_compare))]); if( max_val > 1000) { std::cout << max_val << std::endl; for(auto& lv
            : _latent_variables.get_z_list()) { auto maxc = lv.get_sample()->maxCoeff(); auto minc =
            lv.get_sample()->minCoeff(); std::cout << tz_copy; std::cout << maxc; std::cout << minc;
                }
            }
             */
            sim_vector.row(n) = Eigen::VectorXd::Map(Y_exp_h.data(), static_cast<Eigen::Index>(Y_exp_h.size()));
        }

        // Y_exp = Eigen::VectorXd::Zero(Y_exp.size());
        // mu_exp = Eigen::VectorXd::Zero(mu_exp.size());
    }

    return sim_vector.transpose();
}

std::tuple<std::vector<std::vector<double>>, std::vector<double>, std::vector<double>, std::vector<double>>
ARIMA::summarize_simulations(const Eigen::VectorXd& mean_values, const Eigen::MatrixXd& sim_vector,
                             const std::vector<double>& date_index, size_t h, size_t past_values) const {
    std::vector<std::vector<double>> error_bars;
    for (size_t pre{5}; pre < 100; pre += 5) {
        std::vector error_bars_row{mean_values(Eigen::last - static_cast<Eigen::Index>(h))};
        for (Eigen::Index i{0}; i < sim_vector.rows(); ++i)
            error_bars_row.push_back(utils::percentile(sim_vector.row(i), pre));
        error_bars.push_back(error_bars_row);
    }

    std::vector<double> mv(&mean_values[0], mean_values.data() + mean_values.size());
    std::vector<double> forecasted_values;
    if (_latent_variables.get_estimation_method() == "M-H") {
        forecasted_values.push_back(mean_values[static_cast<Eigen::Index>(-h - 1)]);
        for (Eigen::Index i{0}; i < sim_vector.rows(); ++i)
            forecasted_values.push_back(utils::mean(sim_vector.row(i)));
    } else
        std::copy(mv.end() - static_cast<long>(h + 1), mv.end(), std::back_inserter(forecasted_values));

    std::vector<double> plot_values;
    std::copy(mv.end() - static_cast<long>(h + past_values), mv.end(), std::back_inserter(plot_values));
    std::vector<double> plot_index;
    std::copy(date_index.end() - static_cast<long>(h + past_values), date_index.end(), std::back_inserter(plot_index));

    return {error_bars, forecasted_values, plot_values, plot_index};
}

void ARIMA::plot_fit(std::optional<size_t> width, std::optional<size_t> height) const {
    std::vector<double> date_index;
    std::copy(_data_frame.index.begin() + static_cast<long>(std::max(_ar, _ma)),
              _data_frame.index.begin() + static_cast<long>(_data_length), std::back_inserter(date_index));
    auto mu_Y = _model(_latent_variables.get_z_values());

    // Catch specific family properties (imply different link functions/moments)
    std::vector<double> values_to_plot(mu_Y.first.size());
    if (_model_name2 == "Exponential")
        std::transform(mu_Y.first.begin(), mu_Y.first.end(), values_to_plot.begin(),
                       [this](double x) { return 1.0 / _link(x); });
    else if (_model_name2 == "Skewt") {
        Eigen::VectorXd t_params{transform_z()};
        auto scale_shape_skew{get_scale_and_shape(t_params)};
        double m1{
                (std::sqrt(std::get<1>(scale_shape_skew)) * std::tgamma((std::get<1>(scale_shape_skew) - 1.0) * 0.5)) /
                (std::sqrt(M_PI) * std::tgamma(std::get<1>(scale_shape_skew) * 0.5))};
        double additional_loc{(std::get<2>(scale_shape_skew) - (1.0 / std::get<2>(scale_shape_skew))) *
                              std::get<0>(scale_shape_skew) * m1};
        std::transform(mu_Y.first.begin(), mu_Y.first.end(), values_to_plot.begin(),
                       [additional_loc](double x) { return x + additional_loc; });
    } else
        std::transform(mu_Y.first.begin(), mu_Y.first.end(), values_to_plot.begin(),
                       [this](double x) { return _link(x); });

    std::vector<double> Y(&mu_Y.second[0], mu_Y.second.data() + mu_Y.second.size());

    // Create a Plot object
    sciplot::Plot plot;
    // Draw the data
    plot.drawCurve(date_index, Y).label("Data");
    plot.drawCurve(date_index, values_to_plot).label("ARIMA model").lineColor("black");
    // Set the size
    plot.size(width.value(), height.value());
    // Set the x and y labels
    plot.xlabel("Time");
    plot.ylabel(_data_frame.data_name);
    // Show the legend
    plot.legend().atTopRight().transparent();
    // Save the plot to a PDF file
    plot.save("../data/arima_plots/plot_fit.pdf");
    // Show the plot in a pop-up window
    plot.show();
}

void ARIMA::plot_predict(size_t h, size_t past_values, bool intervals, std::optional<size_t> width,
                         std::optional<size_t> height) const {
    assert(_latent_variables.is_estimated() && "No latent variables estimated!");

    auto mu_Y{_model(_latent_variables.get_z_values())};
    std::vector<double> date_index{shift_dates(h)};
    std::vector<std::vector<double>> error_bars;
    std::vector<double> forecasted_values;
    std::vector<double> plot_values;
    std::vector<double> plot_index;

    if (_latent_variables.get_estimation_method() == "M-H") {
        Eigen::MatrixXd sim_vector{sim_prediction_bayes(static_cast<Eigen::Index>(h), 1500)};
        std::vector<double> Y(&mu_Y.second[0], mu_Y.second.data() + mu_Y.second.size());

        for (size_t pre{5}; pre < 100; pre += 5) {
            std::vector<double> error_bars_row{Y.back()};
            for (Eigen::Index i{0}; i < sim_vector.rows(); ++i)
                error_bars_row.push_back(utils::percentile(sim_vector.row(i), pre));
            error_bars.push_back(error_bars_row);
        }

        forecasted_values.push_back(Y.back());
        for (Eigen::Index i{0}; i < sim_vector.rows(); ++i)
            forecasted_values.push_back(utils::mean(sim_vector.row(i)));

        std::copy(Y.end() - 1 - static_cast<long>(past_values), Y.end() - 2, std::back_inserter(plot_values));
        plot_values.insert(plot_values.end(), forecasted_values.begin(), forecasted_values.end());
        std::copy(date_index.end() - static_cast<long>(h + past_values), date_index.end(),
                  std::back_inserter(plot_index));
    } else {
        Eigen::VectorXd t_z{transform_z()};
        Eigen::VectorXd mean_values{mean_prediction(mu_Y.first, mu_Y.second, h, t_z)};
        Eigen::VectorXd fv = mean_values(Eigen::seq(Eigen::last - static_cast<Eigen::Index>(h) + 1, Eigen::last));
        std::copy(mean_values.end() - static_cast<long>(h), mean_values.end(), std::back_inserter(forecasted_values));

        if (_model_name2 == "Skewt") {
            auto scale_shape_skew{get_scale_and_shape(t_z)};
            double m1{(std::sqrt(std::get<1>(scale_shape_skew)) *
                       std::tgamma((std::get<1>(scale_shape_skew) - 1.0) * 0.5)) /
                      (std::sqrt(M_PI) * std::tgamma(std::get<1>(scale_shape_skew) * 0.5))};
            std::transform(forecasted_values.begin(), forecasted_values.end(), forecasted_values.begin(),
                           [scale_shape_skew, m1](double x) {
                               return x + (std::get<2>(scale_shape_skew) - 1.0 / std::get<2>(scale_shape_skew)) *
                                                  std::get<0>(scale_shape_skew) * m1;
                           });
        }

        Eigen::MatrixXd sim_values;
        if (intervals)
            sim_values = sim_prediction(mu_Y.first, mu_Y.second, h, t_z, 15000);
        else
            sim_values = sim_prediction(mu_Y.first, mu_Y.second, h, t_z, 2);

        auto results{summarize_simulations(mean_values, sim_values, date_index, static_cast<Eigen::Index>(h),
                                           static_cast<Eigen::Index>(past_values))};
        error_bars        = std::get<0>(results);
        forecasted_values = std::get<1>(results);
        plot_values       = std::get<2>(results);
        plot_index        = std::get<3>(results);
    }

    // Create a Plot object
    sciplot::Plot plot;
    // Draw the error zones
    if (intervals) {
        std::vector<double> alpha;
        for (size_t i{50}; i > 12; i -= 2)
            alpha.push_back(0.15 * static_cast<double>(i) * 0.01);
        std::vector<double> date_index_h;
        std::copy(date_index.end() - static_cast<long>(h) - 1, date_index.end(), std::back_inserter(date_index_h));
        for (size_t i{0}; i < error_bars.size() / 2; ++i)
            plot.drawCurvesFilled(date_index_h, error_bars[i], error_bars[error_bars.size() - i - 1])
                    .fillIntensity(alpha[i]);
    }
    // Draw the data
    plot.drawCurve(plot_index, plot_values);
    // Set the size
    plot.size(width.value(), height.value());
    // Set the x and y labels
    plot.xlabel("Time");
    plot.ylabel(_data_frame.data_name);
    // Hide the legend
    plot.legend().hide();
    // Save the plot to a PDF file
    plot.save("../data/arima_plots/plot_predict.pdf");
    // Show the plot in a pop-up window
    plot.show();
}


utils::DataFrame ARIMA::predict_is(size_t h, bool fit_once, const std::string& fit_method, bool intervals) const {
    utils::DataFrame predictions;
    LatentVariables saved_lvs{""};

    std::vector<std::string> names{_data_frame.data_name, "1% Prediction Interval", "5% Prediction Interval",
                                   "95% Prediction Interval", "99% Prediction Interval"};

    std::vector<double> index;
    utils::DataFrame new_prediction;
    for (Eigen::Index t{0}; t < static_cast<Eigen::Index>(h); t++) {
        std::vector<double> data_original_t{};
        std::copy(_data_original.begin(), _data_original.end() - static_cast<long>(h - t),
                  std::back_inserter(data_original_t));
        std::iota(index.begin(), index.end(), 0);
        ARIMA x{data_original_t, _ar, _ma, _integ, "", *_family};
        if (!fit_once) {
            Results* temp_r = x.fit(fit_method);
            delete temp_r;
        }
        if (t == 0) {
            if (fit_once) {
                Results* temp_r = x.fit(fit_method);
                saved_lvs       = x._latent_variables;
                delete temp_r;
            }
        } else {
            if (fit_once)
                x._latent_variables = saved_lvs;
        }
        new_prediction = x.predict(1, intervals);
        for (size_t i{0}; i < new_prediction.data_name.size(); ++i) {
            if (predictions.data.size() == i)
                predictions.data.push_back(new_prediction.data[i]);
            else
                predictions.data[i].insert(predictions.data[i].end(), new_prediction.data[i].begin(),
                                           new_prediction.data[i].end());
        }
    }

    std::copy(_data_frame.index.end() - static_cast<long>(h), _data_frame.index.end(),
              std::back_inserter(predictions.index));

    return predictions;
}

void ARIMA::plot_predict_is(size_t h, bool fit_once, const std::string& fit_method, std::optional<size_t> width,
                            std::optional<size_t> height) const {
    // Create a Plot object
    sciplot::Plot plot;
    // Draw the data and find the range of y values
    auto predictions{predict_is(h, fit_once, fit_method)};
    std::vector<double> data;
    std::copy(_data_frame.data.end() - static_cast<long>(h), _data_frame.data.end(), std::back_inserter(data));
    plot.drawCurve(predictions.index, data).label("Data");
    for (size_t i{0}; i < predictions.data.size(); ++i)
        plot.drawCurve(predictions.index, predictions.data[i]).label("Predictions").lineColor("black");
    // Set the size
    plot.size(width.value(), height.value());
    // Set the x and y labels
    plot.xlabel("Time");
    plot.ylabel(_data_frame.data_name);
    // Show the legend
    plot.legend().atTopRight().transparent();
    // Save the plot to a PDF file
    plot.save("../data/arima_plots/plot_predict_is.pdf");
    // Show the plot in a pop-up window
    plot.show();
}

utils::DataFrame ARIMA::predict(size_t h, bool intervals) const {
    assert(_latent_variables.is_estimated() && "No latent variables estimated!");

    auto mu_Y{_model(_latent_variables.get_z_values())};
    std::vector<double> date_index{shift_dates(h)};

    Eigen::MatrixXd sim_values{};
    utils::DataFrame result;
    std::vector<double> forecasted_values, prediction_01, prediction_05, prediction_95, prediction_99;
    if (_latent_variables.get_estimation_method() == "M-H") {
        sim_values = sim_prediction_bayes(h, 15000);
        for (Eigen::Index i{0}; i < sim_values.rows(); ++i) {
            forecasted_values.push_back(utils::mean(sim_values.row(i)));
            prediction_01.push_back(utils::percentile(sim_values.row(i), 1));
            prediction_05.push_back(utils::percentile(sim_values.row(i), 5));
            prediction_95.push_back(utils::percentile(sim_values.row(i), 95));
            prediction_99.push_back(utils::percentile(sim_values.row(i), 99));
        }
    } else {
        Eigen::VectorXd t_z{transform_z()};
        t_z = t_z.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
        Eigen::VectorXd mean_values{mean_prediction(mu_Y.first, mu_Y.second, h, t_z)};
        if (intervals)
            sim_values = sim_prediction(mu_Y.first, mu_Y.second, h, t_z, 15000);
        else
            sim_values = sim_prediction(mu_Y.first, mu_Y.second, h, t_z, 2);

        forecasted_values = std::vector<double>(h);
        if (_model_name2 == "Skewt") {
            auto scale_shape_skew{get_scale_and_shape(t_z)};
            double m1{(std::sqrt(std::get<1>(scale_shape_skew)) *
                       std::tgamma((std::get<1>(scale_shape_skew) - 1.0) * 0.5)) /
                      (std::sqrt(M_PI) * std::tgamma(std::get<1>(scale_shape_skew) * 0.5))};
            double additional_loc{(std::get<2>(scale_shape_skew) - (1.0 / std::get<2>(scale_shape_skew))) *
                                  std::get<0>(scale_shape_skew) * m1};
            std::transform(mean_values.end() - static_cast<long>(h), mean_values.end(), forecasted_values.begin(),
                           [additional_loc](double x) { return x + additional_loc; });
        } else
            std::copy(mean_values.end() - static_cast<long>(h), mean_values.end(), forecasted_values.begin());
    }
    if (!intervals) {
        result.data.emplace_back(forecasted_values);
        result.data_name.push_back(_data_frame.data_name);
    } else {
        if (_latent_variables.get_estimation_method() != "M-H") {
            // sim_values = sim_prediction(mu_Y.first, mu_Y.second, 5, t_z, 15000);
            for (Eigen::Index i{0}; i < sim_values.rows(); ++i) {
                prediction_01.push_back(utils::percentile(sim_values.row(i), 1));
                prediction_05.push_back(utils::percentile(sim_values.row(i), 5));
                prediction_95.push_back(utils::percentile(sim_values.row(i), 95));
                prediction_99.push_back(utils::percentile(sim_values.row(i), 99));
            }
        }
        result.data_name.push_back(_data_frame.data_name);
        result.data.push_back(forecasted_values);
        result.data_name.emplace_back("1% Prediction Interval");
        result.data.push_back(prediction_01);
        result.data_name.emplace_back("5% Prediction Interval");
        result.data.push_back(prediction_05);
        result.data_name.emplace_back("95% Prediction Interval");
        result.data.push_back(prediction_95);
        result.data_name.emplace_back("99% Prediction Interval");
        result.data.push_back(prediction_99);
    }
    std::copy(date_index.end() - static_cast<long>(h), date_index.end(), std::back_inserter(result.index));

    return result;
}

Eigen::MatrixXd ARIMA::sample(size_t nsims) const {
    assert((_latent_variables.get_estimation_method() == "BBVI" ||
            _latent_variables.get_estimation_method() == "M-H") &&
           "No latent variables estimated!");
    Eigen::MatrixXd lv_draws{draw_latent_variables(nsims)};
    std::vector<Eigen::VectorXd> mus;
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(nsims); ++i)
        mus.push_back(_model(lv_draws.col(i)).first);

    auto scale_shape_skew{get_scale_and_shape_sim(lv_draws)};
    Eigen::VectorXd temp_mus(mus.at(0).size());
    Eigen::MatrixXd data_draws(nsims, mus.at(0).size());
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(nsims); ++i) {
        std::transform(mus[i].begin(), mus[i].end(), temp_mus.begin(), _link);
        // Shape and skew are not used for Normal distributions
        data_draws.row(i) =
                _family->draw_variable(temp_mus, std::get<0>(scale_shape_skew)(i), std::get<1>(scale_shape_skew)(i),
                                       std::get<2>(scale_shape_skew)(i), static_cast<int>(mus.at(i).size()));
    }

    return data_draws;
}

void ARIMA::plot_sample(size_t nsims, bool plot_data, std::optional<size_t> width, std::optional<size_t> height) const {
    assert((_latent_variables.get_estimation_method() == "BBVI" ||
            _latent_variables.get_estimation_method() == "M-H") &&
           "No latent variables estimated!");

    // Create a Plot object
    sciplot::Plot plot;
    // Draw the data
    std::vector<double> date_index;
    std::copy(_data_frame.index.begin() + static_cast<long>(std::max(_ar, _ma)),
              _data_frame.index.begin() + static_cast<long>(_data_length), std::back_inserter(date_index));
    auto mu_Y = _model(_latent_variables.get_z_values());
    Eigen::MatrixXd draws{sample(nsims)};
    for (Eigen::Index i{0}; i < draws.rows(); ++i) {
        if (i == 0)
            plot.drawCurve(date_index, draws.row(i)).label("Posterior Draws");
        else
            plot.drawCurve(date_index, draws.row(i)).labelNone();
    }
    if (plot_data)
        plot.drawPoints(date_index, mu_Y.second).label("Data").lineColor("black").pointType(4);
    // Set the size
    plot.size(width.value(), height.value());
    // Set the x and y labels
    plot.xlabel("Time");
    plot.ylabel(_data_frame.data_name);
    // Show the legend
    plot.legend().atTopRight().transparent();
    // Save the plot to a PDF file
    plot.save("../data/arima_plots/plot_sample.pdf");
    // Show the plot in a pop-up window
    plot.show();
}

double ARIMA::ppc(size_t nsims, const std::function<double(Eigen::VectorXd)>& T) const {
    assert((_latent_variables.get_estimation_method() == "BBVI" ||
            _latent_variables.get_estimation_method() == "M-H") &&
           "No latent variables estimated!");

    Eigen::MatrixXd lv_draws{draw_latent_variables(nsims)};
    std::vector<Eigen::VectorXd> mus;
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(nsims); ++i)
        mus.push_back(_model(lv_draws.col(i)).first);

    auto scale_shape_skew{get_scale_and_shape_sim(lv_draws)};
    Eigen::VectorXd temp_mus(mus.at(0).size());
    Eigen::MatrixXd data_draws(nsims, mus.at(0).size());
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(nsims); ++i) {
        std::transform(mus[i].begin(), mus[i].end(), temp_mus.begin(), _link);
        // Shape and skew are not used for Normal distributions
        data_draws.row(i) =
                _family->draw_variable(temp_mus, std::get<0>(scale_shape_skew)(i), std::get<1>(scale_shape_skew)(i),
                                       std::get<2>(scale_shape_skew)(i), static_cast<int>(mus.at(i).size()));
    }

    Eigen::Matrix sample_data{sample(nsims)};
    std::vector<double> T_sims;
    for (Eigen::Index i{0}; i < sample_data.rows(); ++i)
        T_sims.push_back(T(sample_data.row(i)));
    double T_actual{
            T(Eigen::VectorXd::Map(_data_frame.data.data(), static_cast<Eigen::Index>(_data_frame.data.size())))};

    std::vector<double> T_sims_greater;
    for (size_t i{0}; i < T_sims.size(); ++i) {
        if (T_sims.at(i) > T_actual)
            T_sims_greater.push_back(T_sims.at(i));
    }

    return static_cast<double>(T_sims_greater.size()) / static_cast<double>(nsims);
}

void ARIMA::plot_ppc(size_t nsims, const std::function<double(Eigen::VectorXd)>& T, const std::string& T_name,
                     std::optional<size_t> width, std::optional<size_t> height) const {
    assert((_latent_variables.get_estimation_method() == "BBVI" ||
            _latent_variables.get_estimation_method() == "M-H") &&
           "No latent variables estimated!");

    Eigen::MatrixXd lv_draws{draw_latent_variables(nsims)};
    std::vector<Eigen::VectorXd> mus;
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(nsims); ++i)
        mus.push_back(_model(lv_draws.col(i)).first);

    Eigen::VectorXd temp_mus(mus.at(0).size());
    Eigen::MatrixXd data_draws(nsims, mus.at(0).size());
    auto scale_shape_skew{get_scale_and_shape_sim(lv_draws)};
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(nsims); ++i) {
        std::transform(mus[i].begin(), mus[i].end(), temp_mus.begin(), _link);
        data_draws.row(i) =
                _family->draw_variable(temp_mus, std::get<0>(scale_shape_skew)[i], std::get<1>(scale_shape_skew)[i],
                                       std::get<2>(scale_shape_skew)[i], static_cast<int>(temp_mus.size()));
    }

    Eigen::MatrixXd sample_data{sample(nsims)};
    std::vector<double> T_sims;
    std::vector<int64_t> hist_x{};
    std::vector<int64_t> hist_y{};
    for (Eigen::Index i{0}; i < sample_data.rows(); ++i) {
        double val = T(sample_data.row(i));
        T_sims.push_back(val);
        // Calculate the histogram of T_sims
        auto val_int = static_cast<int64_t>(val);
        auto it      = find(hist_x.begin(), hist_x.end(), val_int);
        if (it != hist_x.end())
            hist_y.at(std::distance(hist_x.begin(), it))++;
        else {
            hist_x.push_back(val_int);
            hist_y.push_back(1);
        }
    }
    double T_actual{
            T(Eigen::VectorXd::Map(_data_frame.data.data(), static_cast<Eigen::Index>(_data_frame.data.size())))};

    std::string description;
    if (T_name == "mean")
        description = " of the mean";
    else if (T_name == "max")
        description = " of the maximum";
    else if (T_name == "min")
        description = " of the minimum";
    else if (T_name == "median")
        description = " of the median";

    // Create a Plot object
    sciplot::Plot plot;
    // Draw the data and calculate the range of y values
    double max_y_value = static_cast<double>(*std::max_element(hist_y.begin(), hist_y.end()));
    plot.drawBoxes(hist_x, hist_y).label("Posterior predictive" + description);
    plot.boxWidthAbsolute(0.5);
    plot.drawCurve(std::vector<double>{T_actual, T_actual}, std::vector<double>{0, max_y_value + 10})
            .lineColor("red")
            .labelNone();
    // Set the size
    plot.size(width.value(), height.value());
    // Set the x and y labels
    plot.xlabel("T(x)");
    plot.ylabel("Frequency");
    // Show the legend
    plot.legend().atTopRight().transparent();
    // Save the plot to a PDF file
    plot.save("../data/arima_plots/plot_ppc.pdf");
    // Show the plot in a pop-up window
    plot.show();
}
