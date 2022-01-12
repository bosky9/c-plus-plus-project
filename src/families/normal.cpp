/**
 * @file normal.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "families/normal.hpp"

#include "Eigen/Core"              // Eigen::VectorXd, Eigen::MatrixXd, Eigen::Index
#include "families/family.hpp"     // Family, FamilyAttributes, lv_to_build
#include "families/flat.hpp"       // Flat
#include "multivariate_normal.hpp" // Mvn::random, Mvn::logpdf

#include <cassert>                 // static_assert, assert
#include <cmath>                   // log, pow, exp, round
#include <memory>                  // std::unique_ptr, std::make_unique
#include <string>                  // std::string, std::to_string
#include <type_traits>             // std::is_same_v
#include <utility>                 // std::pair
#include <vector>                  // std::vector

Normal::Normal(double mu, double sigma, const std::string& transform)
    : Family{transform}, _mu0{mu}, _sigma0{sigma}, _param_no{2}, _covariance_prior{false} {}

bool operator==(const Normal& normal1, const Normal& normal2) {
    return normal1._transform_name == normal2._transform_name && normal1._itransform_name == normal2._itransform_name &&
           normal1._mu0 == normal2._mu0 && normal1._sigma0 == normal2._sigma0 &&
           normal1._param_no == normal2._param_no && normal1._covariance_prior == normal2._covariance_prior;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> Normal::approximating_model(double h_approx, const Eigen::VectorXd& data) {
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> H_mu;
    Eigen::Index size = data.size();
    H_mu.first        = Eigen::MatrixXd::Constant(size, size, h_approx);
    H_mu.second       = Eigen::MatrixXd::Zero(size, size);
    return H_mu;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> Normal::approximating_model_reg(double h_approx,
                                                                            const Eigen::VectorXd& data) {
    // Currently identical to approximating_model()
    return approximating_model(h_approx, data);
}

std::vector<lv_to_build> Normal::build_latent_variables() const {
    std::vector<lv_to_build> lvs_to_build;
    lvs_to_build.push_back(lv_to_build{"Normal scale", new Flat{"exp"}, new Normal{0.0, 3.0}, 0.0});
    return lvs_to_build;
}

Eigen::VectorXd Normal::draw_variable(double loc, double scale, [[maybe_unused]] double shape, [[maybe_unused]] double skew, size_t nsims) const {
    return Mvn::random(loc, scale, nsims);
}

Eigen::VectorXd Normal::draw_variable(const Eigen::VectorXd& loc, double scale, [[maybe_unused]] double shape, [[maybe_unused]] double skew, size_t nsims) const {
    assert(static_cast<size_t>(loc.size()) == nsims &&
           "Vector of locations must be as long as the number of simulations");
    return Mvn::random(loc, scale, nsims);
}

Eigen::VectorXd Normal::draw_variable_local(size_t size) const {
    return Mvn::random(_mu0, _sigma0, size);
}

double Normal::logpdf(double mu) const {
    if (!_transform_name.empty())
        mu = _transform(mu);
    return -log(_sigma0) - (0.5 * pow(mu - _mu0, 2)) / pow(_sigma0, 2);
}

Eigen::VectorXd Normal::markov_blanket(const Eigen::VectorXd& y, const Eigen::VectorXd& mean, double scale) {
    return Mvn::logpdf(y, mean, Eigen::Vector<double, 1>{scale});
}

FamilyAttributes Normal::setup() const {
    return {"Normal", [](double x) { return x; }, true, false, false, [](double x) { return x; }};
}

double Normal::neg_loglikelihood(const Eigen::VectorXd& y, const Eigen::VectorXd& mean, double scale) const {
    assert(y.size() > 0 && "The time series must have at least one value.");
    return Normal::markov_blanket(y, mean, scale).sum();
}

double Normal::pdf(double mu) const {
    if (!_transform_name.empty()) // We need to transform mu if transform != ""
        mu = _transform(mu);
    return (1.0 / _sigma0) * exp(-((0.5 * pow(mu - _mu0, 2)) / pow(_sigma0, 2)));
}

void Normal::vi_change_param(uint8_t index, double value) {
    assert((index == 0 || index == 1) && "Index is neither 0 nor 1");
    if (index == 0)
        _mu0 = value;
    else if (index == 1)
        _sigma0 = exp(value);
}

double Normal::vi_return_param(uint8_t index) const {
    assert((index == 0 || index == 1) && "Index is neither 0 nor 1");
    if (index == 0)
        return _mu0;
    else if (index == 1)
        return log(_sigma0);
    return {};
}

// Template definition
template<typename T>
T Normal::vi_loc_score(const T& x) const {
    return (x - _mu0) / pow(_sigma0, 2);
}

// Template specializations
template double Normal::vi_loc_score<double>(const double& x) const;
template<>
Eigen::VectorXd Normal::vi_loc_score<Eigen::VectorXd>(const Eigen::VectorXd& x) const {
    return (x.array() - _mu0) / pow(_sigma0, 2);
}

// Template definition
template<typename T>
T Normal::vi_scale_score(const T& x) const {
    return exp(-2 * log(_sigma0)) * pow(x - _mu0, 2) - 1;
}

// Template specializations
template double Normal::vi_scale_score<double>(const double& x) const;
template<>
Eigen::VectorXd Normal::vi_scale_score<Eigen::VectorXd>(const Eigen::VectorXd& x) const {
    return exp(-2 * log(_sigma0)) * pow(x.array() - _mu0, 2) - 1;
}

// Template definition
template<typename T>
T Normal::vi_score(const T& x, uint8_t index) const {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, Eigen::VectorXd>,
            "Variable must be a double or an Eigen::VectorXd");
    assert((index == 0 || index == 1) && "Index is neither 0 nor 1");

    if (index == 0)
        return vi_loc_score(x);
    else
        return vi_scale_score(x);
}

// Template specializations
template double Normal::vi_score<double>(const double&, uint8_t) const;
template Eigen::VectorXd Normal::vi_score<Eigen::VectorXd>(const Eigen::VectorXd&, uint8_t) const;

// Get methods ----------------------------------------------------------------------------------------------------

double Normal::get_mu0() const {
    return _mu0;
}

std::string Normal::get_name() const {
    return "Normal";
}

uint8_t Normal::get_param_no() const {
    return _param_no;
}

double Normal::get_sigma0() const {
    return _sigma0;
}

std::string Normal::get_z_name() const {
    return "mu0: " + std::to_string(round(_mu0 * 10000) / 10000) +
           ", sigma0: " + std::to_string(round(_sigma0 * 10000) / 10000);
}

// Get methods ----------------------------------------------------------------------------------------------------

void Normal::set_mu0(double mu0) {
    _mu0 = mu0;
}

void Normal::set_sigma0(double sigma0) {
    _sigma0 = sigma0;
}

// Clone function
// ------------------------------------------------------------------------------------------------------

std::unique_ptr<Family> Normal::clone() const {
    return std::make_unique<Normal>(*this);
}
