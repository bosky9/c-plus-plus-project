/**
 * @file normal.cpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#include "families/normal.hpp"

#include "families/flat.hpp"       // Flat
#include "multivariate_normal.hpp" // Mvn::random()

Normal::Normal(double mu, double sigma, const std::string& transform)
    : Family{transform}, _mu0{mu}, _sigma0{sigma}, _param_no{2}, _covariance_prior{false} {}

Normal::Normal(const Normal& normal){
        _mu0 = normal._mu0;
        _sigma0 = normal._sigma0;
        _param_no = normal._param_no;
        _covariance_prior = normal._covariance_prior;
        _transform_name = normal._transform_name;
        _itransform_name = normal._itransform_name;
        _transform = normal._transform;
};

Normal::Normal(Normal&& normal) noexcept = default;

Normal& Normal::operator=(const Normal& normal) = default;

Normal& Normal::operator=(Normal&& normal) noexcept = default;

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

Eigen::VectorXd Normal::draw_variable(double loc, double scale, int64_t nsims) const {
    return Mvn::random(loc, scale, nsims);
}

Eigen::VectorXd Normal::draw_variable(const Eigen::VectorXd& loc, double scale, int64_t nsims) const {
    assert(loc.size() == nsims && "Vector of locations must be as long as the number of simulations");
    return Mvn::random(loc, scale, nsims);
}

Eigen::VectorXd Normal::draw_variable_local(int64_t size) const {
    return Mvn::random(_mu0, _sigma0, size);
}

double Normal::logpdf(double mu) const {
    if (!_transform_name.empty())
        mu = _transform(mu);
    return -log(_sigma0) - (0.5 * std::pow(mu - _mu0, 2)) / std::pow(_sigma0, 2);
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

double Normal::pdf(double mu) {
    if (!_transform_name.empty()) // We need to transform mu if transform != ""
        mu = _transform(mu);
    return (1.0 / _sigma0) * exp(-((0.5 * std::pow(mu - _mu0, 2)) / std::pow(_sigma0, 2)));
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

template<>
double Normal::vi_loc_score<double>(const double& x) const {
    return (x - _mu0) / pow(_sigma0, 2);
}

template<>
Eigen::VectorXd Normal::vi_loc_score<Eigen::VectorXd>(const Eigen::VectorXd& x) const {
    return (x.array() - _mu0) / pow(_sigma0, 2);
}

template<>
double Normal::vi_scale_score<double>(const double& x) const {
    return exp(-2 * log(_sigma0)) * pow(x - _mu0, 2) - 1;
}

template<>
Eigen::VectorXd Normal::vi_scale_score<Eigen::VectorXd>(const Eigen::VectorXd& x) const {
    return exp(-2 * log(_sigma0)) * pow(x.array() - _mu0, 2) - 1;
}

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

// Clone function
// ------------------------------------------------------------------------------------------------------

std::unique_ptr<Family>  Normal::clone() const {
    return std::make_unique<Normal>(*this);
}
