#include "families/normal.hpp"

Normal::Normal(double mu, double sigma, const std::string& transform)
    : Family{transform}, _mu0{mu}, _sigma0{sigma}, _param_no{2}, _covariance_prior{false} {}

Normal::Normal(const Normal& normal) : Family(normal) {
    _mu0              = normal._mu0;
    _sigma0           = normal._sigma0;
    _param_no         = normal._param_no;
    _covariance_prior = normal._covariance_prior;
}

Normal::Normal(Normal&& normal) noexcept : Family(std::move(normal)) {
    _mu0                     = normal._mu0;
    _sigma0                  = normal._sigma0;
    _param_no                = normal._param_no;
    _covariance_prior        = normal._covariance_prior;
    normal._mu0              = 0;
    normal._sigma0           = 0;
    normal._param_no         = 0;
    normal._covariance_prior = false;
}

Normal& Normal::operator=(const Normal& normal) {
    if (this == &normal)
        return *this;
    Family::operator  =(normal);
    _mu0              = normal._mu0;
    _sigma0           = normal._sigma0;
    _param_no         = normal._param_no;
    _covariance_prior = normal._covariance_prior;
    return *this;
}

Normal& Normal::operator=(Normal&& normal) noexcept {
    _mu0                     = normal._mu0;
    _sigma0                  = normal._sigma0;
    _param_no                = normal._param_no;
    _covariance_prior        = normal._covariance_prior;
    normal._mu0              = 0;
    normal._sigma0           = 0;
    normal._param_no         = 0;
    normal._covariance_prior = false;
    Family::operator         =(std::move(normal));
    return *this;
}

bool operator==(const Normal& normal1, const Normal& normal2) {
    return is_equal(normal1, normal2) && normal1._mu0 == normal2._mu0 && normal1._sigma0 == normal2._sigma0 &&
           normal1._param_no == normal2._param_no && normal1._covariance_prior == normal2._covariance_prior;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
Normal::approximating_model(const Eigen::VectorXd& beta, const Eigen::MatrixXd& T, const Eigen::MatrixXd& Z,
                            const Eigen::MatrixXd& R, const Eigen::MatrixXd& Q, double h_approx,
                            const Eigen::VectorXd& data) {
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> H_mu;
    auto size   = data.size();
    H_mu.first  = Eigen::MatrixXd::Constant(size, size, h_approx);
    H_mu.second = Eigen::MatrixXd::Zero(size, size);

    // Pointer to array (of size 2) of Eigen::MatrixXd
    return H_mu;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
Normal::approximating_model_reg(const Eigen::VectorXd& beta, const Eigen::MatrixXd& T, const Eigen::MatrixXd& Z,
                                const Eigen::MatrixXd& R, const Eigen::MatrixXd& Q, double h_approx,
                                const Eigen::VectorXd& data, const Eigen::VectorXd& X, int state_no) {
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> H_mu;
    auto size   = data.size();
    H_mu.first  = Eigen::MatrixXd::Constant(size, size, h_approx);
    H_mu.second = Eigen::MatrixXd::Zero(size, size);

    // Pointer to array (of size 2) of Eigen::MatrixXd
    return H_mu;
}

// Copy/move constructor may be needed
// What about the transform of the Normal?
std::vector<lv_to_build> Normal::build_latent_variables() const {
    std::vector<lv_to_build> lvs_to_build;
    lvs_to_build.push_back(
            lv_to_build{static_cast<std::string>("Normal scale"), new Flat("exp"), new Normal(0.0, 3.0), 0.0});
    return std::move(lvs_to_build); // return lvs_to_build
}

Eigen::VectorXd Normal::draw_variable(double loc, double scale, double shape, double skewness, int nsims) const {
    std::normal_distribution<double> my_normal{loc, scale}; // Uses the normal library function
    Eigen::VectorXd sims(nsims);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    for (Eigen::Index n = 0; n < nsims; n++)
        sims[n] = my_normal(gen);
    return sims;
}


Eigen::VectorXd Normal::draw_variable(const Eigen::VectorXd& loc, double scale, double shape, double skewness,
                                      int nsims) const {
    assert(loc.size() == nsims);
    // Uses the normal library function
    Eigen::VectorXd sims(nsims);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    for (Eigen::Index n = 0; n < nsims; n++) {
        std::normal_distribution<double> my_normal{loc[n], scale};
        sims[n] = my_normal(gen);
    }
    return sims;
}

Eigen::VectorXd Normal::draw_variable_local(size_t size) const {
    return Mvn::random(_mu0, _sigma0, size);
}

double Normal::logpdf(double mu) {
    if (!_transform_name.empty())
        mu = _transform(mu);
    return -log(_sigma0) - (0.5 * std::pow(mu - _mu0, 2)) / std::pow(_sigma0, 2);
}

Eigen::VectorXd Normal::markov_blanket(const Eigen::VectorXd& y, const Eigen::VectorXd& means, double scale,
                                       double shape, double skewness) {
    return Mvn::logpdf(y, means, Eigen::Vector<double, 1>{scale});
}

FamilyAttributes Normal::setup() const {
    return {"Normal", [](double x) { return x; }, true, false, false, [](double x) { return x; }, true};
}

double Normal::neg_loglikelihood(const Eigen::VectorXd& y, const Eigen::VectorXd& mean, double scale, double shape,
                                 double skewness) const {
    assert(y.size() > 0);
    return Normal::markov_blanket(y, mean, scale, shape, skewness).sum();
}

double Normal::pdf(double mu) {
    if (!_transform_name.empty()) // We need to transform mu if transform != ""
        mu = _transform(mu);
    return (1.0 / _sigma0) * exp(-((0.5 * std::pow(mu - _mu0, 2)) / std::pow(_sigma0, 2)));
}

void Normal::vi_change_param(size_t index, double value) {
    if (index == 0)
        _mu0 = value;
    else if (index == 1)
        _sigma0 = exp(value);
}

double Normal::vi_return_param(size_t index) const {
    if (index == 0)
        return _mu0;
    else if (index == 1)
        return log(_sigma0);
}

short unsigned int Normal::get_param_no() const {
    return _param_no;
}

bool Normal::get_covariance_prior() const {
    return _covariance_prior;
}

// vi_loc_score DEFINITION AND SPECIALIZATIONS -------------------------------------------------------------------------

template<typename T>
T Normal::vi_loc_score(const T& x) const {
    return (x - _mu0) / pow(_sigma0, 2);
}

template double Normal::vi_loc_score<double>(const double&) const;

template<>
Eigen::VectorXd Normal::vi_loc_score<Eigen::VectorXd>(const Eigen::VectorXd& x) const {
    return (x.array() - _mu0) / pow(_sigma0, 2);
}

// vi_scale_score DEFINITION AND SPECIALIZATIONS -----------------------------------------------------------------------

template<typename T>
T Normal::vi_scale_score(const T& x) const {
    return exp(-2 * log(_sigma0)) * pow(x - _mu0, 2) - 1;
}

template double Normal::vi_scale_score<double>(const double&) const;

template<>
Eigen::VectorXd Normal::vi_scale_score<Eigen::VectorXd>(const Eigen::VectorXd& x) const {
    return exp(-2 * log(_sigma0)) * pow(x.array() - _mu0, 2) - 1;
}

std::string Normal::get_name() const {
    return "Normal";
}

std::string Normal::get_z_name() const {
    return "mu0: " + std::to_string(round(_mu0 * 10000) / 10000) +
           ", sigma0: " + std::to_string(round(_sigma0 * 10000) / 10000);
}

Family* Normal::clone() const {
    return new Normal(*this);
}

double Normal::get_mu0() const {
    return _mu0;
}

double Normal::get_sigma0() const {
    return _sigma0;
}
