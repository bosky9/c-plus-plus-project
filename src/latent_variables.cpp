#include "latent_variables.hpp"

#include "Eigen/Core"          // Eigen::VectorXd, Eigen::Index
#include "covariances.hpp"     // acf
#include "families/family.hpp" // Family
#include "families/normal.hpp" // Normal
#include "matplotlibcpp.hpp" // plt::named_plot, plt::subplot, plt::figure_size, plt::xlabel, plt::ylabel, plt::title, plt::legend, plt::save, plt::show
#include "multivariate_normal.hpp" // Mvn::pdf, Mvn::random
#include "output/tableprinter.hpp" // TablePrinter
#include "utilities.hpp"           // isinstance

#include <algorithm>  // std::transform, std::find
#include <cmath>      // ceil
#include <functional> // std::function, std::multiplies, std::divides
#include <list>       // std::list
#include <map>        // std::map
#include <memory>     // std::unique_ptr
#include <numeric>    // std::accumulate, std::partial_sum, std::iota
#include <optional>   // std::optional, std::nullopt
#include <ostream>    // std::ostream
#include <sciplot/Figure.hpp>
#include <sciplot/Plot.hpp>
#include <string>  // std::string, std::to_string
#include <tuple>   // std::tuple
#include <utility> // std::pair, std::move
#include <vector>  // std::vector

LatentVariable::LatentVariable(std::string name, const Family& prior, const Family& q)
    : _name{std::move(name)}, _index{0}, _prior{prior.clone()},
      _transform{prior.get_transform()}, _start{0.0}, _q{q.clone()} {}

LatentVariable::LatentVariable(const LatentVariable& lv)
    : _name{lv.get_name()}, _index{lv._index}, _prior{lv.get_prior_clone()->clone()},
      _transform{lv._transform}, _start{lv.get_start()}, _q{lv.get_q_clone()}, _method{lv.get_method()},
      _value{lv.get_value()}, _std{lv.get_std()}, _sample{lv.get_sample()} {}

LatentVariable::LatentVariable(LatentVariable&& lv) noexcept {
    _name      = lv.get_name();
    _index     = lv._index;
    _prior     = lv.get_prior_clone();
    _transform = lv._transform;
    _start     = lv.get_start();
    _q         = lv.get_q_clone();
    _method    = lv.get_method();
    _value     = lv.get_value();
    _std       = lv.get_std();
    _sample    = lv.get_sample();
    lv._name   = "";
    lv._index  = 0;
    lv._prior.reset();
    lv._transform = {};
    lv._start     = 0;
    lv._q.reset();
    lv._method = "";
    lv._value  = std::nullopt;
    lv._std    = std::nullopt;
    lv._sample = std::nullopt;
}

LatentVariable& LatentVariable::operator=(const LatentVariable& lv) {
    if (this == &lv)
        return *this;
    _name  = lv.get_name();
    _index = lv._index;
    _prior.reset();
    _prior     = lv.get_prior_clone();
    _transform = lv._transform;
    _start     = lv.get_start();
    _q.reset();
    _q      = lv.get_q_clone();
    _method = lv.get_method();
    _value  = lv.get_value();
    _std    = lv.get_std();
    _sample = lv.get_sample();
    return *this;
}

LatentVariable& LatentVariable::operator=(LatentVariable&& lv) noexcept {
    _name  = lv.get_name();
    _index = lv._index;
    _prior.reset();
    _prior     = lv.get_prior_clone();
    _transform = lv._transform;
    _start     = lv.get_start();
    _q.reset();
    _q        = lv.get_q_clone();
    _method   = lv.get_method();
    _value    = lv.get_value();
    _std      = lv.get_std();
    _sample   = lv.get_sample();
    lv._name  = "";
    lv._index = 0;
    lv._prior.reset();
    lv._transform = {};
    lv._start     = 0;
    lv._q.reset();
    lv._method = "";
    lv._value  = std::nullopt;
    lv._std    = std::nullopt;
    lv._sample = std::nullopt;
    return *this;
}

LatentVariable::~LatentVariable() = default;

void LatentVariable::plot_z(size_t width, size_t height) const {
    assert((_sample.has_value() || (_value.has_value() && _std.has_value())) &&
           "No information on latent variables to plot!");
    std::function<double(double)> transform = _prior->get_transform();
    // Create a Plot object
    sciplot::Plot plot;
    // Draw the data
    Eigen::VectorXd x;
    if (_sample.has_value()) {
        Eigen::VectorXd sample{_sample.value()};
        std::transform(sample.begin(), sample.end(), sample.begin(), [transform](double n) { return transform(n); });
        // Calculate mean and std of y
        double mean = sample.mean();
        std::vector<double> diff(sample.size());
        std::transform(sample.begin(), sample.end(), diff.begin(), [mean](double x) { return x - mean; });
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        double std    = std::sqrt(sq_sum / sample.size());
        // Calculate and draw the pdf
        x = Eigen::VectorXd::LinSpaced(100, mean - std * 3.5, mean + std * 3.5);
        Eigen::VectorXd y(x.size());
        Normal distribution{mean, std};
        std::transform(x.begin(), x.end(), y.begin(), [distribution](double n) { return distribution.pdf(n); });
        plot.drawCurve(x, y).label(_method + " estimate of " + _name);
    } else if (_value.has_value() && _std.has_value()) {
        x = Eigen::VectorXd::LinSpaced(100, _value.value() - _std.value() * 3.5, _value.value() + _std.value() * 3.5);
        if (_prior->get_transform_name().empty()) {
            // Plot the pdf of a normal distribution with the mean and std of the latent variable
            Eigen::VectorXd y(x.size());
            Normal distribution{_value.value(), _std.value()};
            std::transform(x.begin(), x.end(), y.begin(), [distribution](double n) { return distribution.pdf(n); });
            plot.drawCurve(x, y).label(_method + " estimate of " + _name);
        } else {
            // Plot the pdf of simulated (transformed) values
            Eigen::VectorXd sims{Mvn::random(_value.value(), _std.value(), 100000)};
            std::transform(sims.begin(), sims.end(), sims.begin(), [transform](double n) { return transform(n); });
            Eigen::VectorXd y(x.size());
            // Calculate mean and std of sims
            double mean = sims.mean();
            std::vector<double> diff(sims.size());
            std::transform(sims.begin(), sims.end(), diff.begin(), [mean](double x) { return x - mean; });
            double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            double std    = std::sqrt(sq_sum / sims.size());
            // Calculate and draw the pdf
            Normal distribution{mean, std};
            std::transform(x.begin(), x.end(), y.begin(), [distribution](double n) { return distribution.pdf(n); });
            plot.drawCurve(x, y).label(_method + " estimate of " + _name);
        }
    }
    // Set the size
    plot.size(width, height);
    // Set the x label
    plot.xlabel("Value");
    // Show the legend
    plot.legend().atTopRight().transparent();
    // Save the plot to a PDF file
    plot.save("../data/latent_variables_plots/plot_z_single.png");
    // Show the plot in a pop-up window
    plot.show();
}

std::string LatentVariable::get_method() const {
    return _method;
}

std::string LatentVariable::get_name() const {
    return _name;
}

std::unique_ptr<Family> LatentVariable::get_prior_clone() const {
    return (_prior->clone());
}

std::function<double(double)> LatentVariable::get_prior_transform() const {
    return _prior->get_transform();
}


std::optional<Eigen::VectorXd> LatentVariable::get_sample() const {
    return _sample;
}

double LatentVariable::get_start() const {
    return _start;
}

std::optional<double> LatentVariable::get_std() const {
    return _std;
}

std::optional<double> LatentVariable::get_value() const {
    return _value;
}

std::unique_ptr<Family> LatentVariable::get_q_clone() const {
    return _q->clone();
}

void LatentVariable::change_q_param(uint8_t index, double value) {
    _q->vi_change_param(index, value);
}


void LatentVariable::set_prior(const Family& prior) {
    _prior.reset();
    _prior = prior.clone();
}

void LatentVariable::set_start(double start) {
    _start = start;
}

void LatentVariable::set_method(const std::string& method) {
    _method = method;
}

void LatentVariable::set_value(double value) {
    _value = value;
}

void LatentVariable::set_std(double std) {
    _std = std;
}

void LatentVariable::set_sample(const Eigen::VectorXd& sample) {
    _sample = sample;
}

void LatentVariable::set_q(const Family& q) {
    _q = (q.clone());
}

void LatentVariable::set_mu0_q(double mu0) {
    if (utils::isinstance<Normal>(_q.get()))
        dynamic_cast<Normal*>(_q.get())->set_mu0(mu0);
}

void LatentVariable::set_sigma0_q(double sigma0) {
    if (utils::isinstance<Normal>(_q.get()))
        dynamic_cast<Normal*>(_q.get())->set_sigma0(sigma0);
}

LatentVariables::LatentVariables(std::string model_name)
    : _model_name{std::move(model_name)}, _z_list{}, _z_indices{} {}

inline std::ostream& operator<<(std::ostream& stream, const LatentVariables& lvs) {
    std::vector<std::string> z_names{lvs.get_z_names()};
    std::pair<std::vector<std::string>, std::vector<std::string>> z_priors_names{lvs.get_z_priors_names()};
    std::vector<std::string> prior_names{z_priors_names.first};
    std::vector<std::string> prior_z_names{z_priors_names.second};
    std::vector<std::string> vardist_names{lvs.get_z_approx_dist_names()};
    std::vector<std::string> transforms{lvs.get_z_transforms_names()};

    std::vector<std::tuple<std::string, std::string, int>> fmt = {
            {"Index", "z_index", 8},        {"Latent Variable", "z_name", 25},
            {"Prior", "z_prior", 15},       {"Prior Hyperparameters", "z_hyper", 25},
            {"V.I. Dist", "z_vardist", 10}, {"Transform", "z_transform", 10}};

    std::list<std::map<std::string, std::string>> z_row;
    for (size_t z{0}; z < lvs._z_list.size(); z++)
        z_row.push_back({{"z_index", std::to_string(z)},
                         {"z_name", z_names[z]},
                         {"z_prior", prior_names[z]},
                         {"z_hyper", prior_z_names[z]},
                         {"z_vardist", vardist_names[z]},
                         {"z_transform", transforms[z]}});

    stream << TablePrinter{fmt, " ", "="}(z_row);
    return stream;
}

void LatentVariables::add_z(const std::string& name, const Family& prior, const Family& q, bool index) {
    LatentVariable lv{name, prior, q};
    _z_list.push_back(std::move(lv));
    if (index)
        _z_indices[name] = {{"start", _z_list.size() - 1}, {"end", _z_list.size() - 1}};
}

void LatentVariables::create(const std::string& name, const std::vector<size_t>& dim, const Family& q,
                             const Family& prior) {
    // Initialize indices vector
    // tot size of elements (it's a tree in Python)
    size_t indices_dim = std::accumulate(dim.begin(), dim.end(), 1, std::multiplies<>());
    std::vector<std::string> indices(indices_dim, "("); // Creates a vector of indices_dim (

    size_t previous_span  = indices_dim;
    size_t previous_value = 1;

    for (Eigen::Index d{0}; d < static_cast<Eigen::Index>(dim.size()); d++) {
        // span is the remaining length
        size_t span           = previous_span / previous_value;
        size_t current_dim    = dim.at(d);
        size_t divide_by      = span / current_dim;
        std::string separator = (d == static_cast<Eigen::Index>(dim.size()) - 1) ? "," : ")";
        // append these fractions to each string of indices
        for (size_t indx{1}; indx < indices_dim; indx++) {
            indices[indx] += std::to_string(ceil(indx / divide_by)) + separator;
            if (indx >= span)
                indx = 1;
        }
    }

    size_t starting_index = _z_list.size();
    _z_indices[name] = {{"start", starting_index}, {"end", starting_index + indices.size() - 1}, {"dim", dim.size()}};
    for (const std::string& index : indices)
        add_z(name + " " + index, prior, q, false);
}

void LatentVariables::adjust_prior(const std::vector<size_t>& index, const Family& prior) {
    for (size_t item : index) {
        assert(item > _z_list.size() - 1);
        _z_list[item].set_prior(prior);
        if (prior.get_name() == "Normal") {
            _z_list[item].set_start(_z_list[item].get_prior_clone()->vi_return_param(0));
            _z_list[item].set_start(_z_list[item].get_prior_clone()->vi_return_param(1));
        }
    }
}

std::vector<LatentVariable> LatentVariables::get_z_list() const {
    return _z_list;
}

void LatentVariables::update_z_list_q(size_t position, uint8_t index, double value) {
    _z_list[position].change_q_param(index, value);
}

std::vector<std::string> LatentVariables::get_z_names() const {
    std::vector<std::string> names;
    for (const LatentVariable& z : _z_list)
        names.push_back(z.get_name());
    return names;
}

std::vector<std::unique_ptr<Family>> LatentVariables::get_z_priors() const {
    std::vector<std::unique_ptr<Family>> priors;
    for (const LatentVariable& z : _z_list)
        priors.push_back(z.get_prior_clone());
    return priors;
}

std::pair<std::vector<std::string>, std::vector<std::string>> LatentVariables::get_z_priors_names() const {
    std::vector<std::unique_ptr<Family>> priors = get_z_priors();
    std::vector<std::string> prior_names;
    std::vector<std::string> prior_z_names;
    for (const auto& prior : priors) {
        prior_names.push_back(prior->get_name());
        prior_z_names.push_back(prior->get_z_name());
    }
    return {prior_names, prior_z_names};
}

std::vector<std::function<double(double)>> LatentVariables::get_z_transforms() const {
    std::vector<std::function<double(double)>> transforms;
    for (const LatentVariable& z : _z_list)
        transforms.push_back(z.get_prior_clone()->get_transform());
    return transforms;
}

std::vector<std::string> LatentVariables::get_z_transforms_names() const {
    std::vector<std::string> transforms;
    for (const LatentVariable& z : _z_list)
        transforms.push_back(z.get_prior_clone()->get_transform_name());
    return transforms;
}

Eigen::VectorXd LatentVariables::get_z_starting_values(bool transformed) const {
    std::vector<std::function<double(double)>> transforms = get_z_transforms();
    Eigen::VectorXd values(_z_list.size());
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(_z_list.size()); i++) {
        values(i) = transformed ? transforms[i](_z_list[i].get_start()) : _z_list[i].get_start();
    }
    return values;
}

Eigen::VectorXd LatentVariables::get_z_values(bool transformed) const {
    assert(_estimated);
    std::vector<std::function<double(double)>> transforms = get_z_transforms();
    Eigen::VectorXd values{Eigen::VectorXd::Zero(static_cast<Eigen::Index>(_z_list.size()))};
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(_z_list.size()); ++i) {
        assert(_z_list[i].get_value().has_value());
        values(i) = transformed ? transforms[i](_z_list[i].get_value().value()) : _z_list[i].get_value().value();
        // std::cout << _z_list[i].get_value().value() << " vs " << transforms[i](_z_list[i].get_value().value()) <<
        // "\n";
    }
    // std::cout << "------------\n\n";
    return values;
}

std::vector<std::unique_ptr<Family>> LatentVariables::get_z_approx_dist() const {
    std::vector<std::unique_ptr<Family>> dists;
    for (const LatentVariable& z : _z_list)
        dists.push_back(z.get_q_clone());
    return dists;
}

std::vector<std::string> LatentVariables::get_z_approx_dist_names() const {
    std::vector<std::unique_ptr<Family>> approx_dists = get_z_approx_dist();
    std::vector<std::string> q_list(approx_dists.size());
    for (auto& approx : approx_dists)
        q_list.emplace_back((utils::isinstance<Normal>(approx.get())) ? "Normal"
                                                                      : "Approximate distribution not detected");
    return q_list;
}

bool LatentVariables::is_estimated() const {
    return _estimated;
}

std::optional<std::string> LatentVariables::get_estimation_method() const {
    return _estimation_method;
}

void LatentVariables::set_estimation_method(const std::string& method) {
    _estimation_method = method;
}

void LatentVariables::set_z_values(const Eigen::VectorXd& values, const std::string& method,
                                   const std::optional<Eigen::VectorXd>& stds,
                                   const std::optional<Eigen::MatrixXd>& samples) {
    assert(static_cast<size_t>(values.size()) == _z_list.size());
    for (size_t i{0}; i < _z_list.size(); ++i) {
        _z_list[i].set_method(method);
        _z_list[i].set_value(values[static_cast<Eigen::Index>(i)]);
        if (stds.has_value())
            _z_list[i].set_std(stds.value()[static_cast<Eigen::Index>(i)]);
        if (samples.has_value())
            _z_list[i].set_sample(samples.value().row(static_cast<Eigen::Index>(i)));
    }
    _estimated = true;
}

void LatentVariables::set_z_starting_values(const Eigen::VectorXd& values) {
    assert(static_cast<size_t>(values.size()) == _z_list.size());
    for (size_t i{0}; i < _z_list.size(); ++i) {
        _z_list[i].set_start(values[static_cast<Eigen::Index>(i)]);
    }
}

void LatentVariables::set_z_starting_value(size_t index, double value) {
    assert(index < _z_list.size());
    _z_list.at(index).set_start(value);
}

void LatentVariables::set_mu0_q(size_t index, double mu0) {
    _z_list.at(index).set_mu0_q(mu0);
}

void LatentVariables::set_sigma0_q(size_t index, double sigma0) {
    _z_list.at(index).set_sigma0_q(sigma0);
}

void LatentVariables::set_z_qs(const std::vector<std::unique_ptr<Family>>& qs) {
    assert(_z_list.size() == qs.size());
    for (size_t i{0}; i < _z_list.size(); i++)
        _z_list.at(i).set_q(*qs.at(i));
}

void LatentVariables::plot_z(const std::optional<std::vector<size_t>>& indices, size_t width, size_t height) const {
    // Create a Plot object
    sciplot::Plot plot;
    // Draw the data
    for (size_t z{0}; z < _z_list.size(); z++) {
        assert((!_z_list[z].get_sample().has_value() ||
                !(_z_list[z].get_value().has_value() && _z_list[z].get_std().has_value())) &&
               "No information on latent variable to plot!");
        if (!indices.has_value() ||
            std::find(indices.value().begin(), indices.value().end(), z) == indices.value().end()) {
            std::function<double(double)> transform = _z_list[z].get_prior_clone()->get_transform();
            if (_z_list[z].get_sample().has_value()) {
                Eigen::VectorXd sample = _z_list[z].get_sample().value();
                std::transform(sample.begin(), sample.end(), sample.begin(),
                               [transform](double n) { return transform(n); });
                // Calculate mean and std of sample
                double mean = sample.mean();
                std::vector<double> diff(sample.size());
                std::transform(sample.begin(), sample.end(), diff.begin(), [mean](double x) { return x - mean; });
                double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
                double std    = std::sqrt(sq_sum / sample.size());
                // Calculate and draw the pdf of the sample values
                Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(100, mean - std * 3.5, mean + std * 3.5);
                Eigen::VectorXd y(x.size());
                Normal distribution{mean, std};
                std::transform(x.begin(), x.end(), y.begin(), [distribution](double n) { return distribution.pdf(n); });
                plot.drawCurve(x, y).label(_z_list[z].get_method() + " estimate of " + _z_list[z].get_name());
            } else if (_z_list[z].get_value().has_value() && _z_list[z].get_std().has_value()) {
                if (_z_list[z].get_prior_clone()->get_transform_name().empty()) {
                    // Calculate and draw the pdf of a normal distribution with mean and std of the latent variable
                    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(
                            100, _z_list[z].get_value().value() - _z_list[z].get_std().value() * 3.5,
                            _z_list[z].get_value().value() + _z_list[z].get_std().value() * 3.5);
                    Eigen::VectorXd y(x.size());
                    Normal distribution{_z_list[z].get_value().value(), _z_list[z].get_std().value()};
                    std::transform(x.begin(), x.end(), y.begin(),
                                   [distribution](double n) { return distribution.pdf(n); });
                    plot.drawCurve(x, y).label(_z_list[z].get_method() + " estimate of " + _z_list[z].get_name());
                } else {
                    Eigen::VectorXd sims{
                            Mvn::random(_z_list[z].get_value().value(), _z_list[z].get_std().value(), 100000)};
                    std::transform(sims.begin(), sims.end(), sims.begin(),
                                   [transform](double n) { return transform(n); });
                    // Calculate mean and std of sims
                    double mean = sims.mean();
                    std::vector<double> diff(sims.size());
                    std::transform(sims.begin(), sims.end(), diff.begin(), [mean](double x) { return x - mean; });
                    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
                    double std    = std::sqrt(sq_sum / sims.size());
                    // Calculate and draw the pdf of the simulated values
                    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(100, mean - std * 3.5, mean + std * 3.5);
                    Eigen::VectorXd y(x.size());
                    Normal distribution{mean, std};
                    std::transform(x.begin(), x.end(), y.begin(),
                                   [distribution](double n) { return distribution.pdf(n); });
                    plot.drawCurve(x, y).label(_z_list[z].get_method() + " estimate of " + _z_list[z].get_name());
                }
            }
        }
    }
    // Set the size
    plot.size(width, height);
    // Set the x label
    plot.xlabel("Value");
    plot.ylabel("Frequency");
    // Show the legend
    plot.legend().atTopRight().transparent();
    // Save the plot to a PDF file
    plot.save("../data/latent_variables_plots/plot_z.png");
    // Show the plot in a pop-up window
    plot.show();
}

void LatentVariables::trace_plot(size_t width, size_t height) {
    assert(_z_list[0].get_sample().has_value() && "No samples to plot!");
    // A color name based plot is used instead of a RGB color based plot
    std::vector<std::string> palette{"royalblue", "green", "orange", "purple", "yellow", "skyblue"};
    std::vector<std::vector<sciplot::PlotVariant>> plots;
    // For each latent variable
    for (size_t i{0}; i < _z_list.size(); ++i) {
        std::vector<sciplot::PlotVariant> row_plots;
        // Transform the sample values of the latent variable
        Eigen::VectorXd chain{_z_list[i].get_sample().value()};
        std::function<double(double)> transform = _z_list[i].get_prior_clone()->get_transform();
        Eigen::VectorXd transformed_chain(chain.size());
        std::transform(chain.begin(), chain.end(), transformed_chain.begin(),
                       [transform](double n) { return transform(n); });
        // Draw 4 different plots
        for (size_t j{0}; j < 4; j++) {
            // Create a Plot object
            sciplot::Plot plot;
            // Draw the data
            if (j == 0) {
                // Calculate and draw the pdf of transformed_chain
                // TODO: Draw the pdf instead the histogram
                plot.drawHistogram(transformed_chain).label("Density Estimate").lineColor(palette[i]);
                plot.ylabel(_z_list[i].get_name());
                plot.legend().atTopRight().transparent();
            } else if (j == 1) {
                // Draw transformed_chain
                Eigen::VectorXd x(transformed_chain.size());
                std::iota(x.begin(), x.end(), 0);
                plot.drawCurve(x, transformed_chain).label("Trace Plot").lineColor(palette[i]);
            } else if (j == 2) {
                // Calculate and draw the cumulative average
                Eigen::VectorXd y(transformed_chain.size());
                std::partial_sum(transformed_chain.begin(), transformed_chain.end(), y.begin());
                std::vector<double> to_divide(transformed_chain.size());
                std::iota(to_divide.begin(), to_divide.end(), 1);
                std::transform(y.begin(), y.end(), to_divide.begin(), y.begin(), std::divides<>());
                Eigen::VectorXd x(y.size());
                std::iota(x.begin(), x.end(), 0);
                plot.drawCurve(x, y).label("Cumulative Average").lineColor(palette[i]);
            } else if (j == 3) {
                // Calculate and draw the ACF
                Eigen::VectorXd x(9);
                std::iota(x.begin(), x.end(), 1);
                Eigen::VectorXd y(9);
                std::transform(x.begin(), x.end(), y.begin(), [chain](size_t n) { return acf(chain, n); });
                plot.drawBoxes(x, y).label("ACF Plot").fillSolid().fillColor(palette[i]);
            }
            row_plots.emplace_back(plot);
        }
        plots.push_back(row_plots);
    }
    // Create the figure containing all plots
    sciplot::Figure fig(std::as_const(plots));
    // Set the size
    fig.size(width, height);
    // Save the figure to a PDF file
    fig.save("../data/latent_variables_plots/trace_plot.png");
    fig.show();
}
std::function<double(double)> LatentVariables::get_prior_transform_at(const size_t &i) const {
    return _z_list.at(i).get_prior_transform();
}

std::function<double(double)> LatentVariables::get_prior_transform_back() const {
    return _z_list.back().get_prior_transform();
}
