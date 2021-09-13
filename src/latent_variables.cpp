#include "latent_variables.hpp"

#include "output/tableprinter.hpp"
#include <list>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

LatentVariable::LatentVariable(std::string name, const Family& prior, Family q)
    : _name{std::move(name)}, _prior{prior}, _transform{prior.get_transform()}, _q{std::move(q)} {}

void LatentVariable::plot_z(double width, double height) {
    assert(_sample || (_value && _std));
    if (_sample) {
        // TODO
    } else {
        // TODO
    }
}

std::string LatentVariable::get_method() const {
    return _method;
}

std::string LatentVariable::get_name() const {
    return _name;
}

Family LatentVariable::get_prior() const {
    return _prior;
}

std::optional<std::vector<double>> LatentVariable::get_sample() const {
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

Family LatentVariable::get_q() const {
    return _q;
}

void LatentVariable::set_prior(const Family& prior) {
    _prior = prior;
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

void LatentVariable::set_sample(const std::vector<double>& sample) {
    _sample = sample;
}

LatentVariables::LatentVariables(std::string model_name) : _model_name{std::move(model_name)} {
    _z_list    = {};
    _z_indices = {};
}

inline std::ostream& operator<<(std::ostream& stream, const LatentVariables& lvs) {
    std::vector<std::string> z_names                                             = lvs.get_z_names();
    std::vector<Family> priors                                                   = lvs.get_z_priors();
    std::pair<std::vector<std::string>, std::vector<std::string>> z_priors_names = lvs.get_z_priors_names();
    std::vector<std::string> prior_names                                         = z_priors_names.first;
    std::vector<std::string> prior_z_names                                       = z_priors_names.second;
    std::vector<std::string> vardist_names                                       = lvs.get_z_approx_dist_names();
    std::vector<std::string> transforms                                          = lvs.get_z_transforms_names();

    std::list<std::tuple<std::string, std::string, int>> fmt = {
            std::make_tuple("Index", "z_index", 8),        std::make_tuple("Latent Variable", "z_name", 25),
            std::make_tuple("Prior", "z_prior", 15),       std::make_tuple("Prior Hyperparameters", "z_hyper", 25),
            std::make_tuple("V.I. Dist", "z_vardist", 10), std::make_tuple("Transform", "z_transform", 10)};

    std::list<std::map<std::string, std::string>> z_row;
    for (size_t z{0}; z < lvs._z_list.size(); z++)
        z_row.push_back({{"z_index", std::to_string(z)},
                         {"z_name", z_names.at(z)},
                         {"z_prior", prior_names.at(z)},
                         {"z_hyper", prior_z_names.at(z)},
                         {"z_vardist", vardist_names.at(z)},
                         {"z_transform", transforms.at(z)}});

    stream << TablePrinter(fmt, " ", "=")._call_(z_row);
    return stream;
}

void LatentVariables::add_z(const std::string& name, const Family& prior, const Family& q, bool index) {
    LatentVariable lv{name, prior, q};
    _z_list.push_back(std::move(lv));
    if (index)
        _z_indices[name] = {{"start", _z_list.size() - 1}, {"end", _z_list.size() - 1}};
}

void LatentVariables::create(const std::string& name, const std::vector<size_t>& dim, const Family& prior,
                             const Family& q) {
    // Initialize indices vector
    size_t indices_dim = std::accumulate(dim.begin(), dim.end(), 1, std::multiplies<>());
    std::vector<std::string> indices(indices_dim, "(");
    for (Eigen::Index d{0}; d < dim.size(); d++) {
        Eigen::Index span     = std::accumulate(dim.begin() + d + 1, dim.end(), 1, std::multiplies<>());
        std::string separator = (d == dim.size() - 1) ? "," : ")";
        for (size_t index{0}; index < indices_dim; index++)
            indices.at(index) += std::to_string(index / span) + separator;
    }

    size_t starting_index = _z_list.size();
    _z_indices[name] = {{"start", starting_index}, {"end", starting_index + indices.size() - 1}, {"dim", dim.size()}};
    for (const std::string& index : indices)
        add_z(name + " " + index, prior, q, false);
}

void LatentVariables::adjust_prior(const std::vector<size_t>& index, const Family& prior) {
    for (size_t item : index) {
        assert(item > _z_list.size() - 1);
        _z_list.at(item).set_prior(prior);
        if (auto mu0 = _z_list.at(item).get_prior().get_mu0())
            _z_list.at(item).set_start(mu0.value());
        if (auto loc0 = _z_list.at(item).get_prior().get_loc0())
            _z_list.at(item).set_start(loc0.value());
    }
}

std::vector<std::string> LatentVariables::get_z_names() const {
    std::vector<std::string> names;
    for (const LatentVariable& z : _z_list)
        names.push_back(z.get_name());
    return names;
}

std::vector<Family> LatentVariables::get_z_priors() const {
    std::vector<Family> priors;
    for (const LatentVariable& z : _z_list)
        priors.push_back(z.get_prior());
    return priors;
}

std::pair<std::vector<std::string>, std::vector<std::string>> LatentVariables::get_z_priors_names() const {
    std::vector<Family> priors = get_z_priors();
    std::vector<std::string> prior_names;
    std::vector<std::string> prior_z_names;
    for (const Family& prior : priors) {
        prior_names.push_back(prior.get_name());
        prior_z_names.push_back(prior.get_z_name());
    }
    return {prior_names, prior_z_names};
}

std::vector<std::function<double(double)>> LatentVariables::get_z_transforms() const {
    std::vector<std::function<double(double)>> transforms;
    for (const LatentVariable& z : _z_list)
        transforms.push_back(z.get_prior().get_transform());
    return transforms;
}

std::vector<std::string> LatentVariables::get_z_transforms_names() const {
    std::vector<std::string> transforms;
    for (const LatentVariable& z : _z_list)
        transforms.push_back(z.get_prior().get_transform_name());
    return transforms;
}

Eigen::VectorXd LatentVariables::get_z_starting_values(bool transformed) const {
    std::vector<std::function<double(double)>> transforms = get_z_transforms();
    Eigen::VectorXd values(_z_list.size());
    for (Eigen::Index i{0}; i < _z_list.size(); i++) {
        values(i) = transformed ? transforms.at(i)(_z_list.at(i).get_start()) : _z_list.at(i).get_start();
    }
    return values;
}

Eigen::VectorXd LatentVariables::get_z_values(bool transformed) const {
    assert(_estimated);
    std::vector<std::function<double(double)>> transforms = get_z_transforms();
    Eigen::VectorXd values(_z_list.size());
    for (Eigen::Index i{0}; i < _z_list.size(); i++) {
        assert(_z_list.at(i).get_value());
        values(i) =
                transformed ? transforms.at(i)(_z_list.at(i).get_value().value()) : _z_list.at(i).get_value().value();
    }
    return values;
}

std::vector<Family> LatentVariables::get_z_approx_dist() const {
    std::vector<Family> dists;
    for (const LatentVariable& z : _z_list)
        dists.push_back(z.get_q());
    return dists;
}

std::vector<std::string> LatentVariables::get_z_approx_dist_names() const {
    std::vector<Family> approx_dists = get_z_approx_dist();
    std::vector<std::string> q_list(approx_dists.size());
    for (const Family& approx : approx_dists)
        q_list.emplace_back((approx.get_name() == "Normal") ? "Normal" : "Approximate distribution not detected");
    return q_list;
}

void LatentVariables::set_z_values(const std::vector<double>& values, const std::string& method,
                                   const std::optional<std::vector<double>>& stds,
                                   const std::optional<std::vector<std::vector<double>>>& samples) {
    assert(values.size() == _z_list.size());
    for (size_t i{0}; i < _z_list.size(); i++) {
        _z_list.at(i).set_method(method);
        _z_list.at(i).set_value(values.at(i));
        if (stds)
            _z_list.at(i).set_std(stds.value().at(i));
        if (samples)
            _z_list.at(i).set_sample(samples.value().at(i));
    }
    _estimated = true;
}

void LatentVariables::set_z_starting_values(const std::vector<double>& values) {
    assert(values.size() == _z_list.size());
    for (size_t i{0}; i < _z_list.size(); i++) {
        _z_list.at(i).set_start(values.at(i));
    }
}

void LatentVariables::plot_z(const std::optional<std::vector<size_t>>& indices, size_t width, size_t height, int loc) {
    plt::figure_size(width, height);
    for (size_t z = 0; z < _z_list.size(); z++) {
        assert(!_z_list[z].get_sample().has_value() ||
               !(_z_list[z].get_value().has_value() && _z_list[z].get_std().has_value()) &&
                       "No information on latent variable to plot!");
        if (!indices.has_value() ||
            std::find(indices.value().begin(), indices.value().end(), z) == indices.value().end()) {
            std::function<double(double)> transform = _z_list[z].get_prior().get_transform();
            if (_z_list[z].get_sample().has_value()) {
                std::vector<double> x{_z_list[z].get_sample().value()};
                std::transform(x.begin(), x.end(), x.begin(), [transform](double n) { return transform(n); });
                plt::named_plot(_z_list[z].get_method() + " estimate of " + _z_list[z].get_name(), x);
            } else if (_z_list[z].get_value().has_value() && _z_list[z].get_std().has_value()) {
                if (_z_list[z].get_prior().get_transform_name() == "") {
                    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(
                            100, _z_list[z].get_value().value() - _z_list[z].get_std().value() * 3.5,
                            _z_list[z].get_value().value() + _z_list[z].get_std().value() * 3.5);
                    std::vector<double> x_v{&x[0], x.data()};
                    Eigen::VectorXd y = Mvn::pdf(x, _z_list[z].get_value().value(), _z_list[z].get_std().value());
                    std::vector<double> y_v{&y[0], y.data()};
                    plt::named_plot(_z_list[z].get_method() + " estimate of " + _z_list[z].get_name(), x_v, y_v);
                } else {
                    Eigen::VectorXd sims{
                            Mvn::random(_z_list[z].get_value().value(), _z_list[z].get_std().value(), 100000)};
                    std::vector<double> sims_v{&sims[0], sims.data()};
                    std::transform(sims_v.begin(), sims_v.end(), sims_v.begin(),
                                   [transform](double n) { return transform(n); });
                    plt::named_plot(_z_list[z].get_method() + " estimate of " + _z_list[z].get_name(), sims_v);
                }
            }
        }
    }
    plt::xlabel("Value");
    plt::ylabel("Frequency");
    plt::title("Latent Variable Plot");
    plt::legend(std::map<std::string, std::string>{{"loc", "upper right"}});
    plt::save("../data/plot_z.png");
    plt::show();
}

void LatentVariables::trace_plot(size_t width, size_t height) {
    assert(_z_list[0].get_sample().has_value() && "No samples to plot!");
    plt::figure_size(width, height);
    // FIXME: Non esiste una funzione plot in cui passare il colore come tupla di 3 valori (solo scatter_colored)
    /*std::vector<std::vector<double>> palette{{0.2980392156862745, 0.4470588235294118, 0.6901960784313725},
                                             {0.3333333333333333, 0.6588235294117647, 0.40784313725490196},
                                             {0.7686274509803922, 0.3058823529411765, 0.3215686274509804},
                                             {0.5058823529411764, 0.4470588235294118, 0.6980392156862745},
                                             {0.8, 0.7254901960784313, 0.4549019607843137},
                                             {0.39215686274509803, 0.7098039215686275, 0.803921568627451}};
    std::transform(palette.begin(), palette.end(), palette.begin(), [this](std::vector<double> v) {
        std::transform(v.begin(), v.end(), v.begin(), [this](double x) { return x * _z_list.size(); });
    });*/
    std::vector<std::string> palette{"royalblue",    "mediumseagreen", "chocolate",
                                     "mediumpurple", "goldenrod",      "skyblue"};

    for (size_t i = 0; i < _z_list.size(); i++) {
        std::vector<double> chain = _z_list[i].get_sample().value();
        for (size_t j = 0; j < 4; j++) {
            size_t iteration = i * 4 + j + 1;
            plt::subplot(static_cast<long>(_z_list.size()), 4, static_cast<long>(iteration));
            if (iteration >= 1 && iteration <= _z_list.size() * 4 + 1) {
                std::function<double(double)> transform = _z_list[i].get_prior().get_transform();
                if (iteration % 4 == 1) {
                    std::vector<double> x{chain};
                    std::transform(x.begin(), x.end(), x.begin(), [transform](double n) { return transform(n); });
                    plt::plot(x, palette[i]);
                    plt::ylabel(_z_list[i].get_name());
                    if (iteration == 1)
                        plt::title("Density Estimate");
                } else if (iteration % 4 == 2) {
                    std::vector<double> x{chain};
                    std::transform(x.begin(), x.end(), x.begin(), [transform](double n) { return transform(n); });
                    plt::plot(x, palette[i]);
                    if (iteration == 2)
                        plt::title("Trace Plot");
                } else if (iteration % 4 == 3) {
                    std::vector<double> x{chain};
                    std::transform(x.begin(), x.end(), x.begin(), [transform](double n) { return transform(n); });
                    std::partial_sum(x.begin(), x.end(), x.begin());
                    std::vector<double> indices(chain.size());
                    std::iota(indices.begin(), indices.end(), 1);
                    std::vector<double> result(chain.size());
                    std::transform(x.begin(), x.end(), indices.begin(), result.begin(), std::divides<>());
                    plt::plot(x, palette[i]);
                    if (iteration == 3)
                        plt::title("Cumulative Average");
                } else if (iteration % 4 == 0) {
                    std::vector<double> x(9);
                    std::iota(x.begin(), x.end(), 1);
                    std::vector<double> y(9);
                    for (size_t lag = 1; lag < 10; lag++) {
                        Eigen::VectorXd eigen_chain =
                                Eigen::VectorXd::Map(chain.data(), static_cast<Eigen::Index>(chain.size()));
                        y[lag - 1] = acf(eigen_chain, lag);
                    }
                    plt::bar(x, y, palette[i]);
                    if (iteration == 4)
                        plt::title("ACF Plot");
                }
            }
        }
    }
    plt::save("../data/trace_plot.png");
    // plt::show();
}