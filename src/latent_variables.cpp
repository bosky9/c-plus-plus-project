#include "latent_variables.hpp"

LatentVariable::LatentVariable(std::string name, const Family& prior, const Family& q)
    : _name{std::move(name)}, _prior{prior.clone()}, _index{0},
      _transform{prior.get_transform()}, _start{0.0}, _q{q.clone()} {}

LatentVariable::LatentVariable(const LatentVariable& lv)
    : _name{lv.get_name()}, _index{lv._index}, _prior{lv.get_prior()},
      _transform{lv._transform}, _start{lv.get_start()}, _q{lv.get_q()}, _method{lv.get_method()},
      _value{lv.get_value()}, _std{lv.get_std()}, _sample{lv.get_sample()} {}

LatentVariable::LatentVariable(LatentVariable&& lv) noexcept {
    _name  = lv.get_name();
    _index = lv._index;
    _prior.reset(lv.get_prior());
    _transform = lv._transform;
    _start     = lv.get_start();
    _q.reset(lv.get_q());
    _method       = lv.get_method();
    _value        = lv.get_value();
    _std          = lv.get_std();
    _sample       = lv.get_sample();
    lv._name      = "";
    lv._index     = 0;
    lv._prior     = nullptr;
    lv._transform = {};
    lv._start     = 0;
    lv._q         = nullptr;
    lv._method    = "";
    lv._value     = std::nullopt;
    lv._std       = std::nullopt;
    lv._sample    = std::nullopt;
}

LatentVariable& LatentVariable::operator=(const LatentVariable& lv) {
    if (this == &lv)
        return *this;
    _name  = lv.get_name();
    _index = lv._index;
    _prior.reset(lv.get_prior());
    _transform = lv._transform;
    _start     = lv.get_start();
    _q.reset(lv.get_q());
    _method = lv.get_method();
    _value  = lv.get_value();
    _std    = lv.get_std();
    _sample = lv.get_sample();
    return *this;
}

LatentVariable& LatentVariable::operator=(LatentVariable&& lv) noexcept {
    _name  = lv.get_name();
    _index = lv._index;
    _prior.reset(lv.get_prior());
    _transform = lv._transform;
    _start     = lv.get_start();
    _q.reset(lv.get_q());
    _method       = lv.get_method();
    _value        = lv.get_value();
    _std          = lv.get_std();
    _sample       = lv.get_sample();
    lv._name      = "";
    lv._index     = 0;
    lv._prior     = nullptr;
    lv._transform = {};
    lv._start     = 0;
    lv._q         = nullptr;
    lv._method    = "";
    lv._value     = std::nullopt;
    lv._std       = std::nullopt;
    lv._sample    = std::nullopt;
    return *this;
}

LatentVariable::~LatentVariable() = default;

void LatentVariable::plot_z(size_t width, size_t height) {
    assert((_sample.has_value() || (_value.has_value() && _std.has_value())) &&
           "No information on latent variables to plot!");
    std::function<double(double)> transform = _prior->get_transform();
    if (_sample.has_value()) {
        std::vector<double> x{&_sample.value()[0], _sample.value().data()};
        std::transform(x.begin(), x.end(), x.begin(), [transform](double n) { return transform(n); });
        plt::named_plot(_method + " estimate of " + _name, x);
    } else if (_value.has_value() && _std.has_value()) {
        plt::figure_size(width, height);
        if (_prior->get_transform_name().empty()) {
            Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(100, _value.value() - _std.value() * 3.5,
                                                           _value.value() + _std.value() * 3.5);
            std::vector<double> x_v{&x[0], x.data()};
            Eigen::VectorXd y = Mvn::pdf(x, _value.value(), _std.value());
            std::vector<double> y_v{&y[0], y.data()};
            plt::named_plot(_method + " estimate of " + _name, x_v, y_v);
        } else {
            Eigen::VectorXd sims{Mvn::random(_value.value(), _std.value(), 100000)};
            std::vector<double> sims_v{&sims[0], sims.data()};
            std::transform(sims_v.begin(), sims_v.end(), sims_v.begin(),
                           [transform](double n) { return transform(n); });
            plt::named_plot(_method + " estimate of " + _name, sims_v);
        }
    }
    plt::xlabel("Value");
    plt::legend();
    plt::save("../data/plot_z_single.png");
    // plt::show();
}

std::string LatentVariable::get_method() const {
    return _method;
}

std::string LatentVariable::get_name() const {
    return _name;
}

Family* LatentVariable::get_prior() const {
    return _prior->clone();
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

Family* LatentVariable::get_q() const {
    return _q->clone();
}

void LatentVariable::set_prior(const Family& prior) {
    _prior.reset(prior.clone());
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

void LatentVariable::set_q(Family* q) {
    _q.reset(q);
}

LatentVariables::LatentVariables(std::string model_name)
    : _model_name{std::move(model_name)}, _z_list{}, _z_indices{} {}

inline std::ostream& operator<<(std::ostream& stream, const LatentVariables& lvs) {
    std::vector<std::string> z_names{lvs.get_z_names()};
    // std::vector<Family*> priors{lvs.get_z_priors()};
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

void LatentVariables::add_z(const std::string& name, Family* prior, Family* q, bool index) {
    LatentVariable lv{name, *prior, *q};
    _z_list.push_back(std::move(lv));
    if (index)
        _z_indices[name] = {{"start", _z_list.size() - 1}, {"end", _z_list.size() - 1}};
}

void LatentVariables::create(const std::string& name, const std::vector<size_t>& dim, const Family& q,
                             const Family& prior) {
    // Initialize indices vector
    size_t indices_dim = std::accumulate(dim.begin(), dim.end(), 1, std::multiplies<>());
    std::vector<std::string> indices(indices_dim, "(");
    for (Eigen::Index d{0}; d < dim.size(); d++) {
        Eigen::Index span     = std::accumulate(dim.begin() + d + 1, dim.end(), 1, std::multiplies<>());
        std::string separator = (d == dim.size() - 1) ? "," : ")";
        for (size_t index{0}; index < indices_dim; index++)
            indices[index] += std::to_string(index / span) + separator;
    }

    size_t starting_index = _z_list.size();
    _z_indices[name] = {{"start", starting_index}, {"end", starting_index + indices.size() - 1}, {"dim", dim.size()}};
    for (const std::string& index : indices)
        add_z(name + " " + index, prior, q, false);
}

void LatentVariables::adjust_prior(const std::vector<size_t>& index, Family& prior) {
    for (size_t item : index) {
        assert(item > _z_list.size() - 1);
        _z_list[item].set_prior(prior);
        if (typeid(prior).name() == "Normal") { // TODO: Controllare che funzioni correttamente
            _z_list[item].set_start(_z_list[item].get_prior()->vi_return_param(0));
            _z_list[item].set_start(_z_list[item].get_prior()->vi_return_param(1));
        }
    }
}

std::vector<LatentVariable> LatentVariables::get_z_list() const {
    return _z_list;
}

std::vector<std::string> LatentVariables::get_z_names() const {
    std::vector<std::string> names;
    for (const LatentVariable& z : _z_list)
        names.push_back(z.get_name());
    return names;
}

std::vector<Family*> LatentVariables::get_z_priors() const {
    std::vector<Family*> priors;
    for (const LatentVariable& z : _z_list)
        priors.push_back(z.get_prior());
    return priors;
}

std::pair<std::vector<std::string>, std::vector<std::string>> LatentVariables::get_z_priors_names() const {
    std::vector<Family*> priors = get_z_priors();
    std::vector<std::string> prior_names;
    std::vector<std::string> prior_z_names;
    for (const Family* prior : priors) {
        prior_names.push_back(prior->get_name());
        prior_z_names.push_back(prior->get_z_name());
    }
    return {prior_names, prior_z_names};
}

std::vector<std::function<double(double)>> LatentVariables::get_z_transforms() const {
    std::vector<std::function<double(double)>> transforms;
    for (const LatentVariable& z : _z_list)
        transforms.push_back(z.get_prior()->get_transform());
    return transforms;
}

std::vector<std::string> LatentVariables::get_z_transforms_names() const {
    std::vector<std::string> transforms;
    for (const LatentVariable& z : _z_list)
        transforms.push_back(z.get_prior()->get_transform_name());
    return transforms;
}

Eigen::VectorXd LatentVariables::get_z_starting_values(bool transformed) const {
    std::vector<std::function<double(double)>> transforms = get_z_transforms();
    Eigen::VectorXd values(_z_list.size());
    for (Eigen::Index i{0}; i < _z_list.size(); i++) {
        values(i) = transformed ? transforms[i](_z_list[i].get_start()) : _z_list[i].get_start();
    }
    return values;
}

Eigen::VectorXd LatentVariables::get_z_values(bool transformed) const {
    assert(_estimated);
    std::vector<std::function<double(double)>> transforms = get_z_transforms();
    Eigen::VectorXd values(_z_list.size());
    for (Eigen::Index i{0}; i < _z_list.size(); i++) {
        assert(_z_list[i].get_value());
        values(i) = transformed ? transforms[i](_z_list[i].get_value().value()) : _z_list[i].get_value().value();
    }
    return values;
}

std::vector<Family*> LatentVariables::get_z_approx_dist() const {
    std::vector<Family*> dists;
    for (const LatentVariable& z : _z_list)
        dists.push_back(z.get_q());
    return dists;
}

std::vector<std::string> LatentVariables::get_z_approx_dist_names() const {
    std::vector<Family*> approx_dists = get_z_approx_dist();
    std::vector<std::string> q_list(approx_dists.size());
    for (const Family* approx : approx_dists)
        q_list.emplace_back((approx->get_name() == "Normal") ? "Normal" : "Approximate distribution not detected");
    return q_list;
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
    assert(values.size() == _z_list.size());
    for (size_t i{0}; i < _z_list.size(); i++) {
        _z_list[i].set_method(method);
        _z_list[i].set_value(values[static_cast<Eigen::Index>(i)]);
        if (stds)
            _z_list[i].set_std(stds.value()[static_cast<Eigen::Index>(i)]);
        if (samples.has_value())
            _z_list[i].set_sample(samples.value().row(static_cast<Eigen::Index>(i)));
    }
    _estimated = true;
}

void LatentVariables::set_z_starting_values(const Eigen::VectorXd& values) {
    assert(values.size() == _z_list.size());
    for (size_t i{0}; i < _z_list.size(); i++) {
        _z_list[i].set_start(values[static_cast<Eigen::Index>(i)]);
    }
}

void LatentVariables::set_z_starting_value(size_t index, double value) {
    assert(index < _z_list.size());
    _z_list.at(index).set_start(value);
}

void LatentVariables::plot_z(const std::optional<std::vector<size_t>>& indices, size_t width, size_t height,
                             std::string loc) {
    plt::figure_size(width, height);
    for (size_t z = 0; z < _z_list.size(); z++) {
        assert(!_z_list[z].get_sample().has_value() ||
               !(_z_list[z].get_value().has_value() && _z_list[z].get_std().has_value()) &&
                       "No information on latent variable to plot!");
        if (!indices.has_value() ||
            std::find(indices.value().begin(), indices.value().end(), z) == indices.value().end()) {
            std::function<double(double)> transform = _z_list[z].get_prior()->get_transform();
            if (_z_list[z].get_sample().has_value()) {
                std::vector<double> x{&_z_list[z].get_sample().value()[0], _z_list[z].get_sample().value().data()};
                std::transform(x.begin(), x.end(), x.begin(), [transform](double n) { return transform(n); });
                plt::named_plot(_z_list[z].get_method() + " estimate of " + _z_list[z].get_name(), x);
            } else if (_z_list[z].get_value().has_value() && _z_list[z].get_std().has_value()) {
                if (_z_list[z].get_prior()->get_transform_name().empty()) {
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
    plt::legend(std::map<std::string, std::string>{{"loc", loc}});
    plt::save("../data/plot_z.png");
    // plt::show();
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
        Eigen::VectorXd chain{_z_list[i].get_sample().value()};
        for (size_t j = 0; j < 4; j++) {
            size_t iteration = i * 4 + j + 1;
            plt::subplot(static_cast<long>(_z_list.size()), 4, static_cast<long>(iteration));
            if (iteration >= 1 && iteration <= _z_list.size() * 4 + 1) {
                std::function<double(double)> transform = _z_list[i].get_prior()->get_transform();
                if (iteration % 4 == 1) {
                    std::vector<double> x{&chain[0], chain.data()};
                    std::transform(x.begin(), x.end(), x.begin(), [transform](double n) { return transform(n); });
                    plt::plot(x, palette[i]);
                    plt::ylabel(_z_list[i].get_name());
                    if (iteration == 1)
                        plt::title("Density Estimate");
                } else if (iteration % 4 == 2) {
                    std::vector<double> x{&chain[0], chain.data()};
                    std::transform(x.begin(), x.end(), x.begin(), [transform](double n) { return transform(n); });
                    plt::plot(x, palette[i]);
                    if (iteration == 2)
                        plt::title("Trace Plot");
                } else if (iteration % 4 == 3) {
                    std::vector<double> x{&chain[0], chain.data()};
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
                    for (size_t lag = 1; lag < 10 && lag < chain.size();
                         lag++) { // Added lag < chain.size() to avoid index error in acf()
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